"""VIIPER server lifecycle manager (replaces Go input-bridge).

Launches the VIIPER server (viiper.exe) directly as a subprocess,
sets up virtual keyboard + mouse devices, and holds persistent
device streams to keep them alive.

VIIPER auto-removes devices if no stream connects within 5 seconds
of creation, and again ~5s after a stream disconnects. The manager
holds streams after add; the hunt input backend must keep its streams
open across Stop/Start (see ViiperBackend) so devices are not removed.
``ensure_devices`` recreates keyboard/mouse if they were already lost.

This replaces the old Go bridge (viiper-input.exe) entirely.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Callable

from pybot.viiper.client import ViiperClient, ViiperError
from pybot.viiper.stream import DeviceStream
from pybot.paths import PROJECT_ROOT

VIIPER_ADDR = "127.0.0.1:3242"
LogFn = Callable[[str], None]


class ViiperManager:
    """Manages the VIIPER server process and virtual input devices.

    Usage::

        mgr = ViiperManager(on_log=print, on_status=lambda t, h: None)
        mgr.start()   # launches viiper.exe, creates bus + keyboard + mouse
        ...
        mgr.shutdown()  # stops viiper.exe
    """

    def __init__(
        self,
        *,
        project_root: Path | None = None,
        stream_store=None,
        on_log: LogFn | None = None,
        on_status: Callable[[str, str], None] | None = None,
    ) -> None:
        self._root = PROJECT_ROOT if project_root is None else project_root
        # One process-wide stream store is shared with the hunt input backend
        # so device streams survive Stop/Start and are closed on app exit.
        if stream_store is None:
            # Lazy import keeps the backend (and its Win32 ctypes bindings)
            # out of import time for app modules that never touch input.
            from pybot.runtime.input.viiper_backend import ViiperStreamStore

            stream_store = ViiperStreamStore()
        self._stream_store = stream_store
        self._on_log = (lambda _msg: None) if on_log is None else on_log
        self._on_status = (
            (lambda _title, _hint: None) if on_status is None else on_status
        )

        # VIIPER server state
        self._server_proc: subprocess.Popen | None = None
        self._shutdown_done = False

        # TCP API client
        self._api = ViiperClient(VIIPER_ADDR)

        # Bus and device info (populated by start())
        self.bus_id: int = 0
        self.mouse_dev_id: str = ""

        # Persistent device streams (keep devices alive)
        self._kb_stream: DeviceStream | None = None
        self._mouse_stream: DeviceStream | None = None

    # ── Public API ────────────────────────────────────────────────────

    def start(self) -> None:
        """Launch VIIPER server and set up virtual keyboard + mouse.

        Opens and holds persistent device streams to prevent the 5-second
        auto-removal timeout.

        Raises:
            FileNotFoundError: viiper.exe not found.
            RuntimeError: Server failed to start or devices could not be created.
        """
        # Check if already running
        if self._server_ready():
            self._log("Virtual keyboard and mouse already ready")
            self._on_status(
                "Input: Ready",
                "Virtual keyboard and mouse active — launch the game now",
            )
            return

        # Launch server process
        viiper_path = self._find_viiper_exe()
        self._log("Launching viiper.exe...")
        self._on_status("Input: Launching server...", "")

        proc = subprocess.Popen(
            [str(viiper_path), "server"],
            cwd=str(viiper_path.parent),
            creationflags=subprocess.CREATE_NO_WINDOW,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env={**os.environ, "VIIPER_LOG_LEVEL": "error"},
        )
        self._server_proc = proc

        if not self._wait_for_server():
            self._kill_server()
            raise RuntimeError(
                "VIIPER server failed to start. Make sure usbip-win2 "
                "is installed and reboot if needed."
            )

        self._log("VIIPER server ready")
        self._on_status("Input: Creating devices...", "")

        # Set up bus and devices (opens & holds streams to keep alive). If
        # device creation fails after this manager launched the server, clean
        # up the owned process before surfacing the original error.
        try:
            self._setup_devices()
        except BaseException:
            self._stream_store.close()
            self._close_streams()
            self._kill_server()
            raise

        self._log("Virtual keyboard and mouse ready")
        self._on_status("Input: Ready", "Virtual keyboard and mouse active — launch the game now")

    def ensure_devices(self) -> None:
        """Recreate keyboard/mouse if VIIPER auto-removed them.

        Closing a device stream starts VIIPER's ~5s removal timer. Hunt stop
        must keep streams open; this repairs the bus if devices were already
        lost (or the server was restarted under us).
        """
        if not self._server_ready():
            raise RuntimeError(
                "VIIPER server is not running. Restart the bot application."
            )
        buses = self._api.bus_list()
        if not buses:
            self._log("No VIIPER bus — recreating keyboard and mouse...")
            self._stream_store.close()
            self._close_streams()
            self._setup_devices()
            return

        bus_id = min(buses)
        devices = self._api.devices_list(bus_id)
        types = {str(dev.get("type", "")) for dev in devices}
        streams_ok = self._kb_stream is not None and self._mouse_stream is not None
        if "keyboard" in types and "mouse" in types and streams_ok:
            self.bus_id = bus_id
            return

        self._log("Virtual keyboard/mouse missing — recreating...")
        self._stream_store.close()
        self._close_streams()
        self._setup_devices()

    def shutdown(self) -> None:
        """Gracefully stop the VIIPER server."""
        if self._shutdown_done:
            return
        self._shutdown_done = True
        self._log("Stopping virtual keyboard and mouse...")
        self._stream_store.close()
        self._close_streams()
        self._kill_server()
        self._log("VIIPER stopped")

    # ── Internals ─────────────────────────────────────────────────────

    def _find_viiper_exe(self) -> Path:
        """Locate the viiper.exe binary.

        Search order:
        1. ``VIIPER/dist/viiper.exe`` (direct submodule build output)
        """
        candidates = [
            self._root / "VIIPER" / "dist" / "viiper.exe",
        ]
        for path in candidates:
            if path.is_file():
                return path
        raise FileNotFoundError(
            f"Could not find viiper.exe. Run build.ps1 first.\n"
            f"  Searched: {candidates[0]}"
        )

    def _server_ready(self) -> bool:
        """Check if the VIIPER server is already responding to ping."""
        try:
            resp = self._api.ping()
            return bool(resp.get("server"))
        except (ConnectionRefusedError, OSError, TimeoutError, ViiperError):
            return False

    def _wait_for_server(self, timeout_s: float = 30.0) -> bool:
        """Poll the VIIPER server until it responds to ping."""
        deadline = time.monotonic() + timeout_s
        last_status = 0.0
        while time.monotonic() < deadline:
            if self._server_ready():
                return True
            now = time.monotonic()
            if now - last_status > 3.0:
                self._log("Waiting for VIIPER server...")
                last_status = now
            time.sleep(0.2)
        return False

    def _setup_devices(self) -> None:
        """Create a bus, add keyboard + mouse devices, hold streams.

        Opening and holding device streams is required because VIIPER
        auto-removes devices if no stream connects within 5 seconds. Device
        setup is transactional: a failed second device cannot leak the first
        TCP stream into the next retry.
        """
        api = self._api
        kb_dev_id = ""
        mouse_dev_id = ""
        kb_stream = None
        mouse_stream = None
        try:
            # List existing buses
            buses = api.bus_list()

            if buses:
                self.bus_id = min(buses)
                self._log(f"Using existing bus {self.bus_id}")

                # Clean up any existing devices on this bus
                devices = api.devices_list(self.bus_id)
                for dev in devices:
                    try:
                        api.device_remove(self.bus_id, dev["devId"])
                    except ViiperError:
                        pass
            else:
                self.bus_id = api.bus_create()
                self._log(f"Created bus {self.bus_id}")

            # Add keyboard device and open stream to keep it alive
            kb_resp = api.device_add(self.bus_id, "keyboard")
            kb_dev_id = kb_resp["devId"]
            kb_stream = DeviceStream.open(
                VIIPER_ADDR, self.bus_id, kb_dev_id
            )
            self._log(f"Keyboard added: bus={self.bus_id} dev={kb_dev_id}")

            # Add mouse device and open stream to keep it alive
            mouse_resp = api.device_add(self.bus_id, "mouse")
            mouse_dev_id = mouse_resp["devId"]
            mouse_stream = DeviceStream.open(
                VIIPER_ADDR, self.bus_id, mouse_dev_id
            )
        except BaseException:
            for stream in (mouse_stream, kb_stream):
                if stream is not None:
                    try:
                        stream.close()
                    except BaseException:
                        pass
            for dev_id in (mouse_dev_id, kb_dev_id):
                if dev_id:
                    try:
                        api.device_remove(self.bus_id, dev_id)
                    except BaseException:
                        pass
            raise

        try:
            self._stream_store.adopt(kb_stream, mouse_stream)
        except BaseException:
            for stream in (mouse_stream, kb_stream):
                try:
                    stream.close()
                except BaseException:
                    pass
            raise
        self.mouse_dev_id = mouse_dev_id
        self._kb_stream = kb_stream
        self._mouse_stream = mouse_stream
        self._log(f"Mouse added: bus={self.bus_id} dev={mouse_dev_id}")

    def _close_streams(self) -> None:
        """Close held device streams."""
        if self._kb_stream:
            try:
                self._kb_stream.close()
            except Exception:
                pass
            self._kb_stream = None
        if self._mouse_stream:
            try:
                self._mouse_stream.close()
            except Exception:
                pass
            self._mouse_stream = None

    def _kill_server(self) -> None:
        """Kill the VIIPER server process (fire-and-forget)."""
        if self._server_proc is not None:
            pid = self._server_proc.pid
            try:
                subprocess.Popen(
                    ["taskkill", "/PID", str(pid), "/T", "/F"],
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
            except OSError:
                pass
            self._server_proc = None
            return
        # No tracked process means the manager connected to an already-running
        # server. Never kill an unrelated viiper.exe by process name.

    def _log(self, message: str) -> None:
        self._on_log(message)
