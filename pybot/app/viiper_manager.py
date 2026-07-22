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
        on_log: LogFn | None = None,
        on_status: Callable[[str, str], None] | None = None,
    ) -> None:
        self._root = project_root or PROJECT_ROOT
        self._on_log = on_log or (lambda _msg: None)
        self._on_status = on_status or (lambda _title, _hint: None)

        # VIIPER server state
        self._server_proc: subprocess.Popen | None = None
        self._shutdown_done = False

        # TCP API client
        self._api = ViiperClient(VIIPER_ADDR)

        # Bus and device info (populated by start())
        self.bus_id: int = 0
        self.keyboard_dev_id: str = ""
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
            self._on_status("Input: Ready", "Virtual keyboard and mouse active — launch the game now")
            return

        # Launch server process
        viiper_path = self._find_viiper_exe()
        self._log("Launching viiper.exe...")
        self._on_status("Input: Launching server...")

        proc = subprocess.Popen(
            [str(viiper_path), "server"],
            cwd=str(viiper_path.parent),
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        self._server_proc = proc

        if not self._wait_for_server():
            self._kill_server()
            raise RuntimeError(
                "VIIPER server failed to start. Make sure usbip-win2 "
                "is installed and reboot if needed."
            )

        self._log("VIIPER server ready")
        self._on_status("Input: Creating devices...")

        # Set up bus and devices (opens & holds streams to keep alive)
        self._setup_devices()

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
            from pybot.runtime.input.viiper_backend import ViiperBackend

            ViiperBackend.close_shared_streams()
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
        from pybot.runtime.input.viiper_backend import ViiperBackend

        ViiperBackend.close_shared_streams()
        self._close_streams()
        self._setup_devices()

    def shutdown(self) -> None:
        """Gracefully stop the VIIPER server."""
        if self._shutdown_done:
            return
        self._shutdown_done = True
        self._log("Stopping virtual keyboard and mouse...")
        from pybot.runtime.input.viiper_backend import ViiperBackend

        ViiperBackend.close_shared_streams()
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
        auto-removes devices if no stream connects within 5 seconds.
        """
        api = self._api

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
        self.keyboard_dev_id = kb_resp["devId"]
        self._kb_stream = DeviceStream.open(
            VIIPER_ADDR, self.bus_id, self.keyboard_dev_id
        )
        self._log(f"Keyboard added: bus={self.bus_id} dev={self.keyboard_dev_id}")

        # Add mouse device and open stream to keep it alive
        mouse_resp = api.device_add(self.bus_id, "mouse")
        self.mouse_dev_id = mouse_resp["devId"]
        self._mouse_stream = DeviceStream.open(
            VIIPER_ADDR, self.bus_id, self.mouse_dev_id
        )
        self._log(f"Mouse added: bus={self.bus_id} dev={self.mouse_dev_id}")

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
        # Fallback: kill any viiper.exe
        try:
            subprocess.Popen(
                ["taskkill", "/IM", "viiper.exe", "/F"],
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
        except OSError:
            pass

    def _log(self, message: str) -> None:
        self._on_log(message)
