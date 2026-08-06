#!/usr/bin/env python3
"""Periodic sit-key helper.

Presses the configured sit key once every 10 seconds by default, through
the exact same VIIPER virtual keyboard the hunt bot uses
(``ViiperBackend.toggle_key`` — the method the sit-on-low-SP worker calls).

Runs standalone: starts the VIIPER virtual input server when it is not
already running, or reuses the existing keyboard/mouse devices otherwise.

The game window must be in the foreground — VIIPER is a virtual USB
keyboard, so keystrokes go to the focused window just like a real one.

Examples:
    python scripts/sit_toggle.py                 # config.ini sit key, 10s
    python scripts/sit_toggle.py --button f2     # explicit key name
    python scripts/sit_toggle.py --interval 30   # every 30 seconds

Stop with Ctrl+C.
"""

from __future__ import annotations

import argparse
import ctypes
import sys
import time

from pybot.app.viiper_manager import ViiperManager
from pybot.config.ini_store import load_settings
from pybot.runtime.input.scan_codes import key_name_to_scan_code
from pybot.runtime.input.viiper_backend import ViiperBackend, ViiperStreamStore
from pybot.viiper.client import ViiperClient, ViiperError
from pybot.viiper.keyboard import vk_to_hid

# MapVirtualKeyW type for scan-code → virtual-key conversion.
MAPVK_VSC_TO_VK = 1

VIIPER_ADDR = "127.0.0.1:3242"


def _server_running() -> bool:
    api = ViiperClient(VIIPER_ADDR, timeout_s=1.0)
    try:
        return bool(api.ping().get("server"))
    except (ConnectionRefusedError, OSError, TimeoutError, ViiperError):
        return False


def _devices_present() -> bool:
    """True when the VIIPER bus already has keyboard + mouse devices."""
    api = ViiperClient(VIIPER_ADDR, timeout_s=2.0)
    try:
        buses = api.bus_list()
        if not buses:
            return False
        devices = api.devices_list(min(buses))
    except (ConnectionRefusedError, OSError, TimeoutError, ViiperError):
        return False
    types = {str(dev.get("type", "")) for dev in devices}
    return "keyboard" in types and "mouse" in types


def _resolve_sit_scan(button: str | None) -> tuple[str, int]:
    """Resolve the sit key name to a scan code (config.ini when not given)."""
    if not button:
        try:
            settings = load_settings()
            button = settings.sit_on_low_sp_button
        except Exception:
            button = ""
    button = (button or "insert").strip() or "insert"
    scan = key_name_to_scan_code(button)
    # A key with a scan code but no HID usage (e.g. media/volume keys) would
    # make toggle_key return True while _key_press silently sends nothing.
    if scan > 0:
        vk = int(ctypes.windll.user32.MapVirtualKeyW(scan, MAPVK_VSC_TO_VK))
        if vk_to_hid(vk) == 0:
            return button, 0
    return button, scan


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Press the sit key on a fixed cadence via VIIPER.",
    )
    ap.add_argument(
        "--button",
        default=None,
        help="Sit key name (default: SitOnLowSpButton from config.ini)",
    )
    ap.add_argument(
        "--interval",
        type=float,
        default=10.0,
        help="Seconds between presses (default: 10)",
    )
    args = ap.parse_args(argv)

    if args.interval <= 0:
        ap.error("--interval must be > 0")

    button, scan = _resolve_sit_scan(args.button)
    if scan <= 0:
        print(f"[sit-tap] unsupported sit key: {button!r}", file=sys.stderr)
        return 2

    store = ViiperStreamStore()
    mgr = ViiperManager(
        stream_store=store,
        on_log=lambda msg: print(f"[sit-tap] {msg}"),
    )
    try:
        if not _server_running():
            mgr.start()  # launches viiper.exe and creates keyboard + mouse
        # Verify devices after the start attempt (covers the race where the
        # server came up between the ping above and start()). Only recreate
        # when they are truly missing — never disturb devices another
        # process (e.g. the main app) is already holding.
        if not _devices_present():
            mgr.ensure_devices()
    except (FileNotFoundError, RuntimeError, ViiperError) as exc:
        print(f"[sit-tap] input setup failed: {exc}", file=sys.stderr)
        mgr.shutdown()
        return 1

    backend = ViiperBackend(stream_store=store)
    try:
        backend.connect()
        # Re-arm input exactly like the hunt runtime does at start
        # (HuntRuntime._begin_input_session) before the first press.
        if not backend.begin_session():
            raise RuntimeError("input session could not be re-armed")
        kb = backend._kb_stream
        mouse = backend._mouse_stream
        if kb is None or mouse is None:
            raise RuntimeError("keyboard/mouse streams not open after connect")
    except RuntimeError as exc:
        print(f"[sit-tap] cannot connect to virtual input: {exc}", file=sys.stderr)
        mgr.shutdown()
        return 1

    print(
        f"[sit-tap] sit key={button!r}"
        f"{' (from config.ini)' if args.button is None else ''} — "
        f"pressing every {args.interval:g}s via VIIPER "
        f"(bus={kb.bus_id} kb_dev={kb.dev_id}). "
        "Keep the game window focused. Ctrl+C to stop."
    )
    try:
        while True:
            if not backend.wait_interruptible(args.interval):
                break
            try:
                ok = backend.toggle_key(scan)
            except (OSError, RuntimeError) as exc:
                print(f"[sit-tap] {time.strftime('%H:%M:%S')} press FAILED: {exc}")
                continue
            print(
                f"[sit-tap] {time.strftime('%H:%M:%S')} "
                f"pressed {button!r}: {'ok' if ok else 'FAILED'}"
            )
    except KeyboardInterrupt:
        print("\n[sit-tap] stopping")
    finally:
        backend.shutdown()
        mgr.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
