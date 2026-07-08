"""Direct VIIPER TCP input backend (replaces HTTP bridge).

Sends binary keyboard/mouse reports directly to the VIIPER server
via persistent TCP device streams, using the native wire protocol.

This eliminates the Go input-bridge entirely.
"""

from __future__ import annotations

import ctypes
import threading
import time

from pybot.runtime.input.input_backend import ShadowInputBackend
from pybot.viiper.client import ViiperClient
from pybot.viiper.keyboard import KeyboardState, vk_to_hid, vk_to_modifier
from pybot.viiper.mouse import MouseState, BUTTON_INDEX_TO_FLAG
from pybot.viiper.stream import DeviceStream

VIIPER_ADDR = "127.0.0.1:3242"
user32 = ctypes.windll.user32

MAPVK_VSC_TO_VK = 1  # MapVirtualKeyW mapping type


def _scan_code_to_vk(scan_code: int) -> int:
    """Convert a Windows scan code to a virtual key code."""
    return user32.MapVirtualKeyW(scan_code, MAPVK_VSC_TO_VK)


class ViiperBackend(ShadowInputBackend):
    """Input backend that sends binary reports directly to VIIPER TCP.

    Opens device streams to the VIIPER server's keyboard and mouse
    devices. Keyboard keys are specified by Windows scan codes
    (converted internally to HID usage codes).

    Mouse movement still uses ``SetCursorPos`` (absolute positioning),
    while button clicks go through the VIIPER mouse stream.
    """

    def __init__(self, addr: str = VIIPER_ADDR) -> None:
        self._addr = addr
        self._api = ViiperClient(addr)
        self._kb_stream: DeviceStream | None = None
        self._mouse_stream: DeviceStream | None = None
        self._mouse_buttons: int = 0  # current button state
        self._mouse_button_left = 0  # button index for left click
        self._connected = False
        self._connect_lock = threading.Lock()

        # Track modifier key state (Ctrl, Shift, Alt, Win)
        self._modifiers: int = 0

    # ── Connection ────────────────────────────────────────────────────

    def connect(self) -> None:
        """Discover and connect to keyboard and mouse device streams.

        Thread-safe: uses a lock to prevent duplicate connections.
        The VIIPER server must already be running with devices created
        (by ViiperManager.start()).
        """
        if self._connected:
            return
        with self._connect_lock:
            if self._connected:
                return
            self._connect_unlocked()

    def _connect_unlocked(self) -> None:
        """Connect to device streams (caller must hold _connect_lock)."""
        # Discover devices on the first available bus
        buses = self._api.bus_list()
        if not buses:
            raise RuntimeError("No VIIPER buses found. Is the server running?")

        bus_id = min(buses)
        devices = self._api.devices_list(bus_id)
        kb_dev_id = ""
        mouse_dev_id = ""

        for dev in devices:
            dev_type = dev.get("type", "")
            if dev_type == "keyboard" and not kb_dev_id:
                kb_dev_id = dev["devId"]
            elif dev_type == "mouse" and not mouse_dev_id:
                mouse_dev_id = dev["devId"]

        if not kb_dev_id or not mouse_dev_id:
            raise RuntimeError(
                "Keyboard or mouse device not found on bus. "
                "Ensure ViiperManager.start() completed first."
            )

        self._kb_stream = DeviceStream.open(self._addr, bus_id, kb_dev_id)
        self._mouse_stream = DeviceStream.open(self._addr, bus_id, mouse_dev_id)
        self._connected = True

    def disconnect(self) -> None:
        """Close device streams."""
        with self._connect_lock:
            if self._kb_stream:
                self._kb_stream.close()
                self._kb_stream = None
            if self._mouse_stream:
                self._mouse_stream.close()
                self._mouse_stream = None
            self._connected = False

    # ── Input methods ─────────────────────────────────────────────────

    def move_mouse(self, x: int, y: int) -> bool:
        """Move the mouse cursor to an absolute screen position.

        Uses Win32 ``SetCursorPos`` directly since VIIPER mouse only
        supports relative movement (deltas), not absolute positioning.
        """
        user32.SetCursorPos(int(x), int(y))
        time.sleep(0.005)
        return True

    def skill_click(self, scan_code: int) -> bool:
        """Press a keyboard key, left-click, then release the key.

        Args:
            scan_code: Windows scan code for the skill key.

        Returns:
            True if successful.
        """
        if scan_code <= 0:
            return False

        self._ensure_connected()

        # Press skill key
        self._key_press(scan_code, down=True)
        time.sleep(0.02)

        # Left click
        self._mouse_click(down=True)
        time.sleep(0.02)
        self._mouse_click(down=False)

        # Release skill key
        self._key_press(scan_code, down=False)
        return True

    def teleport_key(self, scan_code: int) -> bool:
        """Press and release a teleport key.

        Args:
            scan_code: Windows scan code for the teleport key.

        Returns:
            True if successful.
        """
        if scan_code <= 0:
            return False

        self._ensure_connected()

        self._key_press(scan_code, down=True)
        time.sleep(0.05)
        self._key_press(scan_code, down=False)
        return True

    def shutdown(self) -> None:
        """Release pressed keys and close device streams."""
        with self._connect_lock:
            if self._connected and self._kb_stream is not None:
                self._kb_stream.write(KeyboardState(0).marshal())
            self._modifiers = 0
            self.disconnect()

    # ── Low-level helpers ─────────────────────────────────────────────

    def _ensure_connected(self) -> None:
        """Auto-connect on first use (thread-safe)."""
        if not self._connected:
            self.connect()

    def _key_press(self, scan_code: int, down: bool) -> None:
        """Send a key press or release via the keyboard device stream."""
        if self._kb_stream is None:
            return

        vk = _scan_code_to_vk(scan_code)

        # Handle modifier keys
        mod = vk_to_modifier(vk)
        if mod:
            if down:
                self._modifiers |= mod
            else:
                self._modifiers &= ~mod
            state = KeyboardState(self._modifiers)
            self._kb_stream.write(state.marshal())
            return

        # Handle regular keys
        hid = vk_to_hid(vk)
        if not hid:
            return  # Unsupported key

        if down:
            state = KeyboardState.press_key_with_mod(self._modifiers, hid)
            self._kb_stream.write(state.marshal())
        else:
            state = KeyboardState(self._modifiers)
            self._kb_stream.write(state.marshal())

    def _mouse_click(self, down: bool) -> None:
        """Send a mouse button press or release via the mouse device stream."""
        if self._mouse_stream is None:
            return

        flag = BUTTON_INDEX_TO_FLAG.get(self._mouse_button_left, 0x01)
        if down:
            self._mouse_buttons |= flag
        else:
            self._mouse_buttons &= ~flag

        state = MouseState(buttons=self._mouse_buttons)
        self._mouse_stream.write(state.marshal())
