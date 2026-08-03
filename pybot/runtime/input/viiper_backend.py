"""Direct VIIPER TCP input backend (replaces HTTP bridge).

Sends binary keyboard/mouse reports directly to the VIIPER server
via persistent TCP device streams, using the native wire protocol.

This eliminates the Go input-bridge entirely.

Stream lifetime
---------------
VIIPER removes a virtual device ~5s after its last stream disconnects.
Hunt stop therefore must **not** close these streams — only release keys —
or a later Start finds no keyboard/mouse. Streams are process-wide and
closed only on app exit via ``close_shared_streams``.
"""

from __future__ import annotations

import ctypes
import threading

from pybot.runtime.constants import (
    ALT_MOUSE_CLICK_DELAY_S,
)
from pybot.runtime.input.input_backend import ShadowInputBackend
from pybot.runtime.input.scan_codes import key_name_to_scan_code
from pybot.viiper.client import ViiperClient
from pybot.viiper.keyboard import KeyboardState, vk_to_hid, vk_to_modifier
from pybot.viiper.mouse import MouseState, BUTTON_INDEX_TO_FLAG
from pybot.viiper.stream import DeviceStream

VIIPER_ADDR = "127.0.0.1:3242"
user32 = ctypes.windll.user32

MAPVK_VSC_TO_VK = 1  # MapVirtualKeyW mapping type
# AHK Alt / E for inventory toggle (ManageInventoryWindow).
SCAN_CODE_ALT = 56
SCAN_CODE_E = 18
MOUSE_BUTTON_LEFT = 0
MOUSE_BUTTON_RIGHT = 1

_shared_lock = threading.Lock()
# All backend instances may share the same process-wide VIIPER streams.
# Serialise operations globally so overlapping lifecycle threads cannot
# interleave keyboard/mouse reports even during a stop/start race.
_shared_operation_lock = threading.Lock()
_shared_cancel_event = threading.Event()
_shared_kb: DeviceStream | None = None
_shared_mouse: DeviceStream | None = None


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
        self._operation_lock = _shared_operation_lock
        # Set by runtime stop before shutdown joins workers. Long input
        # sequences observe it and release their keys/buttons promptly.
        self._cancel_event = _shared_cancel_event

        # Track modifier key state (Ctrl, Shift, Alt, Win)
        self._modifiers: int = 0

    # ── Connection ────────────────────────────────────────────────────

    def connect(self) -> None:
        """Discover and connect to keyboard and mouse device streams.

        Thread-safe: uses a lock to prevent duplicate connections.
        The VIIPER server must already be running with devices created
        (by ViiperManager.start()). Reuses process-wide streams so hunt
        stop/start does not trigger VIIPER device auto-removal.
        """
        if self._connected:
            return
        with self._connect_lock:
            if self._connected:
                return
            self._connect_unlocked()

    def _connect_unlocked(self) -> None:
        """Connect to device streams (caller must hold _connect_lock)."""
        global _shared_kb, _shared_mouse
        with _shared_lock:
            if _shared_kb is not None and _shared_mouse is not None:
                self._kb_stream = _shared_kb
                self._mouse_stream = _shared_mouse
                self._connected = True
                return

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

            _shared_kb = DeviceStream.open(self._addr, bus_id, kb_dev_id)
            _shared_mouse = DeviceStream.open(self._addr, bus_id, mouse_dev_id)
            self._kb_stream = _shared_kb
            self._mouse_stream = _shared_mouse
            self._connected = True

    def disconnect(self) -> None:
        """Drop this instance's handles without closing shared streams.

        Closing the TCP streams starts VIIPER's ~5s device-removal timer.
        Use ``close_shared_streams`` only on application exit.
        """
        with self._connect_lock:
            self._kb_stream = None
            self._mouse_stream = None
            self._connected = False

    @staticmethod
    def close_shared_streams() -> None:
        """Close process-wide device streams (application shutdown only)."""
        global _shared_kb, _shared_mouse
        _shared_cancel_event.set()
        with _shared_operation_lock:
            with _shared_lock:
                if _shared_kb is not None:
                    try:
                        _shared_kb.close()
                    except OSError:
                        pass
                    _shared_kb = None
                if _shared_mouse is not None:
                    try:
                        _shared_mouse.close()
                    except OSError:
                        pass
                    _shared_mouse = None

    # ── Input methods ─────────────────────────────────────────────────

    def begin_session(self) -> bool:
        """Re-arm input after the previous operation has fully unwound.

        Pause sets the process-wide cancellation event. Acquire the same
        operation lock before clearing it so a canceled storage/drag operation
        cannot race with resume and send fresh input after the pause boundary.
        The bounded acquisition keeps a Tk/UI resume callback responsive.
        """
        if not self._operation_lock.acquire(timeout=0.25):
            return False
        try:
            self._cancel_event.clear()
        finally:
            self._operation_lock.release()
        return True

    def cancel_pending(self) -> None:
        """Interrupt a current/queued input operation without closing streams."""
        self._cancel_event.set()

    def _wait_or_cancel(self, seconds: float) -> bool:
        """Wait interruptibly; return False when runtime shutdown was requested."""
        if seconds <= 0:
            return not self._cancel_event.is_set()
        return not self._cancel_event.wait(seconds)

    def wait_interruptible(self, seconds: float) -> bool:
        """Wait for UI settling without hiding cancellation behind sleep."""
        return self._wait_or_cancel(seconds)

    def move_mouse(self, x: int, y: int) -> bool:
        """Move the mouse cursor to an absolute screen position.

        Uses Win32 ``SetCursorPos`` directly since VIIPER mouse only
        supports relative movement (deltas), not absolute positioning.
        """
        with self._operation_lock:
            user32.SetCursorPos(int(x), int(y))
            if not self._wait_or_cancel(0.005):
                return False
        return True

    def move_and_click(self, x: int, y: int) -> bool:
        """Move and click atomically under the input operation lock."""
        with self._operation_lock:
            self._ensure_connected()
            user32.SetCursorPos(int(x), int(y))
            if not self._wait_or_cancel(0.005):
                return False
            return self._left_click_locked()

    def move_and_double_click(self, x: int, y: int) -> bool:
        """Move and double-click atomically, with no post-click delay.

        Kiting must hand control back to the attack loop immediately after the
        walk command. Both clicks stay under the shared operation lock so a
        buff/heal/storage worker cannot move the cursor between them.
        """
        with self._operation_lock:
            self._ensure_connected()
            user32.SetCursorPos(int(x), int(y))
            if not self._wait_or_cancel(0.005):
                return False
            if not self._left_click_locked():
                return False
            return self._left_click_locked()

    def _left_click_locked(self) -> bool:
        """Emit one left click; caller holds the shared operation lock."""
        if self._cancel_event.is_set():
            return False
        self._mouse_button(MOUSE_BUTTON_LEFT, down=True)
        try:
            return self._wait_or_cancel(0.05)
        finally:
            self._mouse_button(MOUSE_BUTTON_LEFT, down=False)

    def skill_click(self, scan_code: int) -> bool:
        """Press a keyboard key, left-click, then release the key.

        Args:
            scan_code: Windows scan code for the skill key.

        Returns:
            True if successful.
        """
        if scan_code <= 0:
            return False

        with self._operation_lock:
            if self._cancel_event.is_set():
                return False
            self._ensure_connected()
            return self._skill_click_locked(scan_code)

    def skill_click_at(
        self,
        scan_code: int,
        x: int,
        y: int,
        *,
        move_delay_s: float = 0.05,
    ) -> bool:
        """Move cursor to ``(x, y)``, wait, then skill-key + left-click.

        Atomic so nothing can steal the cursor between move and click. The
        optional movement delay lets startup self-buffs use their deliberate
        200 ms cursor-settle window without changing attack/debuff timing.
        """
        if scan_code <= 0:
            return False

        with self._operation_lock:
            if self._cancel_event.is_set():
                return False
            self._ensure_connected()
            user32.SetCursorPos(int(x), int(y))
            if not self._wait_or_cancel(move_delay_s):
                return False
            return self._skill_click_locked(scan_code)

    def _skill_click_locked(self, scan_code: int) -> bool:
        """Skill key down → left click → key up. Caller holds the shared lock."""
        self._key_press(scan_code, down=True)
        try:
            if not self._wait_or_cancel(0.05):
                return False
            self._mouse_button(MOUSE_BUTTON_LEFT, down=True)
            try:
                return self._wait_or_cancel(0.05)
            finally:
                self._mouse_button(MOUSE_BUTTON_LEFT, down=False)
        finally:
            self._key_press(scan_code, down=False)

    def teleport_key(self, scan_code: int) -> bool:
        """Press and release a teleport key.

        Args:
            scan_code: Windows scan code for the teleport key.

        Returns:
            True if successful.
        """
        return self.key_tap(scan_code, press_s=0.05, after_s=0.0)

    def left_click(self) -> bool:
        """AHK ``AHIclick``: left button down 50ms, then up."""
        with self._operation_lock:
            if self._cancel_event.is_set():
                return False
            self._ensure_connected()
            self._left_click_locked()
        return not self._cancel_event.is_set()

    def right_click(self) -> bool:
        """Right button down 50ms, then up."""
        with self._operation_lock:
            if self._cancel_event.is_set():
                return False
            self._ensure_connected()
            self._mouse_button(MOUSE_BUTTON_RIGHT, down=True)
            try:
                self._wait_or_cancel(0.05)
            finally:
                self._mouse_button(MOUSE_BUTTON_RIGHT, down=False)
        return not self._cancel_event.is_set()

    def set_left_button(self, down: bool) -> bool:
        """Press or release the left mouse button (for drag).

        A cancelled session must still be allowed to send the release report.
        Otherwise a drag interrupted by focus loss can leave the desktop mouse
        button logically held until the next full shutdown.
        """
        with self._operation_lock:
            if self._cancel_event.is_set() and down:
                return False
            self._ensure_connected()
            self._mouse_button(MOUSE_BUTTON_LEFT, down=down)
        return not self._cancel_event.is_set() if down else True

    def alt_right_click(self) -> bool:
        """Alt+RMB once, then always wait ``ALT_MOUSE_CLICK_DELAY_S`` (100ms)."""
        click_ok = False
        with self._operation_lock:
            self._ensure_connected()
            if self._cancel_event.is_set():
                return False
            self._key_press(SCAN_CODE_ALT, down=True)
            try:
                if not self._wait_or_cancel(0.05):
                    return False
                self._mouse_button(MOUSE_BUTTON_RIGHT, down=True)
                try:
                    click_ok = self._wait_or_cancel(0.05)
                finally:
                    self._mouse_button(MOUSE_BUTTON_RIGHT, down=False)
            finally:
                self._key_press(SCAN_CODE_ALT, down=False)
        if not click_ok:
            return False
        return self._wait_or_cancel(ALT_MOUSE_CLICK_DELAY_S)

    def alt_right_clicks(self, times: int = 1) -> bool:
        """AHK ``AltClicks``: Alt+RMB × N with 100ms after each click."""
        if times <= 0:
            return False
        for _ in range(times):
            if not self.alt_right_click():
                return False
        return True

    def key_tap(
        self,
        scan_code: int,
        *,
        press_s: float = 0.05,
        after_s: float = 0.30,
    ) -> bool:
        """Press and release a key with AHK ``SendKeyCombo``-style timing."""
        if scan_code <= 0:
            return False
        with self._operation_lock:
            if self._cancel_event.is_set():
                return False
            self._ensure_connected()
            self._key_press(scan_code, down=True)
            try:
                self._wait_or_cancel(press_s)
            finally:
                self._key_press(scan_code, down=False)
            if after_s > 0:
                self._wait_or_cancel(after_s)
        return not self._cancel_event.is_set()

    def toggle_key(self, scan_code: int) -> bool:
        """Press/release a toggle key and report whether it was emitted.

        Cancellation before key-down rejects the operation. Cancellation during
        the press or after-key timing does not turn an already-emitted toggle
        into a false result, which keeps sit/stand state synchronized.
        """
        if scan_code <= 0:
            return False
        with self._operation_lock:
            if self._cancel_event.is_set():
                return False
            self._ensure_connected()
            self._key_press(scan_code, down=True)
            try:
                self._wait_or_cancel(0.05)
            finally:
                self._key_press(scan_code, down=False)
        return True

    def cleanup_toggle_key(self, scan_code: int) -> bool:
        """Emit one final toggle after normal input cancellation.

        Shutdown cancellation must stop long-running actions, but it must not
        prevent the sit worker from undoing a toggle that was already accepted.
        This operation is deliberately limited to the key down/up pair and
        always releases the key before returning.
        """
        if scan_code <= 0:
            return False
        if not self._operation_lock.acquire(timeout=0.25):
            return False
        try:
            self._ensure_connected()
            self._key_press(scan_code, down=True)
            try:
                self._wait_or_cancel(0.05)
            finally:
                self._key_press(scan_code, down=False)
        finally:
            self._operation_lock.release()
        return True

    def type_text(self, text: str) -> bool:
        """Type printable characters (digits/letters) via scan codes."""
        if not text:
            return False
        with self._operation_lock:
            self._ensure_connected()
            for ch in text:
                if self._cancel_event.is_set():
                    return False
                scan = key_name_to_scan_code(ch)
                if scan <= 0:
                    return False
                self._key_press(scan, down=True)
                try:
                    self._wait_or_cancel(0.05)
                finally:
                    self._key_press(scan, down=False)
                if not self._wait_or_cancel(0.05):
                    return False
        return not self._cancel_event.is_set()

    def toggle_inventory(self) -> bool:
        """AHK ``ManageInventoryWindow``: Alt+E with the same sleeps."""
        with self._operation_lock:
            if self._cancel_event.is_set():
                return False
            self._ensure_connected()
            self._key_press(SCAN_CODE_ALT, down=True)
            try:
                if not self._wait_or_cancel(0.05):
                    return False
                if self._cancel_event.is_set():
                    return False
                self._key_press(SCAN_CODE_E, down=True)
                try:
                    self._wait_or_cancel(0.05)
                finally:
                    self._key_press(SCAN_CODE_E, down=False)
            finally:
                self._key_press(SCAN_CODE_ALT, down=False)
            self._wait_or_cancel(0.50)
        return not self._cancel_event.is_set()

    def play_key_chain(
        self, steps: tuple[tuple[str, int, int], ...]
    ) -> bool:
        """Play Open Storage chain under one operation lock."""
        if not steps:
            return False
        with self._operation_lock:
            self._ensure_connected()
            for _button, scan_code, delay_ms in steps:
                if self._cancel_event.is_set() or scan_code <= 0:
                    return False
                self._key_press(scan_code, down=True)
                try:
                    self._wait_or_cancel(0.05)
                finally:
                    self._key_press(scan_code, down=False)
                if delay_ms > 0 and not self._wait_or_cancel(delay_ms / 1000.0):
                    return False
        return not self._cancel_event.is_set()

    def shutdown(self) -> bool:
        """Cancel pending input and release keys/buttons.

        Returns False if another operation still owns the shared input lock;
        callers must not treat that as a clean input shutdown.
        """
        self.cancel_pending()
        acquired = self._operation_lock.acquire(timeout=2.0)
        if not acquired:
            return False
        try:
            with self._connect_lock:
                if self._connected and self._kb_stream is not None:
                    try:
                        self._kb_stream.write(KeyboardState(0).marshal())
                    except (OSError, RuntimeError):
                        pass
                if self._connected and self._mouse_stream is not None:
                    try:
                        self._mouse_stream.write(MouseState(buttons=0).marshal())
                    except (OSError, RuntimeError):
                        pass
                self._modifiers = 0
                self._mouse_buttons = 0
                # Intentionally do not close streams — see module docstring.
        finally:
            self._operation_lock.release()
        return True

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

    def _mouse_button(self, button_index: int, down: bool) -> None:
        """Send a mouse button press or release via the mouse device stream."""
        if self._mouse_stream is None:
            return

        flag = BUTTON_INDEX_TO_FLAG.get(button_index, 0)
        if not flag:
            return
        if down:
            self._mouse_buttons |= flag
        else:
            self._mouse_buttons &= ~flag

        state = MouseState(buttons=self._mouse_buttons)
        self._mouse_stream.write(state.marshal())
