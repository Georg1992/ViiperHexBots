"""F12 hotkey registration and polling for the tkinter GUI.

Uses PeekMessageW filtered to only WM_HOTKEY so tkinter's own message
queue (paint, timer, input events) is never drained.
"""

from __future__ import annotations

import ctypes
import tkinter as tk
from collections.abc import Callable
from ctypes import wintypes

user32 = ctypes.windll.user32
WM_HOTKEY = 0x0312
VK_F12 = 0x7B
HOTKEY_ID = 1


class HotkeyManager:
    """Register and poll a global F12 hotkey to toggle the bot.

    Args:
        root: The tkinter root window (used for hotkey binding and polling).
        on_hotkey: Called on the main thread when F12 is pressed.
    """

    def __init__(self, root: tk.Tk, on_hotkey: Callable[[], None]) -> None:
        self._root = root
        self._hwnd = root.winfo_id()
        self._on_hotkey = on_hotkey
        self._register()
        self._root.after(100, self._poll)

    def destroy(self) -> None:
        """Unregister the hotkey. Call on application shutdown."""
        self._unregister()
        try:
            self._root.unbind("<Destroy>", self._unregister)
        except Exception:
            pass

    # ── Internal ────────────────────────────────────────────────────

    def _register(self) -> None:
        user32.RegisterHotKey(self._hwnd, HOTKEY_ID, 0, VK_F12)
        self._root.bind("<Destroy>", self._unregister)

    def _unregister(self, *_event) -> None:
        if not self._hwnd:
            return
        try:
            user32.UnregisterHotKey(self._hwnd, HOTKEY_ID)
            self._hwnd = 0
        except Exception:
            pass

    def _poll(self) -> None:
        """Poll for WM_HOTKEY messages without draining tkinter's queue.

        Filtering msgMin / msgMax to WM_HOTKEY ensures only hotkey
        messages are peeked and removed — all other message types
        (WM_PAINT, WM_TIMER, input events) are left in the queue.
        """
        msg = wintypes.MSG()
        while user32.PeekMessageW(
            ctypes.byref(msg), None, WM_HOTKEY, WM_HOTKEY, 1
        ):
            if msg.wParam == HOTKEY_ID:
                self._on_hotkey()
        self._root.after(100, self._poll)
