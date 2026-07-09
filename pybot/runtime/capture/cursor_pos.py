"""Screen cursor position for vision isolation during local tracking."""

from __future__ import annotations

import ctypes
from ctypes import wintypes

user32 = ctypes.windll.user32


def get_cursor_screen_position() -> tuple[int, int] | None:
    """Return cursor (x, y) in screen coordinates, or None if unavailable."""
    point = wintypes.POINT()
    if not user32.GetCursorPos(ctypes.byref(point)):
        return None
    return int(point.x), int(point.y)
