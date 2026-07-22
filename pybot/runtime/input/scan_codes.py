"""Map config key names to Windows scan codes."""

from __future__ import annotations

import ctypes
from ctypes import wintypes

user32 = ctypes.windll.user32

MAPVK_VK_TO_VSC = 0

_NAMED_KEY_VK: dict[str, int] = {
    "enter": 0x0D,
    "space": 0x20,
    "tab": 0x09,
    "escape": 0x1B,
    "insert": 0x2D,
    "up": 0x26,
    "down": 0x28,
    "left": 0x25,
    "right": 0x27,
    "f1": 0x70,
    "f2": 0x71,
    "f3": 0x72,
    "f4": 0x73,
    "f5": 0x74,
    "f6": 0x75,
    "f7": 0x76,
    "f8": 0x77,
    "f9": 0x78,
    "f10": 0x79,
    "f11": 0x7A,
    "f12": 0x7B,
}


def keysym_to_key_name(keysym: str) -> str:
    """Map a Tk ``event.keysym`` to a config key name, or empty if unsupported."""
    if not keysym:
        return ""
    lower = keysym.lower()
    if lower == "return":
        return "enter"
    if lower in _NAMED_KEY_VK:
        return lower
    if len(keysym) == 1:
        return keysym.lower()
    return ""


def key_name_to_scan_code(key_name: str) -> int:
    name = (key_name or "").strip()
    if not name:
        return 0
    if len(name) == 1:
        vk = ord(name.upper())
    else:
        vk = _NAMED_KEY_VK.get(name.lower(), 0)
        if vk == 0:
            return 0
    scan_code = user32.MapVirtualKeyW(vk, MAPVK_VK_TO_VSC)
    return int(scan_code) if scan_code else 0
