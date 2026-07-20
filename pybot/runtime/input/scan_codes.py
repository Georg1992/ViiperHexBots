"""Map config key names to Windows scan codes."""

from __future__ import annotations

import ctypes
from ctypes import wintypes

user32 = ctypes.windll.user32

MAPVK_VK_TO_VSC = 0


def key_name_to_scan_code(key_name: str) -> int:
    name = (key_name or "").strip()
    if not name:
        return 0
    if len(name) == 1:
        vk = ord(name.upper())
    elif name.lower() == "enter":
        vk = 0x0D
    elif name.lower() == "space":
        vk = 0x20
    elif name.lower() == "tab":
        vk = 0x09
    elif name.lower() == "escape":
        vk = 0x1B
    elif name.lower() == "insert":
        vk = 0x2D
    else:
        return 0
    scan_code = user32.MapVirtualKeyW(vk, MAPVK_VK_TO_VSC)
    return int(scan_code) if scan_code else 0
