"""HID keyboard constants and wire protocol for VIIPER device streams.

Wire format (client → server)::
    modifiers:u8 count:u8 keys:u8*count

    Byte 0: Modifier bitmask
    Byte 1: Number of pressed keys
    Bytes 2+: HID usage codes of pressed keys

Wire format (server → client)::
    leds:u8
"""

from __future__ import annotations

import struct

# ── Modifier key bitmasks ──────────────────────────────────────────────
MOD_LEFT_CTRL = 0x01
MOD_LEFT_SHIFT = 0x02
MOD_LEFT_ALT = 0x04
MOD_LEFT_GUI = 0x08
MOD_RIGHT_CTRL = 0x10
MOD_RIGHT_SHIFT = 0x20
MOD_RIGHT_ALT = 0x40
MOD_RIGHT_GUI = 0x80

# ── HID Usage codes (USB HID Keyboard/Keypad usage page) ──────────────
KEY_A = 0x04
KEY_B = 0x05
KEY_C = 0x06
KEY_D = 0x07
KEY_E = 0x08
KEY_F = 0x09
KEY_G = 0x0A
KEY_H = 0x0B
KEY_I = 0x0C
KEY_J = 0x0D
KEY_K = 0x0E
KEY_L = 0x0F
KEY_M = 0x10
KEY_N = 0x11
KEY_O = 0x12
KEY_P = 0x13
KEY_Q = 0x14
KEY_R = 0x15
KEY_S = 0x16
KEY_T = 0x17
KEY_U = 0x18
KEY_V = 0x19
KEY_W = 0x1A
KEY_X = 0x1B
KEY_Y = 0x1C
KEY_Z = 0x1D

KEY_1 = 0x1E
KEY_2 = 0x1F
KEY_3 = 0x20
KEY_4 = 0x21
KEY_5 = 0x22
KEY_6 = 0x23
KEY_7 = 0x24
KEY_8 = 0x25
KEY_9 = 0x26
KEY_0 = 0x27

KEY_ENTER = 0x28
KEY_ESCAPE = 0x29
KEY_BACKSPACE = 0x2A
KEY_TAB = 0x2B
KEY_SPACE = 0x2C
KEY_MINUS = 0x2D
KEY_EQUAL = 0x2E
KEY_LEFT_BRACE = 0x2F
KEY_RIGHT_BRACE = 0x30
KEY_BACKSLASH = 0x31
KEY_SEMICOLON = 0x33
KEY_APOSTROPHE = 0x34
KEY_GRAVE = 0x35
KEY_COMMA = 0x36
KEY_PERIOD = 0x37
KEY_SLASH = 0x38
KEY_CAPS_LOCK = 0x39

KEY_F1 = 0x3A
KEY_F2 = 0x3B
KEY_F3 = 0x3C
KEY_F4 = 0x3D
KEY_F5 = 0x3E
KEY_F6 = 0x3F
KEY_F7 = 0x40
KEY_F8 = 0x41
KEY_F9 = 0x42
KEY_F10 = 0x43
KEY_F11 = 0x44
KEY_F12 = 0x45

KEY_PRINT_SCREEN = 0x46
KEY_SCROLL_LOCK = 0x47
KEY_PAUSE = 0x48
KEY_INSERT = 0x49
KEY_HOME = 0x4A
KEY_PAGE_UP = 0x4B
KEY_DELETE = 0x4C
KEY_END = 0x4D
KEY_PAGE_DOWN = 0x4E

KEY_RIGHT = 0x4F
KEY_LEFT = 0x50
KEY_DOWN = 0x51
KEY_UP = 0x52

KEY_NUM_LOCK = 0x53
KEY_KP_SLASH = 0x54
KEY_KP_ASTERISK = 0x55
KEY_KP_MINUS = 0x56
KEY_KP_PLUS = 0x57
KEY_KP_ENTER = 0x58
KEY_KP1 = 0x59
KEY_KP2 = 0x5A
KEY_KP3 = 0x5B
KEY_KP4 = 0x5C
KEY_KP5 = 0x5D
KEY_KP6 = 0x5E
KEY_KP7 = 0x5F
KEY_KP8 = 0x60
KEY_KP9 = 0x61
KEY_KP0 = 0x62
KEY_KP_DOT = 0x63

KEY_APPLICATION = 0x65


class KeyboardState:
    """Represents the state of a HID keyboard.

    Uses a 256-bit bitmap for N-key rollover support.
    """

    __slots__ = ("modifiers", "key_bitmap")

    def __init__(self, modifiers: int = 0) -> None:
        self.modifiers: int = modifiers  # u8 bitmask
        self.key_bitmap: bytearray = bytearray(32)  # 256 bits

    def press(self, *keys: int) -> None:
        """Set bits for one or more HID key codes."""
        for key in keys:
            byte_idx = key // 8
            bit_idx = key % 8
            self.key_bitmap[byte_idx] |= 1 << bit_idx

    def release_all(self) -> None:
        """Clear all pressed keys (keeps modifiers)."""
        for i in range(32):
            self.key_bitmap[i] = 0

    def marshal(self) -> bytes:
        """Encode to VIIPER wire format: modifiers:u8 count:u8 keys:u8*count."""
        # Collect pressed key codes
        keys: list[int] = []
        for i in range(256):
            byte_idx = i // 8
            bit_idx = i % 8
            if self.key_bitmap[byte_idx] & (1 << bit_idx):
                keys.append(i)
        buf = bytearray(2 + len(keys))
        buf[0] = self.modifiers & 0xFF
        buf[1] = len(keys) & 0xFF
        for j, k in enumerate(keys):
            buf[2 + j] = k & 0xFF
        return bytes(buf)

    @classmethod
    def press_key_with_mod(cls, modifiers: int, *keys: int) -> KeyboardState:
        """Create a state with modifiers and keys pressed."""
        state = cls(modifiers)
        state.press(*keys)
        return state

    @classmethod
    def release(cls) -> KeyboardState:
        """Create an empty resting state."""
        return cls(0)


# ── VK (Virtual Key) to HID usage code mapping ────────────────────────
# Ported from input-bridge/keys.go
VK_TO_HID: dict[int, int] = {
    # Letters
    0x41: KEY_A, 0x42: KEY_B, 0x43: KEY_C, 0x44: KEY_D,
    0x45: KEY_E, 0x46: KEY_F, 0x47: KEY_G, 0x48: KEY_H,
    0x49: KEY_I, 0x4A: KEY_J, 0x4B: KEY_K, 0x4C: KEY_L,
    0x4D: KEY_M, 0x4E: KEY_N, 0x4F: KEY_O, 0x50: KEY_P,
    0x51: KEY_Q, 0x52: KEY_R, 0x53: KEY_S, 0x54: KEY_T,
    0x55: KEY_U, 0x56: KEY_V, 0x57: KEY_W, 0x58: KEY_X,
    0x59: KEY_Y, 0x5A: KEY_Z,
    # Numbers (top row)
    0x30: KEY_0, 0x31: KEY_1, 0x32: KEY_2, 0x33: KEY_3,
    0x34: KEY_4, 0x35: KEY_5, 0x36: KEY_6, 0x37: KEY_7,
    0x38: KEY_8, 0x39: KEY_9,
    # Special
    0x20: KEY_SPACE,
    0x0D: KEY_ENTER,
    0x08: KEY_BACKSPACE,
    0x09: KEY_TAB,
    0x1B: KEY_ESCAPE,
    # Arrows
    0x25: KEY_LEFT, 0x26: KEY_UP, 0x27: KEY_RIGHT, 0x28: KEY_DOWN,
    # Editing
    0x2D: KEY_INSERT, 0x2E: KEY_DELETE,
    0x24: KEY_HOME, 0x23: KEY_END,
    0x21: KEY_PAGE_UP, 0x22: KEY_PAGE_DOWN,
    # Function keys
    0x70: KEY_F1, 0x71: KEY_F2, 0x72: KEY_F3, 0x73: KEY_F4,
    0x74: KEY_F5, 0x75: KEY_F6, 0x76: KEY_F7, 0x77: KEY_F8,
    0x78: KEY_F9, 0x79: KEY_F10, 0x7A: KEY_F11, 0x7B: KEY_F12,
    # Punctuation/symbols
    0xBA: KEY_SEMICOLON, 0xBB: KEY_EQUAL, 0xBC: KEY_COMMA,
    0xBD: KEY_MINUS, 0xBE: KEY_PERIOD, 0xBF: KEY_SLASH,
    0xC0: KEY_GRAVE, 0xDB: KEY_LEFT_BRACE, 0xDC: KEY_BACKSLASH,
    0xDD: KEY_RIGHT_BRACE, 0xDE: KEY_APOSTROPHE,
    # Numpad
    0x60: KEY_KP0, 0x61: KEY_KP1, 0x62: KEY_KP2, 0x63: KEY_KP3,
    0x64: KEY_KP4, 0x65: KEY_KP5, 0x66: KEY_KP6, 0x67: KEY_KP7,
    0x68: KEY_KP8, 0x69: KEY_KP9,
    0x6A: KEY_KP_ASTERISK, 0x6B: KEY_KP_PLUS, 0x6D: KEY_KP_MINUS,
    0x6E: KEY_KP_DOT, 0x6F: KEY_KP_SLASH,
}

# VK to modifier bitmask for modifier keys
VK_TO_MODIFIER: dict[int, int] = {
    0xA0: MOD_LEFT_SHIFT,   # VK_LSHIFT
    0xA1: MOD_RIGHT_SHIFT,  # VK_RSHIFT
    0xA2: MOD_LEFT_CTRL,    # VK_LCONTROL
    0xA3: MOD_RIGHT_CTRL,   # VK_RCONTROL
    0xA4: MOD_LEFT_ALT,     # VK_LMENU
    0xA5: MOD_RIGHT_ALT,    # VK_RMENU
    0x5B: MOD_LEFT_GUI,     # VK_LWIN
    0x5C: MOD_RIGHT_GUI,    # VK_RWIN
}


def vk_to_hid(vk: int) -> int:
    """Convert a Windows virtual key code to a HID usage code.

    Returns 0 if the VK is unmapped (e.g. mouse keys, volume keys).
    """
    return VK_TO_HID.get(vk, 0)


def vk_to_modifier(vk: int) -> int:
    """Return the modifier bitmask for a VK, or 0 if not a modifier."""
    return VK_TO_MODIFIER.get(vk, 0)
