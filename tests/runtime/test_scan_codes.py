"""Tests for config key-name → Windows scan-code mapping."""

from __future__ import annotations

import unittest

from pybot.runtime.input.scan_codes import key_name_to_scan_code, keysym_to_key_name


_KEY_NAME_CASES = [
    ("f1", 0x3B),
    ("F2", 0x3C),
    ("f10", 0x44),
    ("f11", 0x57),
    ("f12", 0x58),
    ("e", 0x12),
    ("insert", 0x52),
    ("", 0),
    ("unknown", 0),
]

_KEYSYM_CASES = [
    ("F1", "f1"),
    ("F12", "f12"),
    ("e", "e"),
    ("E", "e"),
    ("Insert", "insert"),
    ("Return", "enter"),
    ("space", "space"),
    ("Down", "down"),
    ("Up", "up"),
    ("Left", "left"),
    ("Right", "right"),
    ("Shift_L", ""),
    ("", ""),
]


class ScanCodeTests(unittest.TestCase):
    def test_key_name_to_scan_code(self) -> None:
        for name, expected in _KEY_NAME_CASES:
            with self.subTest(name=name):
                self.assertEqual(key_name_to_scan_code(name), expected)

    def test_keysym_to_key_name(self) -> None:
        for keysym, expected in _KEYSYM_CASES:
            with self.subTest(keysym=keysym):
                self.assertEqual(keysym_to_key_name(keysym), expected)


if __name__ == "__main__":
    unittest.main()
