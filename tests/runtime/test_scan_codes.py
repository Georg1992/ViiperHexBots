"""Tests for config key-name → Windows scan-code mapping."""

from __future__ import annotations

import pytest

from pybot.runtime.input.scan_codes import key_name_to_scan_code, keysym_to_key_name


@pytest.mark.parametrize(
    "name, expected",
    [
        ("f1", 0x3B),
        ("F2", 0x3C),
        ("f10", 0x44),
        ("f11", 0x57),
        ("f12", 0x58),
        ("e", 0x12),
        ("insert", 0x52),
        ("", 0),
        ("unknown", 0),
    ],
)
def test_key_name_to_scan_code(name: str, expected: int) -> None:
    assert key_name_to_scan_code(name) == expected


@pytest.mark.parametrize(
    "keysym, expected",
    [
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
    ],
)
def test_keysym_to_key_name(keysym: str, expected: str) -> None:
    assert keysym_to_key_name(keysym) == expected
