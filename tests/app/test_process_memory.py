"""Unit tests for memory address parsing (no live process required)."""

from __future__ import annotations

import unittest

from pybot.config.clients import load_client_profile, memory_addresses_from_dict
from pybot.paths import PROJECT_ROOT


class MemoryAddressParseTests(unittest.TestCase):
    def test_parse_hex_and_missing(self) -> None:
        addrs = memory_addresses_from_dict(
            {
                "characterNameAddress": "0x01202568",
                "currentSpAddress": "0x011FF910",
                "maxSpAddress": "0x011FF914",
                "currentWeightAddress": "0x011FBAA0",
                "totalWeightAddress": "0x011FBA9C",
            }
        )
        self.assertEqual(addrs.char_name, 0x01202568)
        self.assertEqual(addrs.current_sp, 0x011FF910)
        self.assertEqual(addrs.max_sp, 0x011FF914)
        self.assertEqual(addrs.current_weight, 0x011FBAA0)
        self.assertEqual(addrs.max_weight, 0x011FBA9C)
        self.assertTrue(addrs.has_any)
        self.assertFalse(memory_addresses_from_dict(None).has_any)

    def test_revenant_profile_includes_name_sp_weight(self) -> None:
        profile = load_client_profile("Revenant", PROJECT_ROOT)
        assert profile is not None
        self.assertTrue(profile.memory.has_any)
        self.assertEqual(profile.memory.char_name, 0x01202568)
        self.assertEqual(profile.memory.current_sp, 0x011FF910)
        self.assertEqual(profile.memory.current_weight, 0x011FBAA0)
        self.assertFalse(hasattr(profile.memory, "current_hp"))


if __name__ == "__main__":
    unittest.main()
