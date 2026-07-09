"""Tests for per-mob attack accounting and unreachable rules."""

from __future__ import annotations

import unittest

from pybot.recognition.rules import (
    MobTrack,
    is_track_unreachable_by_attacks,
    max_attacks_per_mob_before_unreachable,
    mob_attack_anchor_key,
)


class UnreachableAttackRulesTests(unittest.TestCase):
    def test_not_unreachable_below_limit(self) -> None:
        track = MobTrack(id=1, x=100, y=100, attack_count=4)
        self.assertFalse(is_track_unreachable_by_attacks(track, 5))

    def test_unreachable_at_limit(self) -> None:
        track = MobTrack(id=1, x=100, y=100, attack_count=5)
        self.assertTrue(is_track_unreachable_by_attacks(track, 5))

    def test_anchor_key_snaps_to_dedup_cell(self) -> None:
        key = mob_attack_anchor_key(874, 578, cell_px=70)
        self.assertEqual(key, (840, 560))

    def test_formula_3s_over_delay_plus_average(self) -> None:
        # 3000ms / 1000ms + avg 1 = 4
        limit = max_attacks_per_mob_before_unreachable(
            average_attacks_till_death=1.0,
            skill_delay_ms=1000,
        )
        self.assertEqual(limit, 4)

    def test_faster_attack_delay_increases_budget(self) -> None:
        slow = max_attacks_per_mob_before_unreachable(
            average_attacks_till_death=1.0,
            skill_delay_ms=5000,
        )
        fast = max_attacks_per_mob_before_unreachable(
            average_attacks_till_death=1.0,
            skill_delay_ms=500,
        )
        self.assertLess(slow, fast)


if __name__ == "__main__":
    unittest.main()
