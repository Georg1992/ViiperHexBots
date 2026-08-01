"""Surround detection and danger-state reset regression tests."""

from __future__ import annotations

import unittest

from pybot.runtime.character_state import (
    CharacterState,
    is_surrounded_by_tracks,
)


class SurroundTests(unittest.TestCase):
    def test_one_nearby_tracked_mob_is_not_surrounded(self) -> None:
        surrounded, reason = is_surrounded_by_tracks(
            char_x=500,
            char_y=500,
            all_mobs=[(520, 500)],
        )
        self.assertFalse(surrounded)
        self.assertEqual(reason, "")

    def test_two_nearby_on_opposite_sides_is_surrounded(self) -> None:
        surrounded, reason = is_surrounded_by_tracks(
            char_x=500,
            char_y=500,
            all_mobs=[(400, 500), (600, 500)],
        )
        self.assertTrue(surrounded)
        self.assertEqual(reason, "left+right")

    def test_clear_area_threat_resets_stale_surround(self) -> None:
        state = CharacterState()
        state.publish(
            char_x=1,
            char_y=2,
            is_surrounded=True,
            surrounded_reason="above+below",
            nearby_mob_count=3,
            nearby_any_mobs_count=2,
            tick_ms=10,
        )
        state.clear_area_threat()
        self.assertFalse(state.is_surrounded)
        self.assertEqual(state.surrounded_reason, "")
        self.assertEqual(state.nearby_mob_count, 0)
        self.assertEqual(state.nearby_any_mobs_count, 0)
        self.assertEqual(state.char_pos, (1, 2))


if __name__ == "__main__":
    unittest.main()
