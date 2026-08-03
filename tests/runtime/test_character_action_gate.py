"""Shared stagger + buff-priority gate unit tests."""

from __future__ import annotations

import unittest

from pybot.runtime.constants import SKILL_TIMER_STAGGER_MS
from pybot.runtime.gate_controller import CharacterActionGate


class CharacterActionGateTests(unittest.TestCase):
    def test_first_claim_is_always_open(self) -> None:
        gate = CharacterActionGate()
        self.assertTrue(gate.try_claim(is_buff=False, now_ms=1_000))
        self.assertTrue(gate.try_claim(is_buff=True, now_ms=2_000))

    def test_stagger_window_blocks_every_keypress(self) -> None:
        gate = CharacterActionGate()
        self.assertTrue(gate.try_claim(is_buff=True, now_ms=10_000))
        # A timer press within the window is refused.
        self.assertFalse(gate.try_claim(is_buff=False, now_ms=10_200))
        # A buff cast within the window is refused too (shared window).
        self.assertFalse(gate.try_claim(is_buff=True, now_ms=10_200))
        # Once the window elapses, either worker may claim.
        self.assertTrue(
            gate.try_claim(is_buff=True, now_ms=10_000 + SKILL_TIMER_STAGGER_MS)
        )

    def test_buff_burst_makes_timers_yield_but_not_buffs(self) -> None:
        gate = CharacterActionGate()
        gate.begin_buff_burst()
        # Timers cannot claim while a buff burst is pending.
        self.assertFalse(gate.try_claim(is_buff=False, now_ms=20_000))
        # Buffs may still claim their own slot.
        self.assertTrue(gate.try_claim(is_buff=True, now_ms=20_000))
        gate.end_buff_burst()
        # After the burst, timers claim again once the stagger window reopens.
        self.assertFalse(gate.try_claim(is_buff=False, now_ms=20_100))
        self.assertTrue(
            gate.try_claim(
                is_buff=False,
                now_ms=20_000 + SKILL_TIMER_STAGGER_MS,
            )
        )

    def test_buff_cast_records_time_for_timers(self) -> None:
        gate = CharacterActionGate()
        gate.note_action(30_000)
        # A timer due 100ms after the buff cast must wait out the window.
        self.assertFalse(gate.try_claim(is_buff=False, now_ms=30_100))
        self.assertEqual(gate.stagger_remaining_ms(30_100), SKILL_TIMER_STAGGER_MS - 100)
        self.assertTrue(
            gate.try_claim(is_buff=False, now_ms=30_000 + SKILL_TIMER_STAGGER_MS)
        )

    def test_stagger_remaining_is_zero_without_prior_action(self) -> None:
        gate = CharacterActionGate()
        self.assertEqual(gate.stagger_remaining_ms(5_000), 0)


if __name__ == "__main__":
    unittest.main()
