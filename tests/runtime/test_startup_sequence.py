from __future__ import annotations

import unittest

from pybot.runtime.gate_controller import GateController
from pybot.runtime.startup_sequence import HuntStartupSequence


class HuntStartupSequenceTests(unittest.TestCase):
    def test_unmanaged_sequence_is_ready_for_lightweight_fixtures(self) -> None:
        sequence = HuntStartupSequence()

        self.assertTrue(sequence.buffs_done.is_set())
        self.assertTrue(sequence.timers_done.is_set())
        self.assertTrue(sequence.is_combat_ready())
        self.assertEqual(sequence.generation, 0)

    def test_begin_requires_area_buffs_and_timers_for_production_startup(self) -> None:
        sequence = HuntStartupSequence()
        sequence.begin()

        self.assertFalse(sequence.area_clear.is_set())
        self.assertFalse(sequence.buffs_done.is_set())
        self.assertFalse(sequence.timers_done.is_set())
        self.assertFalse(sequence.is_combat_ready())

        sequence.mark_area_clear()
        sequence.mark_buffs_done()
        sequence.mark_timers_done()

        self.assertTrue(sequence.area_clear.is_set())
        self.assertTrue(sequence.buffs_done.is_set())
        self.assertTrue(sequence.is_combat_ready())

    def test_new_hunt_advances_generation_and_clears_milestones_together(self) -> None:
        sequence = HuntStartupSequence()
        sequence.begin()
        sequence.mark_area_clear()
        sequence.mark_buffs_done()
        sequence.mark_timers_done()
        old_generation = sequence.generation

        sequence.begin_new_hunt()

        self.assertEqual(sequence.generation, old_generation + 1)
        self.assertFalse(sequence.area_clear.is_set())
        self.assertFalse(sequence.buffs_done.is_set())
        self.assertFalse(sequence.timers_done.is_set())
        self.assertFalse(sequence.is_combat_ready())

    def test_unmanaged_new_hunt_keeps_fixture_ready_state(self) -> None:
        sequence = HuntStartupSequence()
        sequence.begin_new_hunt()

        self.assertEqual(sequence.generation, 1)
        self.assertTrue(sequence.buffs_done.is_set())
        self.assertTrue(sequence.timers_done.is_set())
        self.assertTrue(sequence.is_combat_ready())

    def test_gate_resets_startup_before_releasing_sit_gate(self) -> None:
        sequence = HuntStartupSequence()
        sequence.begin()
        gates = GateController(startup=sequence)
        self.assertTrue(gates.try_begin_sit_ops())
        sequence.mark_area_clear()
        sequence.mark_buffs_done()
        sequence.mark_timers_done()
        old_generation = sequence.generation

        gates.end_sit_ops()

        self.assertEqual(sequence.generation, old_generation + 1)
        self.assertFalse(gates.sitting_event.is_set())
        self.assertFalse(sequence.area_clear.is_set())
        self.assertFalse(sequence.buffs_done.is_set())
        self.assertFalse(sequence.timers_done.is_set())
        self.assertFalse(gates.should_run_combat())

        sequence.mark_area_clear()
        sequence.mark_buffs_done()
        sequence.mark_timers_done()
        self.assertTrue(gates.should_run_combat())


if __name__ == "__main__":
    unittest.main()
