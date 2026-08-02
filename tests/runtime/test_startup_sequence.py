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
        # The initial area may be populated, so combat remains available for
        # the first clear pass. Once clear is confirmed, startup buffs and
        # timer casts gate combat before the normal hunt proceeds.
        self.assertTrue(sequence.is_combat_ready())

        sequence.mark_area_clear()
        self.assertFalse(sequence.is_combat_ready())
        sequence.mark_buffs_done()
        self.assertFalse(sequence.is_combat_ready())
        self.assertFalse(sequence.timers_done.is_set())
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
        # A recovered landing area may contain mobs. Combat must remain live
        # so discovery/tracking can create tracks and attack can clear it.
        self.assertTrue(sequence.is_combat_ready())
        sequence.mark_area_clear()
        self.assertFalse(sequence.is_combat_ready())
        sequence.mark_buffs_done()
        self.assertFalse(sequence.is_combat_ready())
        sequence.mark_timers_done()
        self.assertTrue(sequence.is_combat_ready())

    def test_recovered_populated_area_can_be_cleared_before_startup_gate(self) -> None:
        sequence = HuntStartupSequence()
        sequence.begin(require_buffs=True, require_timers=True)
        sequence.mark_area_clear()
        sequence.mark_buffs_done()
        sequence.mark_timers_done()
        sequence.begin_new_hunt()

        # This is the post-sit landing case: tracks must be discoverable and
        # attackable before an empty scan can transition to startup actions.
        self.assertTrue(sequence.is_combat_ready())
        sequence.mark_area_clear(False)
        self.assertTrue(sequence.is_combat_ready())
        sequence.mark_area_clear(True)
        self.assertFalse(sequence.is_combat_ready())
        sequence.mark_buffs_done()
        self.assertFalse(sequence.is_combat_ready())
        sequence.mark_timers_done()
        self.assertTrue(sequence.is_combat_ready())

    def test_stale_generation_completion_cannot_unlock_new_hunt(self) -> None:
        sequence = HuntStartupSequence()
        sequence.begin()
        old_generation = sequence.generation
        sequence.begin_new_hunt()
        new_generation = sequence.generation

        self.assertFalse(
            sequence.mark_area_clear(expected_generation=old_generation)
        )
        self.assertFalse(
            sequence.mark_buffs_done(expected_generation=old_generation)
        )
        self.assertFalse(
            sequence.mark_timers_done(expected_generation=old_generation)
        )
        self.assertFalse(sequence.area_clear.is_set())
        self.assertFalse(sequence.buffs_done.is_set())
        self.assertFalse(sequence.timers_done.is_set())
        # Stale completion does not unlock any milestone. Combat may still
        # clear a populated landing area before the first valid clear scan.
        self.assertTrue(sequence.is_combat_ready())

        self.assertTrue(
            sequence.mark_area_clear(expected_generation=new_generation)
        )
        self.assertTrue(
            sequence.mark_buffs_done(expected_generation=new_generation)
        )
        self.assertTrue(
            sequence.mark_timers_done(expected_generation=new_generation)
        )
        self.assertTrue(sequence.is_combat_ready())

    def test_unmanaged_new_hunt_keeps_fixture_ready_state(self) -> None:
        sequence = HuntStartupSequence()
        sequence.begin_new_hunt()

        self.assertEqual(sequence.generation, 1)
        self.assertTrue(sequence.buffs_done.is_set())
        self.assertTrue(sequence.timers_done.is_set())
        self.assertTrue(sequence.is_combat_ready())

    def test_new_hunt_without_startup_workers_stays_combat_ready(self) -> None:
        sequence = HuntStartupSequence()
        sequence.begin(require_buffs=False, require_timers=False)
        sequence.mark_area_clear()
        self.assertTrue(sequence.is_combat_ready())

        sequence.begin_new_hunt()
        self.assertFalse(sequence.area_clear.is_set())
        self.assertTrue(sequence.buffs_done.is_set())
        self.assertTrue(sequence.timers_done.is_set())
        # The next clear scan must not deadlock combat waiting for workers
        # that were never configured.
        sequence.mark_area_clear()
        self.assertTrue(sequence.is_combat_ready())

    def test_new_hunt_with_buffs_only_rearms_buff_gate(self) -> None:
        sequence = HuntStartupSequence()
        sequence.begin(require_buffs=True, require_timers=False)
        sequence.mark_area_clear()
        sequence.mark_buffs_done()

        sequence.begin_new_hunt()
        sequence.mark_area_clear()
        self.assertFalse(sequence.buffs_done.is_set())
        self.assertFalse(sequence.is_combat_ready())
        sequence.mark_buffs_done()
        self.assertTrue(sequence.is_combat_ready())
        self.assertTrue(sequence.timers_done.is_set())

    def test_startup_actions_follow_suspension_matrix(self) -> None:
        gates = GateController()
        gates.startup.begin(require_buffs=False, require_timers=False)
        gates.startup.mark_area_clear()

        # Timer schedules remain alive during heal/storage, but startup
        # character actions and combat are held by those session gates.
        self.assertTrue(gates.should_run_timers())
        self.assertTrue(gates.try_begin_heal_ops())
        self.assertTrue(gates.should_run_timers())
        self.assertFalse(gates.should_run_combat())
        gates.end_heal_ops()

        self.assertTrue(gates.try_begin_storage_ops())
        self.assertTrue(gates.should_run_timers())
        self.assertFalse(gates.should_run_combat())
        gates.end_storage_ops()

        # Safety transitions suspend timer input without changing the
        # healing/storage policy.
        gates.discovery_suspend.set()
        self.assertFalse(gates.should_run_timers())
        gates.discovery_suspend.clear()
        gates.request_danger_sit()
        self.assertFalse(gates.should_run_timers())
        gates.pop_danger_sit_request()

        self.assertTrue(gates.try_begin_sit_ops())
        self.assertFalse(gates.should_run_timers())
        self.assertFalse(gates.should_run_combat())
        gates.end_sit_ops()

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
        # Post-recovery combat remains available while discovery establishes
        # the new area; startup actions still remain gated until clear.
        self.assertTrue(gates.should_run_combat())

        sequence.mark_area_clear()
        self.assertFalse(gates.should_run_combat())
        sequence.mark_buffs_done()
        self.assertFalse(gates.should_run_combat())
        self.assertFalse(sequence.timers_done.is_set())
        sequence.mark_timers_done()
        self.assertTrue(gates.should_run_combat())


if __name__ == "__main__":
    unittest.main()
