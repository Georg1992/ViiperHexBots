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

    def test_begin_trusts_safe_start_and_gates_combat_on_startup_actions(self) -> None:
        sequence = HuntStartupSequence()
        sequence.begin()

        # A fresh hunt is trusted to start at a safe location: the area is
        # considered clear immediately, so startup buffs/timers are not held
        # back by the first discovery scan.
        self.assertTrue(sequence.area_clear.is_set())
        self.assertFalse(sequence.buffs_done.is_set())
        self.assertFalse(sequence.timers_done.is_set())
        # With the area trusted clear, combat waits for buffs then timers.
        self.assertFalse(sequence.is_combat_ready())

        sequence.mark_buffs_done()
        self.assertFalse(sequence.is_combat_ready())
        self.assertFalse(sequence.timers_done.is_set())
        sequence.mark_timers_done()

        self.assertTrue(sequence.area_clear.is_set())
        self.assertTrue(sequence.buffs_done.is_set())
        self.assertTrue(sequence.is_combat_ready())

    def test_begin_area_downgrade_reopens_combat_for_populated_start(self) -> None:
        sequence = HuntStartupSequence()
        sequence.begin()
        self.assertTrue(sequence.area_clear.is_set())

        # The first scan finds mobs: the trusted-clear milestone downgrades
        # and combat reopens so the hunt can clear the populated area.
        self.assertTrue(sequence.mark_area_clear(False))
        self.assertFalse(sequence.area_clear.is_set())
        self.assertTrue(sequence.is_combat_ready())

        sequence.mark_area_clear()
        self.assertFalse(sequence.is_combat_ready())
        sequence.mark_buffs_done()
        sequence.mark_timers_done()
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


    def test_gate_trusts_sit_recovery_start_and_releases_sit_gate(self) -> None:
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
        # A recovery session only completes at a spot the character sat
        # through without damage, so the landing area is trusted clear and
        # startup buffs/timers run immediately — no scan-gated dead window.
        self.assertTrue(sequence.area_clear.is_set())
        self.assertFalse(sequence.buffs_done.is_set())
        self.assertFalse(sequence.timers_done.is_set())
        # With the area trusted clear, combat waits for buffs then timers.
        self.assertFalse(gates.should_run_combat())

        sequence.mark_buffs_done()
        self.assertFalse(gates.should_run_combat())
        self.assertFalse(sequence.timers_done.is_set())
        sequence.mark_timers_done()
        self.assertTrue(gates.should_run_combat())

    def test_danger_escape_keeps_hunt_intact_like_normal_teleport(self) -> None:
        """A hunting danger escape is a normal teleport: it must not break the hunt.

        Only sit/stand recovery and kafra (storage) sessions start a new hunt
        cycle. After an escape the active buffs/timers stay done and the
        generation is unchanged, so combat resumes immediately in the landing
        area instead of re-running the startup sequence.
        """
        sequence = HuntStartupSequence()
        sequence.begin()
        sequence.mark_area_clear()
        sequence.mark_buffs_done()
        sequence.mark_timers_done()
        gates = GateController(startup=sequence)
        old_generation = sequence.generation

        gates.finish_danger_transition(seated=False)

        self.assertEqual(sequence.generation, old_generation)
        self.assertTrue(sequence.area_clear.is_set())
        self.assertTrue(sequence.buffs_done.is_set())
        self.assertTrue(sequence.timers_done.is_set())
        self.assertTrue(gates.should_run_combat())
        # The landing area is scanned promptly so tracks can be re-created.
        self.assertTrue(gates.discovery_wake.is_set())

    def test_seated_danger_escape_never_touches_hunt_state(self) -> None:
        """Seated escapes stay inside the SP session and touch nothing."""
        sequence = HuntStartupSequence()
        sequence.begin()
        sequence.mark_area_clear()
        sequence.mark_buffs_done()
        sequence.mark_timers_done()
        gates = GateController(startup=sequence)
        old_generation = sequence.generation

        gates.finish_danger_transition(seated=True)

        self.assertEqual(sequence.generation, old_generation)
        self.assertTrue(sequence.buffs_done.is_set())
        self.assertTrue(sequence.timers_done.is_set())
        self.assertFalse(gates.discovery_wake.is_set())

    def test_new_hunt_with_trusted_clear_runs_startup_immediately(self) -> None:
        sequence = HuntStartupSequence()
        sequence.begin()
        sequence.mark_buffs_done()
        sequence.mark_timers_done()

        sequence.begin_new_hunt(trusted_clear=True)

        self.assertTrue(sequence.area_clear.is_set())
        self.assertFalse(sequence.buffs_done.is_set())
        self.assertFalse(sequence.timers_done.is_set())
        # Trusted clear releases startup actions without waiting for a scan;
        # combat gates on buffs/timers exactly like a fresh begin().
        self.assertFalse(sequence.is_combat_ready())

        # A populated landing can still downgrade the trusted milestone and
        # reopen combat for the clear pass.
        sequence.mark_area_clear(False)
        self.assertTrue(sequence.is_combat_ready())
        sequence.mark_area_clear()
        self.assertFalse(sequence.is_combat_ready())
        sequence.mark_buffs_done()
        sequence.mark_timers_done()
        self.assertTrue(sequence.is_combat_ready())


if __name__ == "__main__":
    unittest.main()
