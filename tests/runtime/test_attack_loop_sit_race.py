"""Attack loop must not continue after sit claims the combat gate mid-delay."""

from __future__ import annotations

import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from pybot.game_state import PlayerVitals
from pybot.runtime.workers.attack_loop import AttackLoop


class AttackLoopSitRaceTests(unittest.TestCase):
    def test_debuff_prepares_target_before_attack_input(self) -> None:
        ctx = MagicMock()
        ctx.config = SimpleNamespace(
            skill_scan_code=16,
            skill_delay_ms=1,
            use_sprite_grf=False,
        )
        events: list[str] = []
        ctx.stop_event = MagicMock()
        ctx.stop_event.wait.side_effect = lambda _timeout: events.append("delay")
        ctx.should_run_combat.return_value = True
        ctx.tracks.snapshot_for_track.return_value = SimpleNamespace(
            x=10,
            y=20,
            debuff_applied=False,
            was_accessible=False,
            discovery_stationary=False,
            moving=False,
            idle_attack_count=0,
            attack_count=0,
        )
        ctx.tracks.positions_snapshot.return_value = [(10, 20)]
        ctx.tracks.mark_debuff_applied.side_effect = lambda _target_id: events.append("marked") or True
        ctx.tracks.evaluate_idle_attack.return_value = ("none", 0)
        ctx.logger = MagicMock()
        ctx.overlay = MagicMock()
        ctx.policy = MagicMock()

        def prepare_target(*_args, **kwargs):
            events.append("debuff")
            self.assertFalse(kwargs["target_debuffed"])
            self.assertTrue(kwargs["mark_debuffed"]())
            return True

        mob_behavior = MagicMock()
        mob_behavior.prepare_target.side_effect = prepare_target
        mob_behavior.before_attack.side_effect = lambda *_args, **_kwargs: events.append("before")
        mob_behavior.kite_after_attack.side_effect = lambda *_args, **_kwargs: events.append("kite")
        input_backend = MagicMock()
        input_backend.skill_click_at.side_effect = lambda *_args: events.append("attack") or True
        vitals = PlayerVitals()
        loop = AttackLoop(
            ctx,
            MagicMock(),
            input_backend,
            mob_behavior=mob_behavior,
            vitals=vitals,
        )

        loop._attack_one(1, 1)

        self.assertEqual(events, ["debuff", "marked", "before", "attack", "kite", "delay"])
        ctx.tracks.mark_debuff_applied.assert_called_once_with(1)

    def test_run_keeps_healing_in_normal_target_and_idle_paths(self) -> None:
        ctx = MagicMock()
        ctx.config = SimpleNamespace(skill_scan_code=16, skill_delay_ms=1)
        ctx.logger = MagicMock()
        ctx.should_run_combat.return_value = True
        ctx.is_stopped.side_effect = [False, False, True]
        ctx.tracks.tracks_for_policy.return_value = []
        ctx.policy.select_target.side_effect = [1, None]
        ctx.stop_event = MagicMock()
        ctx.character_screen_pos.return_value = (100, 100)

        mob_behavior = MagicMock()
        loop = AttackLoop(
            ctx,
            MagicMock(),
            MagicMock(),
            mob_behavior=mob_behavior,
            vitals=PlayerVitals(),
        )
        loop._attack_one = MagicMock()

        loop.run()

        self.assertEqual(mob_behavior.heal_if_needed.call_count, 2)
        mob_behavior.heal_if_needed.assert_any_call(100, 100, loop._input)
        loop._attack_one.assert_called_once_with(1, unittest.mock.ANY)

    def test_attack_kites_before_sit_blocks_combat_during_skill_delay(self) -> None:
        ctx = MagicMock()
        ctx.config = SimpleNamespace(skill_scan_code=16, skill_delay_ms=50)
        events: list[str] = []
        ctx.stop_event = MagicMock()
        ctx.stop_event.wait.side_effect = lambda _timeout: events.append("delay")
        ctx.should_run_combat.side_effect = [True, False]
        snap = SimpleNamespace(x=10, y=20)
        ctx.tracks.snapshot_for_track.return_value = snap
        ctx.character_screen_pos.return_value = (0, 0)
        ctx.logger = MagicMock()

        input_backend = MagicMock()
        input_backend.skill_click_at.side_effect = lambda *_args: events.append("attack") or True
        mob_behavior = MagicMock()
        mob_behavior.heal_if_needed = MagicMock()
        mob_behavior.kite_after_attack.side_effect = lambda *_args, **_kwargs: events.append("kite")
        vitals = PlayerVitals()
        vitals.publish_sp(100, 200)
        loop = AttackLoop(
            ctx,
            MagicMock(),
            input_backend,
            mob_behavior=mob_behavior,
            vitals=vitals,
        )

        # A sit claim that is already visible at the final admission boundary
        # must reject the stale attack and its follow-up kite entirely.
        ctx.should_run_combat.side_effect = None
        ctx.should_run_combat.return_value = False
        loop._attack_one(1, 1)

        input_backend.skill_click_at.assert_not_called()
        mob_behavior.kite_after_attack.assert_not_called()
        self.assertEqual(events, [])
        mob_behavior.heal_if_needed.assert_not_called()


if __name__ == "__main__":
    unittest.main()
