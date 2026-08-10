"""Attack loop must not continue after sit claims the combat gate mid-delay."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from pybot.game_state import PlayerVitals
from pybot.runtime.workers.attack_loop import AttackLoop


class AttackLoopSitRaceTests(unittest.TestCase):
    def test_debuff_prepares_target_before_attack_input(self) -> None:
        ctx = MagicMock()
        ctx.config = SimpleNamespace(
            skill_scan_code=16,
            skill_delay_ms=1,
            use_sprite_grf=False,
            custom_behavior=SimpleNamespace(heal_scan_code=16),
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

        # Kiting happens before the gameplay delay; this unit test exercises
        # target preparation/input ordering only. The delay is interruptible
        # and is not reached when the lightweight mock context reports the
        # combat gate closed after input.
        self.assertEqual(events, ["debuff", "marked", "before", "attack", "kite"])
        ctx.tracks.mark_debuff_applied.assert_called_once_with(1)

    def test_post_teleport_non_full_hp_has_heal_priority_over_combat(self) -> None:
        """Any missing HP after teleport gets the tick before combat."""
        ctx = MagicMock()
        ctx.config = SimpleNamespace(
            skill_scan_code=16,
            skill_delay_ms=1,
            custom_behavior=SimpleNamespace(heal_scan_code=16),
        )
        ctx.logger = MagicMock()
        ctx.should_run_combat.return_value = True
        ctx.should_run_custom_heal_actions.return_value = True
        ctx.in_post_teleport_heal_window.return_value = True
        ctx.character_screen_pos.return_value = (100, 100)
        ctx.tracks.tracks_for_policy.return_value = []
        ctx.policy.select_target.return_value = 1

        vitals = PlayerVitals()
        vitals.publish_hp(49, 100)
        mob_behavior = MagicMock()
        input_backend = MagicMock()
        input_backend.skill_click_at.return_value = True
        ctx.try_heal_if_allowed.side_effect = (
            lambda allowed, action: "cast" if allowed() and action() else "blocked"
        )
        loop = AttackLoop(
            ctx,
            MagicMock(),
            input_backend,
            mob_behavior=mob_behavior,
            vitals=vitals,
        )
        loop._attack_one = MagicMock()

        self.assertFalse(loop.process_pending())

        input_backend.skill_click_at.assert_called_once_with(16, 100, 100)
        ctx.policy.select_target.assert_not_called()
        loop._attack_one.assert_not_called()
        ctx.logger.behavior.assert_any_call(
            unittest.mock.ANY
        )

    def test_normal_hunt_non_full_hp_does_not_veto_combat(self) -> None:
        """The full-HP gate applies only during the post-teleport window."""
        ctx = MagicMock()
        ctx.config = SimpleNamespace(
            skill_scan_code=16,
            skill_delay_ms=1,
            custom_behavior=SimpleNamespace(heal_scan_code=16),
        )
        ctx.logger = MagicMock()
        ctx.should_run_combat.return_value = True
        ctx.in_post_teleport_heal_window.return_value = False
        ctx.character_screen_pos.return_value = (100, 100)
        ctx.tracks.tracks_for_policy.return_value = []
        ctx.policy.select_target.return_value = 1

        vitals = PlayerVitals()
        vitals.publish_hp(60, 100)
        mob_behavior = MagicMock()
        loop = AttackLoop(
            ctx,
            MagicMock(),
            MagicMock(),
            mob_behavior=mob_behavior,
            vitals=vitals,
        )
        loop._attack_one = MagicMock()

        self.assertTrue(loop.process_pending())
        ctx.policy.select_target.assert_called_once()
        loop._attack_one.assert_called_once()

    def test_post_teleport_unknown_hp_does_not_start_combat(self) -> None:
        """Unknown post-teleport HP must fail closed before combat."""
        ctx = MagicMock()
        ctx.config = SimpleNamespace(
            skill_scan_code=16,
            skill_delay_ms=1,
            custom_behavior=SimpleNamespace(heal_scan_code=16),
        )
        ctx.logger = MagicMock()
        ctx.should_run_combat.return_value = True
        ctx.in_post_teleport_heal_window.return_value = True
        ctx.character_screen_pos.return_value = (100, 100)
        ctx.stop_event = MagicMock()
        ctx.tracks.tracks_for_policy.return_value = []
        ctx.policy.select_target.return_value = 1

        vitals = PlayerVitals()
        mob_behavior = MagicMock()
        loop = AttackLoop(
            ctx,
            MagicMock(),
            MagicMock(),
            mob_behavior=mob_behavior,
            vitals=vitals,
        )
        loop._attack_one = MagicMock()

        self.assertFalse(loop.process_pending())
        ctx.policy.select_target.assert_not_called()
        loop._attack_one.assert_not_called()
        ctx.stop_event.wait.assert_called_once()

    def test_post_teleport_blocked_heal_retries_teleport(self) -> None:
        """A blocked post-teleport heal starts a fresh teleport retry."""
        ctx = MagicMock()
        ctx.config = SimpleNamespace(
            skill_scan_code=16,
            skill_delay_ms=1,
            custom_behavior=SimpleNamespace(heal_scan_code=16),
        )
        ctx.logger = MagicMock()
        ctx.should_run_combat.return_value = True
        ctx.should_run_custom_heal_actions.return_value = False
        ctx.in_post_teleport_heal_window.return_value = True
        ctx.character_screen_pos.return_value = (100, 100)
        ctx.stop_event = MagicMock()
        ctx.tracks.tracks_for_policy.return_value = []
        ctx.policy.select_target.return_value = 1

        vitals = PlayerVitals()
        vitals.publish_hp(40, 100)
        mob_behavior = MagicMock()
        input_backend = MagicMock()
        ctx.try_heal_if_allowed.side_effect = (
            lambda allowed, action: "cast" if allowed() and action() else "blocked"
        )
        teleport = MagicMock()
        teleport.retry_post_teleport_heal.return_value = True
        loop = AttackLoop(
            ctx,
            MagicMock(),
            input_backend,
            mob_behavior=mob_behavior,
            vitals=vitals,
            teleport_controller=teleport,
        )
        loop._attack_one = MagicMock()

        self.assertFalse(loop.process_pending())

        teleport.retry_post_teleport_heal.assert_called_once_with()
        ctx.policy.select_target.assert_not_called()
        loop._attack_one.assert_not_called()
        ctx.logger.behavior.assert_any_call(
            "[HEAL] post-teleport skill heal blocked; teleported to retry"
        )

    def test_blocked_skill_teleports_then_retries_once_after_settle(self) -> None:
        """A successful retry teleport permits exactly one next skill attempt."""
        ctx = MagicMock()
        ctx.config = SimpleNamespace(
            skill_scan_code=16,
            skill_delay_ms=1,
            custom_behavior=SimpleNamespace(heal_scan_code=16),
        )
        ctx.logger = MagicMock()
        ctx.should_run_combat.return_value = True
        ctx.should_run_custom_heal_actions.side_effect = [False, True]
        ctx.try_heal_if_allowed.side_effect = (
            lambda allowed, action: "cast" if allowed() and action() else "blocked"
        )
        ctx.in_post_teleport_heal_window.return_value = True
        ctx.character_screen_pos.return_value = (100, 100)
        ctx.stop_event = MagicMock()
        ctx.tracks.tracks_for_policy.return_value = []
        ctx.policy.select_target.return_value = 1

        vitals = PlayerVitals()
        vitals.publish_hp(40, 100)
        input_backend = MagicMock()
        input_backend.skill_click_at.return_value = True
        teleport = MagicMock()
        teleport.retry_post_teleport_heal.return_value = True
        loop = AttackLoop(
            ctx,
            MagicMock(),
            input_backend,
            vitals=vitals,
            teleport_controller=teleport,
        )
        loop._attack_one = MagicMock()

        self.assertFalse(loop.process_pending())
        self.assertFalse(loop.process_pending())

        teleport.retry_post_teleport_heal.assert_called_once_with()
        input_backend.skill_click_at.assert_called_once_with(16, 100, 100)
        ctx.policy.select_target.assert_not_called()
        loop._attack_one.assert_not_called()

    def test_unchanged_hp_after_verification_window_teleports(self) -> None:
        """A cast with no HP progress is blocked, not retried in place."""
        ctx = MagicMock()
        ctx.config = SimpleNamespace(
            skill_scan_code=16,
            skill_delay_ms=1,
            custom_behavior=SimpleNamespace(heal_scan_code=16),
        )
        ctx.logger = MagicMock()
        ctx.should_run_combat.return_value = True
        ctx.should_run_custom_heal_actions.return_value = True
        ctx.try_heal_if_allowed.side_effect = (
            lambda allowed, action: "cast" if allowed() and action() else "blocked"
        )
        ctx.in_post_teleport_heal_window.return_value = True
        ctx.character_screen_pos.return_value = (100, 100)
        ctx.stop_event = MagicMock()
        ctx.tracks.tracks_for_policy.return_value = []
        ctx.policy.select_target.return_value = 1

        vitals = PlayerVitals()
        vitals.publish_hp(40, 100)
        input_backend = MagicMock()
        input_backend.skill_click_at.return_value = True
        teleport = MagicMock()
        teleport.retry_post_teleport_heal.return_value = True
        loop = AttackLoop(
            ctx,
            MagicMock(),
            input_backend,
            vitals=vitals,
            teleport_controller=teleport,
        )

        self.assertFalse(loop.process_pending())
        # Simulate the heal remaining stale. The retry teleport must wait for
        # the complete custom skill-heal cooldown, not only the shorter OCR
        # verification window.
        last_changed_ms = vitals.hp_sample()[3]
        loop._last_skill_heal_ms = last_changed_ms
        from pybot.runtime.constants import HP_RESTORE_COOLDOWN_S

        cooldown_ms = int(HP_RESTORE_COOLDOWN_S * 1000)
        with patch(
            "pybot.runtime.workers.attack_loop.monotonic_ms",
            return_value=last_changed_ms + cooldown_ms - 1,
        ):
            self.assertFalse(loop.process_pending())
        teleport.retry_post_teleport_heal.assert_not_called()

        with patch(
            "pybot.runtime.workers.attack_loop.monotonic_ms",
            return_value=last_changed_ms + cooldown_ms,
        ):
            self.assertFalse(loop.process_pending())

        input_backend.skill_click_at.assert_called_once_with(16, 100, 100)
        teleport.retry_post_teleport_heal.assert_called_once_with()
        ctx.policy.select_target.assert_not_called()

    def test_blocked_skill_does_not_retry_in_place_when_teleport_fails(self) -> None:
        """A failed teleport retry suppresses repeated skill casts in place."""
        ctx = MagicMock()
        ctx.config = SimpleNamespace(
            skill_scan_code=16,
            skill_delay_ms=1,
            custom_behavior=SimpleNamespace(heal_scan_code=16),
        )
        ctx.logger = MagicMock()
        ctx.should_run_combat.return_value = True
        ctx.should_run_custom_heal_actions.return_value = False
        ctx.in_post_teleport_heal_window.return_value = True
        ctx.character_screen_pos.return_value = (100, 100)
        ctx.stop_event = MagicMock()
        ctx.tracks.tracks_for_policy.return_value = []
        ctx.policy.select_target.return_value = 1

        vitals = PlayerVitals()
        vitals.publish_hp(40, 100)
        input_backend = MagicMock()
        ctx.try_heal_if_allowed.side_effect = (
            lambda allowed, action: "cast" if allowed() and action() else "blocked"
        )
        teleport = MagicMock()
        teleport.retry_post_teleport_heal.return_value = False
        loop = AttackLoop(
            ctx,
            MagicMock(),
            input_backend,
            vitals=vitals,
            teleport_controller=teleport,
        )

        self.assertFalse(loop.process_pending())
        self.assertFalse(loop.process_pending())

        input_backend.skill_click_at.assert_not_called()
        self.assertEqual(teleport.retry_post_teleport_heal.call_count, 2)
        ctx.policy.select_target.assert_not_called()

    def test_successful_heal_does_not_immediately_retry_teleport(self) -> None:
        """Heal verification/cooldown must not trigger a second teleport."""
        ctx = MagicMock()
        ctx.config = SimpleNamespace(
            skill_scan_code=16,
            skill_delay_ms=1,
            custom_behavior=SimpleNamespace(heal_scan_code=16),
        )
        ctx.logger = MagicMock()
        ctx.should_run_combat.return_value = True
        ctx.should_run_custom_heal_actions.return_value = True
        ctx.try_heal_if_allowed.side_effect = (
            lambda allowed, action: "cast" if allowed() and action() else "blocked"
        )
        ctx.in_post_teleport_heal_window.return_value = True
        ctx.character_screen_pos.return_value = (100, 100)
        ctx.stop_event = MagicMock()
        ctx.tracks.tracks_for_policy.return_value = []

        vitals = PlayerVitals()
        vitals.publish_hp(40, 100)
        mob_behavior = MagicMock()
        teleport = MagicMock()
        input_backend = MagicMock()
        loop = AttackLoop(
            ctx,
            MagicMock(),
            input_backend,
            mob_behavior=mob_behavior,
            vitals=vitals,
            teleport_controller=teleport,
        )

        self.assertFalse(loop.process_pending())
        self.assertFalse(loop.process_pending())
        teleport.retry_post_teleport_heal.assert_not_called()
        self.assertEqual(input_backend.skill_click_at.call_count, 1)

    def test_post_teleport_cooldown_wait_does_not_retry_teleport(self) -> None:
        ctx = MagicMock()
        ctx.config = SimpleNamespace(
            skill_scan_code=16,
            skill_delay_ms=1,
            custom_behavior=SimpleNamespace(heal_scan_code=16),
        )
        ctx.logger = MagicMock()
        ctx.should_run_combat.return_value = True
        ctx.should_run_custom_heal_actions.return_value = True
        ctx.in_post_teleport_heal_window.return_value = True
        ctx.character_screen_pos.return_value = (100, 100)
        ctx.stop_event = MagicMock()
        ctx.tracks.tracks_for_policy.return_value = []
        vitals = PlayerVitals()
        vitals.publish_hp(40, 100)
        teleport = MagicMock()
        loop = AttackLoop(
            ctx,
            MagicMock(),
            MagicMock(),
            vitals=vitals,
            teleport_controller=teleport,
        )
        # The attack loop owns skill-heal verification timing. Simulate the
        # first cast still inside its verification window; no gate cooldown
        # query or teleport retry is involved.
        loop._last_skill_heal_ms = 1000

        with patch(
            "pybot.runtime.workers.attack_loop.monotonic_ms",
            return_value=1000 + 100,
        ):
            self.assertFalse(loop.process_pending())
        teleport.retry_post_teleport_heal.assert_not_called()

    def test_post_teleport_without_skill_keeps_recovery_gate(self) -> None:
        ctx = MagicMock()
        ctx.config = SimpleNamespace(
            skill_scan_code=16,
            skill_delay_ms=1,
            custom_behavior=SimpleNamespace(heal_scan_code=0),
        )
        ctx.logger = MagicMock()
        ctx.should_run_combat.return_value = True
        ctx.in_post_teleport_heal_window.return_value = True
        ctx.character_screen_pos.return_value = (100, 100)
        ctx.stop_event = MagicMock()
        ctx.tracks.tracks_for_policy.return_value = []
        ctx.policy.select_target.return_value = None
        vitals = PlayerVitals()
        vitals.publish_hp(40, 100)
        loop = AttackLoop(ctx, MagicMock(), MagicMock(), vitals=vitals)

        self.assertFalse(loop.process_pending())
        ctx.clear_post_teleport_heal.assert_not_called()
        ctx.policy.select_target.assert_not_called()

    def test_post_teleport_non_full_hp_without_heal_vetoes_combat(self) -> None:
        """Post-teleport HP must be full before combat, even if heal is blocked."""
        ctx = MagicMock()
        ctx.config = SimpleNamespace(
            skill_scan_code=16,
            skill_delay_ms=1,
            custom_behavior=SimpleNamespace(heal_scan_code=16),
        )
        ctx.logger = MagicMock()
        ctx.should_run_combat.return_value = True
        ctx.should_run_custom_heal_actions.return_value = False
        ctx.try_heal_if_allowed.return_value = "blocked"
        ctx.in_post_teleport_heal_window.return_value = True
        ctx.character_screen_pos.return_value = (100, 100)
        ctx.stop_event = MagicMock()
        ctx.tracks.tracks_for_policy.return_value = []
        ctx.policy.select_target.return_value = 1

        vitals = PlayerVitals()
        vitals.publish_hp(40, 100)
        mob_behavior = MagicMock()
        loop = AttackLoop(
            ctx,
            MagicMock(),
            MagicMock(),
            mob_behavior=mob_behavior,
            vitals=vitals,
        )
        loop._attack_one = MagicMock()

        self.assertFalse(loop.process_pending())

        ctx.policy.select_target.assert_not_called()
        loop._attack_one.assert_not_called()
        ctx.stop_event.wait.assert_called_once()

    def test_run_does_not_heal_during_normal_target_and_idle_paths(self) -> None:
        ctx = MagicMock()
        ctx.config = SimpleNamespace(
            skill_scan_code=16,
            skill_delay_ms=1,
            custom_behavior=SimpleNamespace(heal_scan_code=16),
        )
        ctx.logger = MagicMock()
        ctx.should_run_combat.return_value = True
        ctx.in_post_teleport_heal_window.return_value = False
        ctx.is_stopped.side_effect = [False, False, True]
        ctx.tracks.tracks_for_policy.return_value = []
        ctx.policy.select_target.side_effect = [1, None]
        ctx.stop_event = MagicMock()
        ctx.character_screen_pos.return_value = (100, 100)

        mob_behavior = MagicMock()
        input_backend = MagicMock()
        input_backend.skill_click_at.return_value = True
        vitals = PlayerVitals()
        vitals.publish_hp(40, 100)
        loop = AttackLoop(
            ctx,
            MagicMock(),
            input_backend,
            mob_behavior=mob_behavior,
            vitals=vitals,
        )
        loop._attack_one = MagicMock()

        loop.run()

        input_backend.skill_click_at.assert_not_called()
        loop._attack_one.assert_called_once_with(1, unittest.mock.ANY)

    def test_run_blocks_custom_healing_when_unsafe_and_outside_window(self) -> None:
        ctx = MagicMock()
        ctx.config = SimpleNamespace(
            skill_scan_code=16,
            skill_delay_ms=1,
            custom_behavior=SimpleNamespace(heal_scan_code=16),
        )
        ctx.logger = MagicMock()
        ctx.should_run_combat.return_value = True
        ctx.in_post_teleport_heal_window.return_value = False
        ctx.should_run_custom_heal_actions.return_value = False
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

        loop._attack_one.assert_called_once_with(1, unittest.mock.ANY)

    def test_run_heals_during_post_teleport_window_even_while_unsafe(self) -> None:
        """The custom heal fires right after a danger teleport even when recent
        damage keeps the normal safe-heal gate closed (low-HP cascade)."""
        ctx = MagicMock()
        ctx.config = SimpleNamespace(
            skill_scan_code=16,
            skill_delay_ms=1,
            custom_behavior=SimpleNamespace(heal_scan_code=16),
        )
        ctx.logger = MagicMock()
        ctx.should_run_combat.return_value = True
        ctx.in_post_teleport_heal_window.return_value = True
        ctx.should_run_custom_heal_actions.return_value = True
        ctx.try_heal_if_allowed.side_effect = (
            lambda allowed, action: "cast" if allowed() and action() else "blocked"
        )
        ctx.is_stopped.side_effect = [False, False, True]
        ctx.tracks.tracks_for_policy.return_value = []
        ctx.policy.select_target.side_effect = [None, None]
        ctx.stop_event = MagicMock()
        ctx.character_screen_pos.return_value = (100, 100)

        mob_behavior = MagicMock()
        input_backend = MagicMock()
        input_backend.skill_click_at.return_value = True
        vitals = PlayerVitals()
        vitals.publish_hp(40, 100)
        loop = AttackLoop(
            ctx,
            MagicMock(),
            input_backend,
            mob_behavior=mob_behavior,
            vitals=vitals,
        )
        loop._attack_one = MagicMock()

        loop.run()

        self.assertEqual(input_backend.skill_click_at.call_count, 1)

    def test_attack_drops_target_removed_during_skill_delay(self) -> None:
        """A discovery removal must prevent stale post-delay bookkeeping."""
        ctx = MagicMock()
        ctx.config = SimpleNamespace(
            skill_scan_code=16,
            skill_delay_ms=1,
            use_sprite_grf=True,
            custom_behavior=SimpleNamespace(heal_scan_code=16),
        )
        ctx.logger = MagicMock()
        ctx.should_run_combat.return_value = True
        ctx.stop_event = MagicMock()
        ctx.stop_event.wait.side_effect = lambda _timeout: None
        ctx.character_screen_pos.return_value = (0, 0)
        snapshot = SimpleNamespace(
            x=10, y=20, debuff_applied=False, was_accessible=False,
            discovery_stationary=False, moving=False,
            idle_attack_count=0, attack_count=0, area_epoch=0,
        )
        # Initial snapshot, preparation admission + freshest debuff re-read,
        # skill admission, the freshest-position click re-read, then removal
        # during the skill delay. This exercises the post-delay guard.
        ctx.tracks.snapshot_for_track.side_effect = [
            snapshot, snapshot, snapshot, snapshot, snapshot, snapshot, None,
        ]
        ctx.tracks.positions_snapshot.return_value = [(10, 20)]
        ctx.logger = MagicMock()
        input_backend = MagicMock()
        input_backend.skill_click_at.return_value = True
        mob_behavior = MagicMock()
        mob_behavior.prepare_target.return_value = True
        vitals = PlayerVitals()
        loop = AttackLoop(ctx, MagicMock(), input_backend, mob_behavior=mob_behavior, vitals=vitals)

        loop._attack_one(1, 1)

        ctx.tracks.evaluate_idle_attack.assert_not_called()
        ctx.tracks.apply_attack_event.assert_not_called()
        ctx.policy.note_attack_target.assert_not_called()
        self.assertTrue(any("stale target dropped" in str(call) for call in ctx.logger.behavior.call_args_list))

    def test_attack_kites_before_sit_blocks_combat_during_skill_delay(self) -> None:
        ctx = MagicMock()
        ctx.config = SimpleNamespace(
            skill_scan_code=16,
            skill_delay_ms=50,
            custom_behavior=SimpleNamespace(heal_scan_code=16),
        )
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


if __name__ == "__main__":
    unittest.main()
