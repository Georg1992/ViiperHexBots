"""HP item worker and danger detector regression tests."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from pybot.game_state import PlayerVitals
from pybot.runtime.constants import HP_RESTORE_RATIO
from pybot.runtime.danger_detector import (
    CRITICAL_DAMAGE_RATIO,
    CRITICAL_HP_RATIO,
    DangerDetector,
    DangerLevel,
)
from pybot.runtime.teleport import TeleportController
from pybot.runtime.runtime_context import HuntRuntimeContext
from pybot.runtime.workers.hp_restore_worker import HpRestoreWorker


class HpRestoreWorkerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = MagicMock()
        self.config.hp_button = "f1"
        self.config.hp_scan_code = 59
        self.ctx = HuntRuntimeContext(
            config=self.config,
            logger=MagicMock(),
            tracks=MagicMock(),
            policy=MagicMock(),
            capture=MagicMock(),
            detector=MagicMock(),
            tracker=MagicMock(),
            validation=MagicMock(),
            control=MagicMock(),
            overlay=MagicMock(),
        )
        self.input = MagicMock()
        self.vitals = PlayerVitals()

    def _worker(self) -> HpRestoreWorker:
        return HpRestoreWorker(self.ctx, self.input, self.vitals)

    def test_skips_when_no_hp_item_key(self) -> None:
        self.config.hp_scan_code = 0
        self._worker().run()
        self.input.key_tap.assert_not_called()

    def test_presses_hp_item_key_when_below_fifty_percent(self) -> None:
        self.vitals.publish_hp(49, 100)

        def stop_after_press(scan: int, *, after_s: float) -> bool:
            self.assertEqual(scan, 59)
            self.assertEqual(after_s, 0.0)
            self.ctx.stop_event.set()
            return True

        self.input.key_tap.side_effect = stop_after_press
        self._worker().run()

        self.input.key_tap.assert_called_once_with(59, after_s=0.0)
        self.assertLess(49 / 100, HP_RESTORE_RATIO)

    def test_does_not_press_hp_item_key_at_or_above_fifty_percent(self) -> None:
        self.vitals.publish_hp(50, 100)

        def stop_soon(*_args, **_kwargs) -> bool:
            self.ctx.stop_event.set()
            return False

        self.ctx.stop_event.wait = stop_soon  # type: ignore[method-assign]
        self._worker().run()
        self.input.key_tap.assert_not_called()

    def test_item_worker_does_not_use_skill_click(self) -> None:
        self.vitals.publish_hp(40, 100)

        def stop_after_press(*_args, **_kwargs) -> bool:
            self.ctx.stop_event.set()
            return True

        self.input.key_tap.side_effect = stop_after_press
        self._worker().run()
        self.input.skill_click_at.assert_not_called()


class DangerTeleportPriorityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = MagicMock()
        self.config.sit_on_low_sp_scan_code = 82
        self.ctx = HuntRuntimeContext(
            config=self.config,
            logger=MagicMock(),
            tracks=MagicMock(),
            policy=MagicMock(),
            capture=MagicMock(),
            detector=MagicMock(),
            tracker=MagicMock(),
            validation=MagicMock(),
            control=MagicMock(),
            overlay=MagicMock(),
        )
        self.ctx.mark_running()
        self.teleport = MagicMock()
        self.input = MagicMock()
        self.vitals = PlayerVitals()
        self.danger = DangerDetector(self.ctx, vitals=self.vitals)

    def test_critical_ratios_are_internal_thresholds(self) -> None:
        self.assertEqual(CRITICAL_HP_RATIO, 0.5)
        self.assertEqual(CRITICAL_DAMAGE_RATIO, 0.2)

    def test_danger_level_transitions_safe_danger_critical(self) -> None:
        self.assertEqual(self.danger.danger_level(), DangerLevel.SAFE)

        self.vitals.publish_hp(90, 100)
        self.danger._poll_hp()
        self.vitals.publish_hp(80, 100)
        self.danger._poll_hp()
        self.assertEqual(self.danger.danger_level(), DangerLevel.DANGER)


    def test_low_hp_damage_is_critical(self) -> None:
        self.vitals.publish_hp(80, 100)
        self.danger._poll_hp()
        self.vitals.publish_hp(40, 100)
        self.danger._poll_hp()
        self.assertEqual(self.danger.danger_level(), DangerLevel.CRITICAL)

    def test_drop_greater_than_twenty_percent_of_previous_tick_is_critical(self) -> None:
        self.vitals.publish_hp(150, 200)
        self.danger._poll_hp()
        # 22% of the previous tick's 150 HP; current/max is still 58.5%.
        self.vitals.publish_hp(117, 200)
        self.danger._poll_hp()
        self.assertEqual(self.danger.danger_level(), DangerLevel.CRITICAL)

    def test_drop_of_exactly_twenty_percent_is_not_critical(self) -> None:
        self.vitals.publish_hp(150, 200)
        self.danger._poll_hp()
        self.vitals.publish_hp(120, 200)
        self.danger._poll_hp()
        self.assertEqual(self.danger.danger_level(), DangerLevel.DANGER)

    def test_drop_uses_previous_hp_not_hp_max_for_ratio(self) -> None:
        self.vitals.publish_hp(300, 1000)
        self.danger._poll_hp()
        # 90/300 = 30% of previous HP, despite only 9% of max HP.
        self.vitals.publish_hp(210, 1000)
        self.danger._poll_hp()
        self.assertEqual(self.danger.danger_level(), DangerLevel.CRITICAL)

    def test_critical_drop_with_damage_requests_sit_before_heal(self) -> None:
        self.vitals.publish_hp(80, 100)
        self.danger._poll_hp()
        self.vitals.publish_hp(40, 100)
        self.danger._poll_hp()
        self.assertTrue(self.ctx.danger_sit_requested.is_set())
        self.teleport.danger_teleport.assert_not_called()
        self.assertFalse(hasattr(self.danger, "pop_heal_until_full_requested"))

    def test_critical_hp_without_recent_damage_does_not_teleport(self) -> None:
        # A low-HP snapshot establishes the baseline; it is not an attack.
        self.vitals.publish_hp(40, 100)
        self.danger._poll_hp()
        self.teleport.danger_teleport.assert_not_called()

    def test_repeated_low_hp_without_new_damage_never_requeues_danger(self) -> None:
        # Polling the same low HP repeatedly must not produce an infinite
        # sequence of danger teleports.
        self.vitals.publish_hp(40, 100)
        self.danger._poll_hp()
        self.danger._poll_hp()
        self.danger._poll_hp()

        self.assertFalse(self.ctx.danger_sit_requested.is_set())
        self.teleport.danger_teleport.assert_not_called()

        # One actual decrease is the only event that queues danger.
        self.vitals.publish_hp(39, 100)
        self.danger._poll_hp()
        self.assertTrue(self.ctx.danger_sit_requested.is_set())

    def test_exactly_fifty_percent_is_not_critical(self) -> None:
        self.vitals.publish_hp(80, 100)
        self.danger._poll_hp()
        self.vitals.publish_hp(50, 100)
        self.danger._poll_hp()
        self.teleport.danger_teleport.assert_not_called()

    def test_non_critical_hp_drop_does_not_urgent_teleport(self) -> None:
        self.vitals.publish_hp(90, 100)
        self.danger._poll_hp()
        self.vitals.publish_hp(80, 100)
        self.danger._poll_hp()
        self.teleport.danger_teleport.assert_not_called()
        self.assertTrue(self.danger.has_recent_damage(1.0))

    def test_recent_damage_is_time_based(self) -> None:
        self.ctx.mark_paused()
        self.vitals.publish_hp(80, 100)
        self.danger._poll_hp()
        self.vitals.publish_hp(70, 100)
        self.danger._poll_hp()
        self.assertTrue(self.danger.has_recent_damage(1.0))
        self.assertFalse(self.danger.has_recent_damage(0.0))
        self.teleport.danger_teleport.assert_not_called()

    def test_reset_after_teleport_returns_to_safe(self) -> None:
        self.vitals.publish_hp(90, 100)
        self.danger._poll_hp()
        self.vitals.publish_hp(80, 100)
        self.danger._poll_hp()
        self.assertEqual(self.danger.danger_level(), DangerLevel.DANGER)

        self.danger.reset_after_teleport()
        self.assertEqual(self.danger.danger_level(), DangerLevel.SAFE)
        # Resetting damage classification does not consume the sit request;
        # the sit worker owns that event.
        self.assertTrue(self.ctx.danger_sit_requested.is_set())
        self.assertTrue(self.ctx.pop_danger_sit_request())
        self.assertFalse(self.ctx.danger_sit_requested.is_set())

    def test_reset_after_teleport_preserves_damage_observed_during_settle(self) -> None:
        import time

        self.vitals.publish_hp(90, 100)
        self.danger._poll_hp()
        teleport_started = time.monotonic()
        self.vitals.publish_hp(80, 100)
        self.danger._poll_hp()

        self.danger.reset_after_teleport(teleport_started)

        self.assertTrue(self.danger.has_recent_damage(1.0))
        self.assertEqual(self.danger.danger_level(), DangerLevel.DANGER)

    def test_reset_after_teleport_rebaselines_first_post_teleport_sample(self) -> None:
        self.vitals.publish_hp(90, 100)
        self.danger._poll_hp()
        self.vitals.publish_hp(80, 100)
        self.danger._poll_hp()
        self.assertTrue(self.ctx.pop_danger_sit_request())

        self.danger.reset_after_teleport()
        self.danger._poll_hp()  # Same HP on the first sample after landing.

        self.assertFalse(self.danger.has_recent_damage(1.0))
        self.assertEqual(self.danger.danger_level(), DangerLevel.SAFE)
        self.assertFalse(self.ctx.danger_sit_requested.is_set())

    def test_damage_danger_keeps_sit_request_until_sit_worker_consumes_it(self) -> None:
        self.vitals.publish_hp(90, 100)
        self.danger._poll_hp()
        self.vitals.publish_hp(80, 100)
        self.danger._poll_hp()

        self.assertEqual(self.danger.danger_level(), DangerLevel.DANGER)
        self.assertTrue(self.ctx.danger_sit_requested.is_set())
        self.assertFalse(self.teleport.danger_teleport.called)

        self.assertTrue(self.ctx.pop_danger_sit_request())
        self.assertFalse(self.ctx.danger_sit_requested.is_set())

    def test_damage_danger_requests_sit_during_heal_ops(self) -> None:
        self.assertTrue(self.ctx.begin_heal_ops())
        self.assertFalse(self.ctx.should_run_combat())
        self.vitals.publish_hp(80, 100)
        self.danger._poll_hp()
        self.vitals.publish_hp(40, 100)
        self.danger._poll_hp()
        self.assertTrue(self.ctx.danger_sit_requested.is_set())
        self.teleport.danger_teleport.assert_not_called()
        self.ctx.end_heal_ops()

    def test_combat_blocked_during_discovery_suspend(self) -> None:
        self.ctx.mark_running()
        self.assertTrue(self.ctx.should_run_combat())
        self.ctx.discovery_suspend.set()
        self.assertFalse(self.ctx.should_run_combat())
        self.ctx.discovery_suspend.clear()
        self.assertTrue(self.ctx.should_run_combat())

    def test_successful_teleport_clears_damage_danger(self) -> None:
        danger = DangerDetector(self.ctx, vitals=self.vitals)
        self.vitals.publish_hp(90, 100)
        danger._poll_hp()
        self.vitals.publish_hp(80, 100)
        danger._poll_hp()

        self.ctx.config.teleport_scan_code = 16
        self.ctx.config.teleport_button = "q"
        self.ctx.config.teleport_duration_ms = 10
        self.ctx.danger_detector = danger
        tport = TeleportController(self.ctx, self.input, MagicMock())

        self.assertTrue(tport.teleport_once(scan_code=16))
        self.assertEqual(danger.danger_level(), DangerLevel.SAFE)
        # Teleport reset clears the old HP sample, but does not consume the
        # sit request owned by the danger/sit lifecycle.
        self.assertTrue(self.ctx.danger_sit_requested.is_set())
        self.assertTrue(self.ctx.pop_danger_sit_request())
        self.assertFalse(self.ctx.danger_sit_requested.is_set())

    def test_danger_teleport_resets_damage_state(self) -> None:
        danger = DangerDetector(self.ctx, vitals=self.vitals)
        self.vitals.publish_hp(90, 100)
        danger._poll_hp()
        self.vitals.publish_hp(80, 100)
        danger._poll_hp()

        self.ctx.config.teleport_scan_code = 16
        self.ctx.config.teleport_button = "q"
        self.ctx.config.teleport_duration_ms = 10
        self.ctx.danger_detector = danger
        tport = TeleportController(self.ctx, self.input, MagicMock())

        self.assertTrue(tport.danger_teleport(reason="damage"))
        self.ctx.tracks.area_reset.assert_called_once_with()
        self.assertEqual(danger.danger_level(), DangerLevel.SAFE)

    def test_failed_teleport_preserves_damage_danger(self) -> None:
        danger = DangerDetector(self.ctx, vitals=self.vitals)
        self.vitals.publish_hp(90, 100)
        danger._poll_hp()
        self.vitals.publish_hp(80, 100)
        danger._poll_hp()

        self.ctx.config.teleport_scan_code = 16
        self.ctx.config.teleport_button = "q"
        self.ctx.config.teleport_duration_ms = 10
        self.ctx.gates.wait_unless_stopped = MagicMock(return_value=False)
        self.ctx.danger_detector = danger
        tport = TeleportController(self.ctx, self.input, MagicMock())

        self.assertFalse(tport.teleport_once(scan_code=16))
        self.assertEqual(danger.danger_level(), DangerLevel.DANGER)
        self.assertTrue(self.ctx.danger_sit_requested.is_set())

    def test_danger_teleport_blocked_during_storage(self) -> None:
        self.assertTrue(self.ctx.begin_storage_ops())
        self.assertFalse(self.ctx.should_allow_danger_teleport())
        self.vitals.publish_hp(80, 100)
        self.danger._poll_hp()
        self.vitals.publish_hp(40, 100)
        self.danger._poll_hp()
        self.teleport.danger_teleport.assert_not_called()
        self.ctx.end_storage_ops()


if __name__ == "__main__":
    unittest.main()
