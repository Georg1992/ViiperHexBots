"""HP item worker and danger detector regression tests."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from pybot.game_state import PlayerVitals
from pybot.runtime.constants import HP_RESTORE_RATIO
from pybot.runtime.danger_detector import CRITICAL_HP_RATIO, DangerDetector
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
        self.vitals = PlayerVitals()
        self.character_state = MagicMock()
        self.character_state.is_surrounded = False
        self.character_state.nearby_mob_count = 0
        self.character_state.nearby_any_mobs_count = 0
        self.danger = DangerDetector(
            self.ctx, self.teleport, self.character_state, vitals=self.vitals
        )

    def test_critical_ratio_is_internal_half(self) -> None:
        self.assertEqual(CRITICAL_HP_RATIO, 0.5)

    def test_critical_drop_teleports_without_heal_coupling(self) -> None:
        self.vitals.publish_hp(80, 100)
        self.danger._poll_hp()
        self.vitals.publish_hp(40, 100)
        self.danger._poll_hp()
        self.teleport.danger_teleport.assert_called_with(reason="critical_hp")
        self.assertFalse(hasattr(self.danger, "pop_heal_until_full_requested"))

    def test_non_critical_hp_drop_does_not_urgent_teleport(self) -> None:
        self.vitals.publish_hp(90, 100)
        self.danger._poll_hp()
        self.vitals.publish_hp(80, 100)
        self.danger._poll_hp()
        self.teleport.danger_teleport.assert_not_called()
        self.assertTrue(self.danger.has_recent_damage(1.0))

    def test_surround_does_not_urgent_teleport(self) -> None:
        self.assertFalse(hasattr(self.danger, "_poll_surround"))
        self.character_state.is_surrounded = True
        self.character_state.surrounded_reason = "above+below"
        self.teleport.danger_teleport.assert_not_called()

    def test_nearby_mobs_are_heal_threat(self) -> None:
        self.assertFalse(self.danger.has_nearby_threat())
        self.character_state.nearby_any_mobs_count = 2
        self.assertTrue(self.danger.has_nearby_threat())
        self.character_state.nearby_any_mobs_count = 0
        self.character_state.nearby_mob_count = 1
        self.assertTrue(self.danger.has_nearby_threat())

    def test_recent_damage_is_time_based(self) -> None:
        self.ctx.mark_paused()
        self.vitals.publish_hp(80, 100)
        self.danger._poll_hp()
        self.vitals.publish_hp(70, 100)
        self.danger._poll_hp()
        self.assertTrue(self.danger.has_recent_damage(1.0))
        self.assertFalse(self.danger.has_recent_damage(0.0))
        self.teleport.danger_teleport.assert_not_called()

    def test_danger_teleport_allowed_during_heal_ops(self) -> None:
        self.assertTrue(self.ctx.begin_heal_ops())
        self.assertFalse(self.ctx.should_run_combat())
        self.assertTrue(self.ctx.should_allow_danger_teleport())
        self.vitals.publish_hp(80, 100)
        self.danger._poll_hp()
        self.vitals.publish_hp(40, 100)
        self.danger._poll_hp()
        self.teleport.danger_teleport.assert_called_with(reason="critical_hp")
        self.ctx.end_heal_ops()

    def test_combat_blocked_during_discovery_suspend(self) -> None:
        self.ctx.mark_running()
        self.assertTrue(self.ctx.should_run_combat())
        self.ctx.discovery_suspend.set()
        self.assertFalse(self.ctx.should_run_combat())
        self.ctx.discovery_suspend.clear()
        self.assertTrue(self.ctx.should_run_combat())

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
