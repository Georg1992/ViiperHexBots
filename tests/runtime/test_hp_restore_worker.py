"""HP Restore worker — item key below threshold, or heal skill until full."""

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
        self.config.heal_skill = False
        self.config.skill_delay_ms = 200
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
        self.ctx.capture.is_valid.return_value = True
        self.ctx.capture.get_hunt_roi.return_value = MagicMock(
            x=100, y=200, w=400, h=300
        )
        self.input = MagicMock()
        self.vitals = PlayerVitals()

    def test_skips_when_no_hp_key(self) -> None:
        self.config.hp_scan_code = 0
        worker = HpRestoreWorker(self.ctx, self.input, self.vitals)
        worker.run()
        self.input.teleport_key.assert_not_called()

    def test_presses_when_hp_below_threshold(self) -> None:
        self.vitals.publish_hp(40, 100)
        worker = HpRestoreWorker(self.ctx, self.input, self.vitals)

        def stop_after_press(*_a, **_k):
            self.ctx.stop_event.set()
            return True

        self.input.teleport_key.side_effect = stop_after_press
        worker.run()
        self.input.teleport_key.assert_called_with(59)
        self.assertLess(40 / 100, HP_RESTORE_RATIO)

    def test_no_press_when_hp_ok(self) -> None:
        self.vitals.publish_hp(80, 100)
        worker = HpRestoreWorker(self.ctx, self.input, self.vitals)

        def stop_soon(*_a, **_k):
            self.ctx.stop_event.set()
            return False

        self.ctx.stop_event.wait = stop_soon  # type: ignore[method-assign]
        worker.run()
        self.input.teleport_key.assert_not_called()

    def test_heal_skill_casts_until_full_when_below_threshold(self) -> None:
        self.config.heal_skill = True
        self.vitals.publish_hp(40, 100)
        cast_count = {"n": 0}

        def cast_heal(_scan: int) -> bool:
            cast_count["n"] += 1
            if cast_count["n"] >= 2:
                self.vitals.publish_hp(100, 100)
            return True

        self.input.skill_click.side_effect = cast_heal

        def stop_when_full(*_a, **_k):
            hp, hp_max = self.vitals.hp_pair()
            if hp is not None and hp_max is not None and hp >= hp_max:
                self.ctx.stop_event.set()
            return False

        self.ctx.stop_event.wait = stop_when_full  # type: ignore[method-assign]
        worker = HpRestoreWorker(self.ctx, self.input, self.vitals)
        worker.run()

        self.assertGreaterEqual(cast_count["n"], 2)
        self.input.move_mouse.assert_called_with(300, 350)
        self.input.skill_click.assert_called_with(59)
        self.input.teleport_key.assert_not_called()
        self.assertFalse(self.ctx.healing_event.is_set())

    def test_heal_skill_idle_when_hp_above_threshold(self) -> None:
        self.config.heal_skill = True
        self.vitals.publish_hp(80, 100)

        def stop_soon(*_a, **_k):
            self.ctx.stop_event.set()
            return False

        self.ctx.stop_event.wait = stop_soon  # type: ignore[method-assign]
        worker = HpRestoreWorker(self.ctx, self.input, self.vitals)
        worker.run()
        self.input.skill_click.assert_not_called()
        self.input.teleport_key.assert_not_called()

    def test_heal_skill_waits_while_teleport_suspended(self) -> None:
        self.config.heal_skill = True
        self.vitals.publish_hp(40, 100)
        self.ctx.discovery_suspend.set()
        polls = {"n": 0}

        def stop_after_yield(*_a, **_k):
            polls["n"] += 1
            if polls["n"] >= 2:
                self.ctx.stop_event.set()
            return False

        self.ctx.stop_event.wait = stop_after_yield  # type: ignore[method-assign]
        worker = HpRestoreWorker(self.ctx, self.input, self.vitals)
        worker.run()
        self.input.skill_click.assert_not_called()
        self.assertFalse(self.ctx.healing_event.is_set())


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
        self.danger = DangerDetector(
            self.ctx, self.teleport, self.character_state, vitals=self.vitals,
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
