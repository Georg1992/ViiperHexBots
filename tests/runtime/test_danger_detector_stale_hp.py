"""DangerDetector tests for strict HP-damage-only danger handling."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from pybot.game_state import PlayerVitals
from pybot.game_state import player_vitals as pv_mod
from pybot.runtime.danger_detector import DangerDetector, DangerLevel
from pybot.runtime.runtime_context import HuntRuntimeContext


class DangerDetectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ctx = HuntRuntimeContext(
            config=MagicMock(),
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
        self.vitals = PlayerVitals()
        self.danger_wake = self.ctx.danger_wake
        self.danger = DangerDetector(
            self.ctx,
            vitals=self.vitals,
            wake_event=self.danger_wake,
        )

    def test_unreadable_sample_is_not_a_danger_event(self) -> None:
        self.vitals.publish_hp(90, 100)
        self.danger._poll_hp()
        self.vitals.publish_hp(None, None)
        self.danger._poll_hp()

        self.assertEqual(self.danger.damage_sequence, 0)
        self.assertFalse(self.danger.has_recent_damage(1.0))
        self.assertFalse(self.danger_wake.is_set())

    def test_only_real_hp_drop_updates_fact_and_wakes_owner(self) -> None:
        self.vitals.publish_hp(90, 100)
        self.danger._poll_hp()
        self.vitals.publish_hp(80, 100)
        self.danger._poll_hp()

        self.assertEqual(self.danger.damage_sequence, 1)
        self.assertTrue(self.danger.has_recent_damage(1.0))
        self.assertEqual(self.danger.danger_level(), DangerLevel.DANGER)
        self.assertTrue(self.danger_wake.is_set())
        # The observer is fact-only: it never presses keys or requests a sit.
        self.ctx.cancel_gameplay_input = MagicMock()
        self.ctx.cancel_gameplay_input.assert_not_called()

    def test_repeated_same_hp_never_creates_second_damage_fact(self) -> None:
        self.vitals.publish_hp(80, 100)
        self.danger._poll_hp()
        self.danger._poll_hp()
        self.danger._poll_hp()

        self.assertEqual(self.danger.damage_sequence, 0)
        self.assertFalse(self.danger_wake.is_set())
        self.ctx.logger.behavior.assert_not_called()

    def test_stale_pre_teleport_hp_never_becomes_baseline(self) -> None:
        """A stale pre-teleport sample cannot cause a phantom damage drop."""
        with patch.object(pv_mod.time, "monotonic", return_value=1000.0):
            self.vitals.publish_hp(90, 100)
            self.danger._poll_hp()

        teleport_started = 1001.0
        epoch = self.vitals.begin_observation_epoch()
        self.danger.reset_after_teleport(teleport_started)
        self.assertIsNone(self.danger._prev_hp)

        # The first fresh landing reading establishes the baseline, even though
        # it differs from the previous area's HP.
        self.assertTrue(self.vitals.complete_observation_epoch(epoch))
        with patch.object(pv_mod.time, "monotonic", return_value=1002.0):
            self.assertTrue(self.vitals.publish_hp_if_current(70, 100, epoch))
            self.danger._poll_hp()
            self.assertFalse(self.danger.has_recent_damage(1.0))
            self.assertEqual(self.danger.danger_level(), DangerLevel.SAFE)
        self.assertEqual(self.danger._prev_hp, 70)

        # A genuine drop after the new baseline remains real damage.
        with patch.object(pv_mod.time, "monotonic", return_value=1003.0):
            self.assertTrue(self.vitals.publish_hp_if_current(60, 100, epoch))
            self.danger._poll_hp()
        with patch.object(pv_mod.time, "monotonic", return_value=1003.1):
            self.assertTrue(self.danger.has_recent_damage(1.0))

    def test_fresh_post_teleport_sample_becomes_baseline(self) -> None:
        """A fresh post-landing reading is a valid new baseline."""
        with patch.object(pv_mod.time, "monotonic", return_value=1000.0):
            self.vitals.publish_hp(90, 100)
            self.danger._poll_hp()

        teleport_started = 1001.0
        epoch = self.vitals.begin_observation_epoch()
        self.assertTrue(self.vitals.complete_observation_epoch(epoch))
        with patch.object(pv_mod.time, "monotonic", return_value=1001.5):
            self.assertTrue(self.vitals.publish_hp_if_current(90, 100, epoch))
            self.danger.reset_after_teleport(teleport_started)
        self.assertEqual(self.danger._prev_hp, 90)

        with patch.object(pv_mod.time, "monotonic", return_value=1002.0):
            self.assertTrue(self.vitals.publish_hp_if_current(80, 100, epoch))
            self.danger._poll_hp()
        with patch.object(pv_mod.time, "monotonic", return_value=1002.1):
            self.assertTrue(self.danger.has_recent_damage(1.0))

    def test_damage_observed_after_teleport_start_is_not_erased(self) -> None:
        """Resetting the baseline must not discard a landing hit."""
        with patch.object(pv_mod.time, "monotonic", side_effect=[1000.0, 1000.0, 1001.2, 1001.2]):
            self.vitals.publish_hp(90, 100)
            self.danger._poll_hp()
            self.vitals.publish_hp(70, 100)
            self.danger._poll_hp()
            self.danger.reset_after_teleport(1001.0)
            self.assertTrue(self.danger.has_recent_damage(1.0))

        # The landing hit remains the active danger fact; it is not erased by
        # the post-teleport baseline reset.
        self.assertEqual(self.danger._last_damage_ratio, 20 / 90)
