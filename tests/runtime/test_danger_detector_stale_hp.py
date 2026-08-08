"""DangerDetector tests for strict HP-damage-only danger handling."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from pybot.game_state import player_vitals as pv_mod

from pybot.game_state import PlayerVitals
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
        self.danger = DangerDetector(self.ctx, vitals=self.vitals)

    def test_unreadable_sample_is_not_a_danger_event(self) -> None:
        self.vitals.publish_hp(90, 100)
        self.danger._poll_hp()
        self.vitals.publish_hp(None, None)
        self.danger._poll_hp()
        self.vitals.publish_hp(80, 100)
        self.danger._poll_hp()

        self.assertTrue(self.danger.has_recent_damage(1.0))
        self.assertFalse(self.ctx.danger_sit_requested.is_set())

    def test_only_real_hp_drop_queues_one_danger_event(self) -> None:
        self.ctx.sitting_event.set()
        self.vitals.publish_hp(90, 100)
        self.danger._poll_hp()
        self.vitals.publish_hp(80, 100)
        self.danger._poll_hp()

        self.assertTrue(self.ctx.danger_sit_requested.is_set())
        self.ctx.pop_danger_sit_request()

        # Re-reading the same damaged HP is not a second damage episode.
        self.danger._poll_hp()
        self.assertFalse(self.ctx.danger_sit_requested.is_set())

    def test_repeated_same_hp_never_requests_danger(self) -> None:
        self.ctx.sitting_event.set()
        self.vitals.publish_hp(80, 100)
        self.danger._poll_hp()
        self.danger._poll_hp()
        self.danger._poll_hp()

        self.assertFalse(self.ctx.danger_sit_requested.is_set())
        self.assertFalse(self.ctx.critical_danger_requested.is_set())
        self.ctx.logger.behavior.assert_not_called()

    def test_stale_pre_teleport_hp_never_becomes_baseline(self) -> None:
        """A stale pre-teleport sample cannot cause a phantom post-landing
        drop — the re-escape loop.

        After a recovery/escape teleport the OCR feed is blind at reset
        time, so the stored HP is still the pre-teleport value. Rebaselining
        from it makes the first fresh landing reading look like damage, the
        seated session urgently escapes, and the loop repeats forever.
        """
        with patch.object(pv_mod.time, "monotonic", return_value=1000.0):
            self.vitals.publish_hp(90, 100)
            self.danger._poll_hp()
        teleport_started = 1001.0
        # The feed is blind the instant the teleport lands: nothing was
        # published after the teleport key was pressed, so the stored
        # sample is pre-teleport and the baseline must stay unknown.
        with patch.object(pv_mod.time, "monotonic", return_value=1001.0):
            self.danger.reset_after_teleport(teleport_started)
        self.assertIsNone(self.danger._prev_hp)

        # The first fresh landing reading establishes the baseline — no
        # phantom damage even though it differs from the pre-teleport value.
        with patch.object(pv_mod.time, "monotonic", return_value=1002.0):
            self.vitals.publish_hp(70, 100)
            self.danger._poll_hp()
            self.assertFalse(self.danger.has_recent_damage(1.0))
            self.assertEqual(self.danger.danger_level(), DangerLevel.SAFE)
        self.assertFalse(self.ctx.danger_sit_requested.is_set())
        self.assertEqual(self.danger._prev_hp, 70)

        # A genuine drop after the baseline is still real damage.
        with patch.object(pv_mod.time, "monotonic", return_value=1003.0):
            self.vitals.publish_hp(60, 100)
            self.danger._poll_hp()
        with patch.object(pv_mod.time, "monotonic", return_value=1003.1):
            self.assertTrue(self.danger.has_recent_damage(1.0))

    def test_fresh_post_teleport_sample_becomes_baseline(self) -> None:
        """A fresh post-landing reading is a valid new baseline."""
        with patch.object(pv_mod.time, "monotonic", return_value=1000.0):
            self.vitals.publish_hp(90, 100)
            self.danger._poll_hp()
        teleport_started = 1001.0
        # A fresh reading arrives during settle (published after the
        # teleport key was pressed) — it may serve as the baseline.
        with patch.object(pv_mod.time, "monotonic", return_value=1001.5):
            self.vitals.publish_hp(90, 100)
            self.danger.reset_after_teleport(teleport_started)
        self.assertEqual(self.danger._prev_hp, 90)

        with patch.object(pv_mod.time, "monotonic", return_value=1002.0):
            self.vitals.publish_hp(80, 100)
            self.danger._poll_hp()
        with patch.object(pv_mod.time, "monotonic", return_value=1002.1):
            self.assertTrue(self.danger.has_recent_damage(1.0))


if __name__ == "__main__":
    unittest.main()
