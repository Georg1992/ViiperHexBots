"""DangerDetector tests for strict HP-damage-only danger handling."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from pybot.game_state import PlayerVitals
from pybot.runtime.danger_detector import DangerDetector
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


if __name__ == "__main__":
    unittest.main()
