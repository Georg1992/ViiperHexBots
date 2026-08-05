"""DangerDetector seated-HP staleness diagnostic tests.

The diagnostic must be observation-only: it logs when the character is
sitting and the HP feed has stopped publishing, without ever changing
damage requests or gates.
"""

from __future__ import annotations

import time
import unittest
from unittest.mock import MagicMock

from pybot.game_state import PlayerVitals
from pybot.runtime.danger_detector import (
    DangerDetector,
    HP_STALE_LOG_INTERVAL_MS,
    HP_STALE_THRESHOLD_S,
)
from pybot.runtime.runtime_context import HuntRuntimeContext


class SeatedHpStaleLogTests(unittest.TestCase):
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

    def _make_hp_stale(self, stale_s: float = 60.0) -> None:
        """Publish a valid HP, then age the observation clock into the past."""
        self.vitals.publish_hp(3000, 3627)
        with self.vitals._lock:
            self.vitals._hp_observed_ms = int(
                time.monotonic() * 1000 - stale_s * 1000
            )

    def test_logs_when_seated_and_hp_feed_stale(self) -> None:
        self.ctx.sitting_event.set()
        self._make_hp_stale()
        self.danger._poll_hp()
        messages = [
            str(call) for call in self.ctx.logger.behavior.call_args_list
        ]
        self.assertTrue(
            any("HP feed stale while sitting" in m for m in messages),
            messages,
        )

    def test_silent_while_hunting_even_if_stale(self) -> None:
        self._make_hp_stale()
        self.danger._poll_hp()
        messages = [
            str(call) for call in self.ctx.logger.behavior.call_args_list
        ]
        self.assertFalse(
            any("HP feed stale while sitting" in m for m in messages),
            messages,
        )

    def test_silent_when_seated_and_hp_fresh(self) -> None:
        self.ctx.sitting_event.set()
        self.vitals.publish_hp(3000, 3627)
        self.danger._poll_hp()
        messages = [
            str(call) for call in self.ctx.logger.behavior.call_args_list
        ]
        self.assertFalse(
            any("HP feed stale while sitting" in m for m in messages),
            messages,
        )

    def test_silent_when_hp_unobserved_and_stale_but_hunting(self) -> None:
        # No HP ever published while hunting must stay silent too.
        self.danger._poll_hp()
        messages = [
            str(call) for call in self.ctx.logger.behavior.call_args_list
        ]
        self.assertFalse(
            any("HP feed stale while sitting" in m for m in messages),
            messages,
        )

    def test_throttled_repeat_logging(self) -> None:
        self.ctx.sitting_event.set()
        self._make_hp_stale()
        self.danger._poll_hp()
        self.danger._poll_hp()
        self.danger._poll_hp()
        messages = [
            str(call) for call in self.ctx.logger.behavior.call_args_list
        ]
        self.assertEqual(
            sum("HP feed stale while sitting" in m for m in messages), 1
        )
        # After the throttle window elapses the diagnostic repeats.
        self.danger._last_hp_stale_log_ms -= HP_STALE_LOG_INTERVAL_MS + 1
        self.danger._poll_hp()
        messages = [
            str(call) for call in self.ctx.logger.behavior.call_args_list
        ]
        self.assertEqual(
            sum("HP feed stale while sitting" in m for m in messages), 2
        )

    def test_stale_threshold_requests_one_emergency_escape(self) -> None:
        self.ctx.sitting_event.set()
        self._make_hp_stale()
        request = MagicMock()
        self.ctx.request_danger_sit = request

        self.danger._poll_hp()
        self.danger._poll_hp()

        request.assert_called_once_with()
        messages = [
            str(call) for call in self.ctx.logger.behavior.call_args_list
        ]
        self.assertTrue(
            any("requesting emergency escape" in m for m in messages),
            messages,
        )

    def test_stale_threshold_below_seconds_is_silent(self) -> None:
        self.ctx.sitting_event.set()
        self._make_hp_stale(stale_s=max(0.1, HP_STALE_THRESHOLD_S - 1.0))
        self.danger._poll_hp()
        messages = [
            str(call) for call in self.ctx.logger.behavior.call_args_list
        ]
        self.assertFalse(
            any("HP feed stale while sitting" in m for m in messages),
            messages,
        )


if __name__ == "__main__":
    unittest.main()
