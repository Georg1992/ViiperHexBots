"""Quiet-area helpers: clear + idle + recheck (via TeleportController)."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from pybot.runtime.constants import SIT_SP_POLL_INTERVAL_S
from pybot.runtime.danger_detector import DangerLevel
from pybot.runtime.teleport import TeleportController


class TeleportUntilQuietTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ctx = MagicMock()
        self.ctx.is_stopped.return_value = False
        self.ctx.wait_unless_stopped.return_value = True
        self.ctx.logger = MagicMock()
        self.ctx.capture.is_valid.return_value = False
        self.ctx.danger_detector = MagicMock()
        self.ctx.danger_detector.danger_level.return_value = DangerLevel.SAFE
        self.input = MagicMock()
        self.hunt_mode = MagicMock()
        self.tport = TeleportController(self.ctx, self.input, self.hunt_mode)

    def test_proceeds_when_still_clear_after_idle(self) -> None:
        self.tport._scan_living_count = MagicMock(return_value=0)  # type: ignore[method-assign]
        self.tport.teleport_until_clear = MagicMock(return_value=True)  # type: ignore[method-assign]

        clock = [0.0]

        def fake_monotonic() -> float:
            return clock[0]

        def advance_clock(timeout_s: float) -> bool:
            clock[0] += timeout_s
            return True

        self.ctx.wait_unless_stopped.side_effect = advance_clock
        with patch("pybot.runtime.teleport.time.monotonic", side_effect=fake_monotonic):
            ok = self.tport.teleport_until_quiet(
                log_tag="STORAGE", idle_s=1.0
            )
        self.assertTrue(ok)
        waits = [call.args[0] for call in self.ctx.wait_unless_stopped.call_args_list]
        self.assertGreater(len(waits), 1)
        self.assertTrue(all(0 < timeout <= SIT_SP_POLL_INTERVAL_S for timeout in waits))
        self.tport._scan_living_count.assert_called_once()

    def test_retries_when_mobs_appear_during_idle(self) -> None:
        self.tport._scan_living_count = MagicMock(side_effect=[2, 0])  # type: ignore[method-assign]
        self.tport.teleport_until_clear = MagicMock(return_value=True)  # type: ignore[method-assign]
        ok = self.tport.teleport_until_quiet(
            log_tag="STORAGE", idle_s=1.0
        )
        self.assertTrue(ok)
        self.assertEqual(self.tport.teleport_until_clear.call_count, 2)
        self.assertEqual(self.tport._scan_living_count.call_count, 2)

    def test_danger_interrupts_idle_before_post_idle_scan(self) -> None:
        self.tport._scan_living_count = MagicMock(return_value=0)  # type: ignore[method-assign]
        self.tport.teleport_until_clear = MagicMock(return_value=True)  # type: ignore[method-assign]

        def wait_and_raise(_timeout: float) -> bool:
            self.ctx.danger_detector.danger_level.return_value = DangerLevel.DANGER
            return True

        self.ctx.wait_unless_stopped.side_effect = wait_and_raise

        self.assertFalse(
            self.tport.teleport_until_quiet(log_tag="SIT", idle_s=1.0)
        )
        self.tport._scan_living_count.assert_not_called()
        self.tport.teleport_until_clear.assert_called_once_with(log_tag="SIT")


if __name__ == "__main__":
    unittest.main()
