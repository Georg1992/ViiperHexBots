"""Quiet-area helpers: clear + idle + recheck."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from pybot.runtime.clear_area import teleport_until_quiet


class TeleportUntilQuietTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ctx = MagicMock()
        self.ctx.is_stopped.return_value = False
        self.ctx.wait_unless_stopped.return_value = True
        self.ctx.logger = MagicMock()
        self.input = MagicMock()
        self.hunt_mode = MagicMock()

    @patch("pybot.runtime.clear_area.scan_living_count")
    @patch("pybot.runtime.clear_area.teleport_until_clear", return_value=True)
    def test_proceeds_when_still_clear_after_idle(
        self, _clear: MagicMock, scan: MagicMock
    ) -> None:
        scan.return_value = 0
        ok = teleport_until_quiet(
            self.ctx, self.input, self.hunt_mode, log_tag="STORAGE", idle_s=1.0
        )
        self.assertTrue(ok)
        self.ctx.wait_unless_stopped.assert_called_once_with(1.0)
        scan.assert_called_once()

    @patch("pybot.runtime.clear_area.scan_living_count")
    @patch("pybot.runtime.clear_area.teleport_until_clear", return_value=True)
    def test_retries_when_mobs_appear_during_idle(
        self, clear: MagicMock, scan: MagicMock
    ) -> None:
        scan.side_effect = [2, 0]
        ok = teleport_until_quiet(
            self.ctx, self.input, self.hunt_mode, log_tag="STORAGE", idle_s=1.0
        )
        self.assertTrue(ok)
        self.assertEqual(clear.call_count, 2)
        self.assertEqual(scan.call_count, 2)


    @patch("pybot.runtime.clear_area.scan_living_count")
    @patch("pybot.runtime.clear_area.teleport_until_clear", return_value=True)
    @patch("pybot.runtime.clear_area.force_teleport", return_value=True)
    def test_force_first_teleports_before_clear(
        self, force: MagicMock, clear: MagicMock, scan: MagicMock
    ) -> None:
        from pybot.runtime.clear_area import teleport_until_quiet

        scan.return_value = 0
        ok = teleport_until_quiet(
            self.ctx,
            self.input,
            self.hunt_mode,
            log_tag="SIT",
            idle_s=1.0,
            force_first=True,
        )
        self.assertTrue(ok)
        force.assert_called_once()
        clear.assert_called_once()

    @patch("pybot.runtime.clear_area.scan_living_count")
    @patch("pybot.runtime.clear_area.teleport_until_clear", return_value=True)
    @patch("pybot.runtime.clear_area.force_teleport", return_value=True)
    def test_without_force_first_skips_forced_tp(
        self, force: MagicMock, clear: MagicMock, scan: MagicMock
    ) -> None:
        from pybot.runtime.clear_area import teleport_until_quiet

        scan.return_value = 0
        ok = teleport_until_quiet(
            self.ctx,
            self.input,
            self.hunt_mode,
            log_tag="SIT",
            idle_s=1.0,
            force_first=False,
        )
        self.assertTrue(ok)
        force.assert_not_called()
        clear.assert_called_once()


if __name__ == "__main__":
    unittest.main()
