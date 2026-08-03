"""Teleport key selection: mode uses creamy-first; danger uses wing-first."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from pybot.runtime.teleport import TeleportController


class TeleportKeySelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ctx = MagicMock()
        self.ctx.config.teleport_button = "q"
        self.ctx.config.teleport_scan_code = 16
        self.ctx.config.creamy_tp_button = "w"
        self.ctx.config.creamy_tp_scan_code = 17
        self.ctx.config.teleport_duration_ms = 10
        self.ctx.wait_unless_stopped.return_value = True
        self.ctx.logger = MagicMock()
        self.input = MagicMock()
        self.tport = TeleportController(self.ctx, self.input, MagicMock())

    def test_mode_prefers_creamy_when_assigned(self) -> None:
        self.assertEqual(self.tport.active_scan_code(), 17)
        self.assertEqual(self.tport.active_button(), "w")

    def test_danger_prefers_wing_when_assigned(self) -> None:
        self.assertEqual(self.tport.danger_scan_code(), 16)
        self.assertEqual(self.tport.danger_button(), "q")

    def test_danger_uses_creamy_when_wing_unassigned(self) -> None:
        self.ctx.config.teleport_scan_code = 0
        self.ctx.config.teleport_button = ""
        self.assertEqual(self.tport.danger_scan_code(), 17)
        self.assertEqual(self.tport.danger_button(), "w")

    def test_danger_uses_creamy_when_wing_button_blank_even_if_scan_set(self) -> None:
        """Blank Teleport Key must not count as assigned."""
        self.ctx.config.teleport_scan_code = 16
        self.ctx.config.teleport_button = "   "
        self.assertEqual(self.tport.danger_scan_code(), 17)
        self.assertEqual(self.tport.danger_button(), "w")

    def test_danger_teleport_resets_hunt_mode_discovery_state(self) -> None:
        self.ctx.config.teleport_scan_code = 16
        self.ctx.config.teleport_button = "q"
        self.ctx.config.teleport_duration_ms = 10
        self.ctx.danger_detector = MagicMock()
        strategy = MagicMock()
        tport = TeleportController(self.ctx, self.input, strategy)

        self.assertTrue(tport.danger_teleport(reason="critical_hunt"))
        strategy.on_area_reset.assert_called_once_with()

    def test_danger_teleport_presses_wing_key(self) -> None:
        self.tport.danger_teleport(reason="critical_hp")
        self.input.teleport_key.assert_called_once_with(16)
        self.ctx.note_teleport_for_wings.assert_called_once()

    def test_danger_teleport_creamy_when_wing_unassigned(self) -> None:
        self.ctx.config.teleport_scan_code = 0
        self.ctx.config.teleport_button = ""
        self.tport.danger_teleport(reason="critical_hp")
        self.input.teleport_key.assert_called_once_with(17)
        self.ctx.note_teleport_for_wings.assert_not_called()

    def test_mode_teleport_once_uses_creamy_and_does_not_count_wing(self) -> None:
        self.assertTrue(self.tport.teleport_once())
        self.input.teleport_key.assert_called_once_with(17)
        self.ctx.note_teleport_for_wings.assert_not_called()


if __name__ == "__main__":
    unittest.main()
