"""Teleport key selection: mode/SAFE danger use creamy-first; urgent danger uses wing-first."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from pybot.game_state import PlayerVitals
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
        # Not exhausted: the wing key remains the urgent escape until
        # GetFlyWings reports storage has no wings left.
        self.ctx.fly_wings_exhausted = False
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

    def test_active_button_ignores_whitespace_only_creamy_binding(self) -> None:
        self.ctx.config.creamy_tp_button = "   "
        self.assertEqual(self.tport.active_button(), "q")

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

    def test_safe_danger_teleport_prefers_creamy_when_assigned(self) -> None:
        """Recovery-session escapes use the safe key (creamy / save point)."""
        self.tport.danger_teleport(
            reason="sit_danger",
            prefer_safe_key=True,
        )
        self.input.teleport_key.assert_called_once_with(17)
        self.ctx.note_teleport_for_wings.assert_not_called()

    def test_safe_danger_teleport_falls_back_to_wing_without_creamy(self) -> None:
        self.ctx.config.creamy_tp_scan_code = 0
        self.ctx.config.creamy_tp_button = ""
        self.tport.danger_teleport(
            reason="sit_danger",
            prefer_safe_key=True,
        )
        self.input.teleport_key.assert_called_once_with(16)
        self.ctx.note_teleport_for_wings.assert_called_once()

    def test_urgent_danger_teleport_still_prefers_wing(self) -> None:
        """Hunting critical escapes keep the urgent random fly wing."""
        self.tport.danger_teleport(reason="critical_hunt")
        self.input.teleport_key.assert_called_once_with(16)

    def test_danger_scan_code_prefers_creamy_when_wings_exhausted(self) -> None:
        """Once wings are gone, the critical escape key is Creamy TP."""
        self.ctx.fly_wings_exhausted = True
        self.assertEqual(self.tport.danger_scan_code(), 17)
        self.assertEqual(self.tport.danger_button(), "w")

    def test_danger_teleport_uses_creamy_when_wings_exhausted(self) -> None:
        """A no-op wing key must never be pressed after exhaustion."""
        self.ctx.fly_wings_exhausted = True
        self.tport.danger_teleport(reason="critical_hunt")
        self.input.teleport_key.assert_called_once_with(17)
        self.ctx.note_teleport_for_wings.assert_not_called()

    def test_danger_teleport_keeps_wing_when_exhausted_but_no_creamy(self) -> None:
        """Without a creamy binding the wing key remains the last-resort escape."""
        self.ctx.fly_wings_exhausted = True
        self.ctx.config.creamy_tp_scan_code = 0
        self.ctx.config.creamy_tp_button = ""
        self.assertEqual(self.tport.danger_button(), "q")
        self.tport.danger_teleport(reason="critical_hunt")
        self.input.teleport_key.assert_called_once_with(16)

    def test_mode_teleport_once_uses_creamy_and_does_not_count_wing(self) -> None:
        self.assertTrue(self.tport.teleport_once())
        self.input.teleport_key.assert_called_once_with(17)
        self.ctx.note_teleport_for_wings.assert_not_called()

    def test_mode_teleport_with_only_wing_key_uses_and_counts_wing(self) -> None:
        """Area-clear teleports fall back to the wing key and count it when no creamy is bound."""
        self.ctx.config.creamy_tp_button = ""
        self.ctx.config.creamy_tp_scan_code = 0
        self.assertTrue(self.tport.teleport_once())
        self.input.teleport_key.assert_called_once_with(16)
        self.ctx.note_teleport_for_wings.assert_called_once()

    def test_explicit_teleport_code_is_rejected_when_binding_was_cleared(self) -> None:
        self.ctx.config.creamy_tp_button = ""
        self.ctx.config.creamy_tp_scan_code = 17
        self.assertFalse(self.tport.teleport_once(scan_code=17))
        self.input.teleport_key.assert_not_called()

    def test_successful_teleport_invalidates_stale_vitals_and_reopens_reader(self) -> None:
        vitals = PlayerVitals()
        vitals.publish_sp(574, 1454)
        tport = TeleportController(
            self.ctx,
            self.input,
            MagicMock(),
            vitals=vitals,
        )

        self.assertTrue(tport.teleport_once(scan_code=17))
        self.assertIsNone(vitals.sp)
        self.assertIsNone(vitals.sp_max)
        # The reader remains alive across the transition. Once settle completes,
        # the new epoch must accept a genuinely fresh value again.
        epoch = vitals.observation_epoch
        self.assertTrue(
            vitals.publish_sp_if_current(350, 1454, epoch),
        )
        self.assertEqual(vitals.sp_pair(), (350, 1454))

    def test_interrupted_settle_reopens_observation_epoch(self) -> None:
        """Stop during settle must not leave app-scoped vitals quarantined."""
        vitals = PlayerVitals()
        vitals.publish_sp(574, 1454)
        self.ctx.wait_unless_stopped.return_value = False
        tport = TeleportController(
            self.ctx,
            self.input,
            MagicMock(),
            vitals=vitals,
        )

        self.assertFalse(tport.teleport_once(scan_code=17))
        epoch = vitals.observation_epoch
        self.assertTrue(vitals.publish_sp_if_current(350, 1454, epoch))
        self.assertEqual(vitals.sp_pair(), (350, 1454))

    def test_teleport_once_clears_tracks_after_settle(self) -> None:
        """Every successful teleport must drop prior-area tracks."""
        self.ctx.danger_detector = MagicMock()
        self.assertTrue(self.tport.teleport_once())
        self.ctx.area_reset.assert_called_once_with("teleport")
        self.ctx.overlay.set_track_positions.assert_called_once_with([])

    def _set_escape_in_flight(self) -> None:
        self.ctx.danger_escape_active = MagicMock()
        self.ctx.danger_escape_active.is_set.return_value = True

    def test_placement_teleport_refuses_during_critical_escape(self) -> None:
        """A sit placement must never press a teleport key mid-escape."""
        self._set_escape_in_flight()
        self.assertFalse(self.tport.teleport_once_for_sit(log_tag="SIT"))
        self.input.teleport_key.assert_not_called()

    def test_mode_teleport_refuses_during_critical_escape(self) -> None:
        self._set_escape_in_flight()
        self.assertFalse(self.tport.mode_teleport())
        self.input.teleport_key.assert_not_called()
        self.ctx.discovery_suspend.set.assert_not_called()

    def _enable_sit_recovery(self, vitals: PlayerVitals | None) -> TeleportController:
        self.ctx.config.sit_on_low_sp = True
        self.ctx.config.sit_on_low_sp_button = "insert"
        self.ctx.config.sit_on_low_sp_scan_code = 82
        self.ctx.tracks.can_claim_clear_for_teleport.return_value = True
        return TeleportController(
            self.ctx,
            self.input,
            MagicMock(),
            vitals=vitals,
        )

    def test_mode_teleport_refuses_when_sp_is_low(self) -> None:
        vitals = PlayerVitals()
        vitals.publish_sp(4, 100)
        tport = self._enable_sit_recovery(vitals)
        self.assertEqual(tport.hunt_teleport_blocked_reason(), "low_sp")
        self.assertFalse(tport.mode_teleport())
        self.input.teleport_key.assert_not_called()

    def test_mode_teleport_refuses_when_sp_is_unread(self) -> None:
        tport = self._enable_sit_recovery(PlayerVitals())
        self.assertEqual(tport.hunt_teleport_blocked_reason(), "sp_unknown")
        self.assertFalse(tport.mode_teleport())
        self.input.teleport_key.assert_not_called()

    def test_mode_teleport_proceeds_when_sp_is_healthy(self) -> None:
        vitals = PlayerVitals()
        vitals.publish_sp(50, 100)
        tport = self._enable_sit_recovery(vitals)
        self.assertIsNone(tport.hunt_teleport_blocked_reason())
        self.assertTrue(tport.mode_teleport())
        self.input.teleport_key.assert_called_once()

    def test_danger_and_sit_placement_still_teleport_when_sp_is_low(self) -> None:
        vitals = PlayerVitals()
        vitals.publish_sp(1, 100)
        tport = self._enable_sit_recovery(vitals)
        self.assertTrue(tport.danger_teleport(reason="critical_hunt"))
        self.input.teleport_key.assert_called_once_with(16)
        self.input.teleport_key.reset_mock()
        vitals.publish_sp(1, 100)
        self.assertTrue(tport.teleport_once_for_sit(log_tag="SIT"))
        self.input.teleport_key.assert_called_once_with(17)


if __name__ == "__main__":
    unittest.main()
