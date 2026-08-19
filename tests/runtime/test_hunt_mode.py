"""HuntMode teleport / area-clear tests."""

from __future__ import annotations

import threading
import unittest
from unittest.mock import MagicMock

from pybot.game_state import PlayerVitals
from pybot.runtime.capture.hunt_capture import HuntWindowCapture
from pybot.paths import PROJECT_ROOT
from pybot.config.runtime import HuntRuntimeConfig
from pybot.runtime.control import RuntimeControl
from pybot.runtime.hunt_mode import create_hunt_mode
from pybot.runtime.hunt_policy import HuntPolicy
from pybot.runtime.hunt_tracks import HuntTracks, monotonic_ms
from pybot.runtime.input.input_backend import ShadowInputBackend
from pybot.runtime.logging import HuntLogger
from pybot.runtime.runtime_context import HuntRuntimeContext
from pybot.runtime.validation_log import HuntValidationLogger
from pybot.runtime.detection.detector_session import DetectorSession
from pybot.runtime.teleport import TeleportController


def _make_tport(ctx, input_backend, vitals=None):
    """Create a TeleportController for tests with a mocked hunt_mode."""
    from unittest.mock import MagicMock
    tport = TeleportController(ctx, input_backend, MagicMock(), vitals=vitals)
    # Only mock the scan code lookup — teleport_once uses the real ctx.
    tport.active_scan_code = MagicMock(return_value=16)  # type: ignore[method-assign]
    return tport


def make_config(**overrides) -> HuntRuntimeConfig:
    base = {
        "config_path": PROJECT_ROOT / "config.ini",
        "hwnd": 123,
        "mob_name": "horn",
        "hunt_mode": "teleport",
        "skill_delay_ms": 500,
        "skill_button": "e",
        "skill_scan_code": 18,
        "teleport_button": "q",
        "teleport_scan_code": 16,
        "take_fly_wings": True,
        "search_range_cells": 16,
        "cell_size_px": 64,
        "discovery_interval_ms": 3000,
        "teleport_duration_ms": 500,
        "validation_enabled": False,
        "control_file": None,
    }
    base.update(overrides)
    return HuntRuntimeConfig(**base)


class HuntModeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = make_config()
        self.logger = HuntLogger(session_id="test_hunt_mode")
        self.tracks = HuntTracks()
        self.detector = MagicMock(spec=DetectorSession)
        self.ctx = HuntRuntimeContext(
            config=self.config,
            logger=self.logger,
            tracks=self.tracks,
            policy=HuntPolicy(),
            capture=MagicMock(spec=HuntWindowCapture),
            detector=self.detector,
            tracker=self.detector,
            validation=HuntValidationLogger(self.logger, self.tracks, enabled=False),
            control=RuntimeControl(None),
        )
        self.mode = create_hunt_mode(
            self.ctx, ShadowInputBackend(),
            teleport_controller=_make_tport(self.ctx, ShadowInputBackend()),
        )

    def test_blocks_teleport_without_discovery(self) -> None:
        teleported = self.mode.on_no_attackable_targets()
        self.assertFalse(teleported)
        self.assertEqual(self.tracks.get_track_count(), 0)

    def test_does_not_teleport_when_startup_gate_changes_before_commit(self) -> None:
        self.mode.note_discovery_scan_completed(
            living_count=0,
            added_count=0,
            area_epoch=self.tracks.area_epoch,
        )
        self.ctx.gates.startup.begin(require_buffs=True, require_timers=True)
        self.ctx.gates.startup.mark_area_clear()
        self.ctx.gates.startup.mark_buffs_done()
        self.ctx.gates.startup.mark_timers_done()
        self.ctx.gates.startup.timers_done.clear()
        self.ctx.should_run_combat = MagicMock(side_effect=[True, False])

        # The first check admits the stale no-target decision; the final
        # admission check must observe that startup timers became pending.
        self.assertFalse(self.mode.on_no_attackable_targets())
        self.assertEqual(self.tracks.area_epoch, 0)

    def test_mode_teleport_shares_lifecycle_admission_boundary(self) -> None:
        self.mode.note_discovery_scan_completed(
            living_count=0,
            added_count=0,
            area_epoch=self.tracks.area_epoch,
        )
        teleport = self.mode._strategy._teleport
        teleport.mode_teleport = MagicMock(return_value=True)

        self.ctx.gates._sit_storage_lock.acquire()
        try:
            result: list[bool] = []
            thread = threading.Thread(
                target=lambda: result.append(self.mode.on_no_attackable_targets()),
                daemon=True,
            )
            thread.start()
            thread.join(timeout=0.05)
            self.assertTrue(thread.is_alive())
        finally:
            self.ctx.gates._sit_storage_lock.release()

        thread.join(timeout=1.0)
        self.assertFalse(thread.is_alive())
        self.assertEqual(result, [True])
        teleport.mode_teleport.assert_called_once_with()

    def test_shadow_teleport_on_area_clear(self) -> None:
        self.mode.note_discovery_scan_completed(
            living_count=0,
            added_count=0,
            area_epoch=self.tracks.area_epoch,
        )
        teleported = self.mode.on_no_attackable_targets()
        self.assertTrue(teleported)
        self.assertEqual(self.tracks.get_track_count(), 0)
        # The accepted teleport owns the single reset; no pre-key mutation is
        # needed now that the key can be rejected.
        self.assertEqual(self.tracks.area_epoch, 1)
        # Post-settle: discovery may scan again; suspend must be clear + wake set.
        self.assertFalse(self.ctx.discovery_suspend.is_set())
        self.assertTrue(self.ctx.discovery_wake.is_set())

    def test_rejected_mode_teleport_preserves_current_area_and_wake(self) -> None:
        """A rejected key must not reset or partially mutate the hunt area."""
        self.mode.note_discovery_scan_completed(
            living_count=0,
            added_count=0,
            area_epoch=self.tracks.area_epoch,
        )
        self.ctx.attack_wake.set()
        teleport = self.mode._strategy._teleport
        teleport.active_scan_code = MagicMock(return_value=0)

        self.assertFalse(self.mode.on_no_attackable_targets())
        self.assertEqual(self.tracks.area_epoch, 0)
        self.assertTrue(self.ctx.attack_wake.is_set())
        self.assertTrue(self.mode.discovery_since_reset)

    def test_suspends_discovery_during_teleport_delay(self) -> None:
        self.mode.note_discovery_scan_completed(
            living_count=0,
            added_count=0,
            area_epoch=self.tracks.area_epoch,
        )
        # Hold the settle wait so we can observe suspend mid-teleport.
        gate = {"released": False}
        original_wait = self.ctx.wait_unless_stopped

        def _hold_settle(timeout_s: float) -> bool:
            self.assertTrue(self.ctx.discovery_suspend.is_set())
            self.assertFalse(self.ctx.discovery_wake.is_set())
            gate["released"] = True
            return original_wait(0.01)

        self.ctx.wait_unless_stopped = _hold_settle  # type: ignore[method-assign]
        self.assertTrue(self.mode.on_no_attackable_targets())
        self.assertTrue(gate["released"])
        self.assertFalse(self.ctx.discovery_suspend.is_set())
        self.assertTrue(self.ctx.discovery_wake.is_set())

    def test_blocks_teleport_until_discovery_confirms_clear(self) -> None:
        # Discovery saw living mobs earlier; tracks later empty must not teleport
        # until a scan reports living_count == 0.
        self.mode.note_discovery_scan_completed(
            living_count=2,
            added_count=2,
            area_epoch=self.tracks.area_epoch,
        )
        self.assertTrue(self.mode.discovery_since_reset)
        self.assertFalse(self.mode.discovery_confirmed_clear)
        self.ctx.discovery_wake.clear()
        teleported = self.mode.on_no_attackable_targets()
        self.assertFalse(teleported)
        self.assertTrue(self.ctx.discovery_wake.is_set())

        self.mode.note_discovery_scan_completed(
            living_count=0,
            added_count=0,
            area_epoch=self.tracks.area_epoch,
        )
        self.assertTrue(self.mode.discovery_confirmed_clear)
        self.assertTrue(self.mode.on_no_attackable_targets())

    def test_blocks_teleport_while_discovery_candidates_pending(self) -> None:
        from pybot.recognition.rules import DiscoveryDetection

        self.mode.note_discovery_scan_completed(
            living_count=0,
            added_count=0,
            area_epoch=self.tracks.area_epoch,
        )
        self.tracks.process_discovery_scan(
            [
                DiscoveryDetection(
                    x=100, y=100, confidence=0.9, candidate_scale=1.0,
                    living=True, bbox=(0, 0, 10, 10),
                )
            ],
            mob_name="anubis",
            now_tick=1,
        )
        self.assertTrue(self.tracks.has_pending_discovery_candidates())
        self.assertFalse(self.mode.discovery_confirmed_clear)
        self.ctx.discovery_wake.clear()
        self.assertFalse(self.mode.on_no_attackable_targets())
        # Tracking owns candidate ingestion; do not wake discovery repeatedly
        # while waiting for tracking to consume the pending candidate.
        self.assertFalse(self.ctx.discovery_wake.is_set())

        # Tracking consumes the candidate; after the next empty discovery scan
        # confirms the same area, the mode may teleport normally.
        candidates = self.tracks.get_and_clear_new_candidates()
        self.assertEqual(len(candidates), 1)
        # The tracking worker would create/resolve the candidate here; for
        # this strategy regression, consuming it is the ownership boundary.
        self.mode.note_discovery_scan_completed(
            living_count=0,
            added_count=0,
            area_epoch=self.tracks.area_epoch,
        )
        self.assertTrue(self.mode.on_no_attackable_targets())

    def test_one_post_delay_scan_decides_next_action(self) -> None:
        """After teleport settle, the first completed scan decides the next action."""
        self.mode.note_discovery_scan_completed(
            living_count=0,
            added_count=0,
            area_epoch=self.tracks.area_epoch,
        )
        self.assertTrue(self.mode.on_no_attackable_targets())
        self.assertFalse(self.mode.discovery_since_reset)

        # This is the first scan after the mode teleport's configured settle
        # delay. An empty result is sufficient to authorize the next teleport;
        # there is no artificial second-scan requirement.
        self.mode.note_discovery_scan_completed(
            living_count=0,
            added_count=0,
            area_epoch=self.tracks.area_epoch,
        )
        self.assertTrue(self.mode.discovery_since_reset)
        self.assertTrue(self.mode.discovery_confirmed_clear)
        self.assertTrue(self.mode.on_no_attackable_targets())

    def test_ignores_stale_discovery_after_area_reset(self) -> None:
        self.mode.note_discovery_scan_completed(
            living_count=0,
            added_count=0,
            area_epoch=0,
        )
        self.tracks.area_reset()
        self.mode.on_area_reset()

        self.mode.note_discovery_scan_completed(
            living_count=0,
            added_count=0,
            area_epoch=0,
        )
        self.assertFalse(self.mode.discovery_since_reset)

    def test_discovery_epoch_reads_do_not_invert_track_and_strategy_locks(self) -> None:
        """Post-teleport reset and discovery completion must not deadlock."""
        strategy = self.mode._strategy
        strategy._discovery_area_epoch = self.tracks.area_epoch
        strategy._discovery_confirmed_clear = True

        track_holds_lock = threading.Event()
        release_track_lock = threading.Event()
        original_epoch = type(self.tracks).area_epoch

        def blocked_epoch(tracks):
            track_holds_lock.set()
            release_track_lock.wait(timeout=1.0)
            return original_epoch.__get__(tracks, type(tracks))

        type(self.tracks).area_epoch = property(blocked_epoch)
        try:
            reader_done = threading.Event()

            def read_clear() -> None:
                self.mode.discovery_confirmed_clear
                reader_done.set()

            reader = threading.Thread(target=read_clear, daemon=True)
            reader.start()
            self.assertTrue(track_holds_lock.wait(timeout=1.0))

            reset_done = threading.Event()

            def reset() -> None:
                self.tracks.area_reset()
                self.mode.on_area_reset()
                reset_done.set()

            resetter = threading.Thread(target=reset, daemon=True)
            resetter.start()
            self.assertTrue(reset_done.wait(timeout=1.0))

            release_track_lock.set()
            reader.join(timeout=1.0)
            resetter.join(timeout=1.0)
            self.assertFalse(reader.is_alive())
            self.assertFalse(resetter.is_alive())
            self.assertTrue(reader_done.is_set())
        finally:
            type(self.tracks).area_epoch = original_epoch

    def test_attacks_when_alive_tracks_exist_does_not_teleport(self) -> None:
        now = monotonic_ms()
        track = self.tracks.create_track("horn", 100, 200, 0.7, 0.9, now_tick=now)
        self.tracks.apply_attack_event(track.id)
        self.mode.note_discovery_scan_completed(
            living_count=1,
            added_count=1,
            area_epoch=self.tracks.area_epoch,
        )
        teleported = self.mode.on_no_attackable_targets()
        # Track is still alive after attack (no pending state), so no teleport
        self.assertFalse(teleported)

    def _rebuild_with_sit(self, vitals: PlayerVitals) -> None:
        self.config = make_config(
            sit_on_low_sp=True,
            sit_on_low_sp_button="insert",
            sit_on_low_sp_scan_code=82,
        )
        self.ctx = HuntRuntimeContext(
            config=self.config,
            logger=self.logger,
            tracks=self.tracks,
            policy=HuntPolicy(),
            capture=MagicMock(spec=HuntWindowCapture),
            detector=self.detector,
            tracker=self.detector,
            validation=HuntValidationLogger(self.logger, self.tracks, enabled=False),
            control=RuntimeControl(None),
        )
        self.mode = create_hunt_mode(
            self.ctx,
            ShadowInputBackend(),
            teleport_controller=_make_tport(
                self.ctx, ShadowInputBackend(), vitals=vitals
            ),
        )

    def test_does_not_teleport_when_sit_enabled_and_sp_is_low(self) -> None:
        vitals = PlayerVitals()
        vitals.publish_sp(4, 100)
        self._rebuild_with_sit(vitals)
        self.mode.note_discovery_scan_completed(
            living_count=0,
            added_count=0,
            area_epoch=self.tracks.area_epoch,
        )
        self.assertFalse(self.mode.on_no_attackable_targets())
        self.assertEqual(self.tracks.area_epoch, 0)

    def test_does_not_chain_teleports_while_sp_unread_after_landing(self) -> None:
        """A hunt teleport clears SP; the next area-clear must wait for a sample."""
        vitals = PlayerVitals()
        vitals.publish_sp(50, 100)
        self._rebuild_with_sit(vitals)
        self.mode.note_discovery_scan_completed(
            living_count=0,
            added_count=0,
            area_epoch=self.tracks.area_epoch,
        )
        self.assertTrue(self.mode.on_no_attackable_targets())
        self.assertIsNone(vitals.sp)
        self.mode.note_discovery_scan_completed(
            living_count=0,
            added_count=0,
            area_epoch=self.tracks.area_epoch,
        )
        self.assertFalse(self.mode.on_no_attackable_targets())
        epoch = vitals.observation_epoch
        self.assertTrue(vitals.publish_sp_if_current(4, 100, epoch))
        self.assertFalse(self.mode.on_no_attackable_targets())
        self.assertTrue(vitals.publish_sp_if_current(50, 100, epoch))
        self.assertTrue(self.mode.on_no_attackable_targets())

    def test_hybrid_placeholder_does_not_teleport(self) -> None:
        self.config = make_config(hunt_mode="hybrid")
        self.ctx = HuntRuntimeContext(
            config=self.config,
            logger=self.logger,
            tracks=self.tracks,
            policy=HuntPolicy(),
            capture=MagicMock(spec=HuntWindowCapture),
            detector=self.detector,
            tracker=self.detector,
            validation=HuntValidationLogger(self.logger, self.tracks, enabled=False),
            control=RuntimeControl(None),
        )
        self.mode = create_hunt_mode(
            self.ctx, ShadowInputBackend(),
            teleport_controller=_make_tport(self.ctx, ShadowInputBackend()),
        )
        self.mode.note_discovery_scan_completed(
            living_count=0,
            added_count=0,
            area_epoch=self.tracks.area_epoch,
        )
        self.assertFalse(self.mode.on_no_attackable_targets())


if __name__ == "__main__":
    unittest.main()
