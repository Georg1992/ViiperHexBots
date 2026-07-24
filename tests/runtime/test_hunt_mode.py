"""HuntMode teleport / area-clear tests."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from pybot.runtime.capture.hunt_capture import HuntWindowCapture
from pybot.paths import PROJECT_ROOT
from pybot.recognition.rules import DiscoveryDetection
from pybot.runtime.config import HuntRuntimeConfig
from pybot.runtime.control import RuntimeControl
from pybot.runtime.hunt_mode import create_hunt_mode
from pybot.runtime.hunt_policy import HuntPolicy
from pybot.runtime.hunt_tracks import HuntTracks, monotonic_ms
from pybot.runtime.input.input_backend import ShadowInputBackend
from pybot.runtime.logging import HuntLogger
from pybot.runtime.runtime_context import HuntRuntimeContext
from pybot.runtime.validation_log import HuntValidationLogger
from pybot.runtime.detection.detector_session import DetectorSession
from tests.runtime.test_hunt_tracks import _death_result


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
        self.detector.is_busy.return_value = False
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
        self.mode = create_hunt_mode(self.ctx, ShadowInputBackend())

    def test_blocks_teleport_without_discovery(self) -> None:
        teleported = self.mode.on_no_attackable_targets()
        self.assertFalse(teleported)
        self.assertEqual(self.tracks.get_track_count(), 0)

    def test_shadow_teleport_on_area_clear(self) -> None:
        self.mode.note_discovery_scan_completed(
            living_count=0,
            added_count=0,
            area_epoch=self.tracks.area_epoch,
        )
        teleported = self.mode.on_no_attackable_targets()
        self.assertTrue(teleported)
        self.assertEqual(self.tracks.get_track_count(), 0)
        self.assertEqual(self.tracks.area_epoch, 1)
        # Post-settle: discovery may scan again; suspend must be clear + wake set.
        self.assertFalse(self.ctx.discovery_suspend.is_set())
        self.assertTrue(self.ctx.discovery_wake.is_set())

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

    def test_death_site_ghost_detections_block_clear(self) -> None:
        # After a kill, discovery may still heat the corpse. Ghost matching
        # keeps alive_after=0, but the scan still saw a living candidate — that
        # must block teleport clear so we do not wipe ghosts and recreate it.
        now = monotonic_ms()
        track_id = self.tracks.create_track(
            "horn", 874, 578, 0.65, 0.9, now_tick=now
        ).id
        self.tracks.apply_death_results([_death_result(track_id)], now_tick=now + 1)
        summary = self.tracks.reconcile_detections(
            [
                DiscoveryDetection(
                    x=874, y=578, confidence=0.75, candidate_scale=0.9, living=True
                )
            ],
            mob_name="horn",
            now_tick=now + 100,
        )
        self.assertEqual(summary.alive_after, 0)
        self.assertEqual(summary.matched_count, 1)
        self.mode.note_discovery_scan_completed(
            living_count=1,
            added_count=summary.added_count,
            area_epoch=self.tracks.area_epoch,
        )
        self.assertFalse(self.mode.discovery_confirmed_clear)

    def test_blocks_teleport_until_post_teleport_discovery(self) -> None:
        self.mode.note_discovery_scan_completed(
            living_count=0,
            added_count=0,
            area_epoch=self.tracks.area_epoch,
        )
        self.assertTrue(self.mode.on_no_attackable_targets())

        self.assertFalse(self.mode.discovery_since_reset)
        self.assertFalse(self.mode.discovery_confirmed_clear)
        teleported = self.mode.on_no_attackable_targets()
        self.assertFalse(teleported)

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

    def test_attacks_when_alive_tracks_exist_does_not_teleport(self) -> None:
        now = monotonic_ms()
        track = self.tracks.create_track("horn", 100, 200, 0.7, 0.9, now_tick=now)
        self.tracks.apply_attack_event(track.id, now_tick=now + 10)
        self.mode.note_discovery_scan_completed(
            living_count=1,
            added_count=1,
            area_epoch=self.tracks.area_epoch,
        )
        teleported = self.mode.on_no_attackable_targets()
        # Track is still alive after attack (no pending state), so no teleport
        self.assertFalse(teleported)

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
        self.mode = create_hunt_mode(self.ctx, ShadowInputBackend())
        self.mode.note_discovery_scan_completed(
            living_count=0,
            added_count=0,
            area_epoch=self.tracks.area_epoch,
        )
        self.assertFalse(self.mode.on_no_attackable_targets())


if __name__ == "__main__":
    unittest.main()
