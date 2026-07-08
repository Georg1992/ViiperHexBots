"""HuntMode teleport / area-clear tests."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from pybot.runtime.capture.hunt_capture import HuntWindowCapture
from pybot.paths import PROJECT_ROOT
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
        self.mode.note_discovery_scan_completed(living_count=0, added_count=0)
        teleported = self.mode.on_no_attackable_targets()
        self.assertTrue(teleported)
        self.assertEqual(self.tracks.get_track_count(), 0)
        self.assertEqual(self.tracks.area_epoch, 1)

    def test_attacks_when_alive_tracks_exist_does_not_teleport(self) -> None:
        now = monotonic_ms()
        track = self.tracks.create_track("horn", 100, 200, 0.7, 0.9, now_tick=now)
        self.tracks.apply_attack_event(track.id, now_tick=now + 10)
        self.mode.note_discovery_scan_completed(living_count=1, added_count=1)
        teleported = self.mode.on_no_attackable_targets()
        # Track is still alive after attack (no pending state), so no teleport
        self.assertFalse(teleported)


if __name__ == "__main__":
    unittest.main()
