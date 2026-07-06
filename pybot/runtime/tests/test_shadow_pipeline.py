"""End-to-end shadow pipeline on fixture frame."""

from __future__ import annotations

import threading
import unittest
from pathlib import Path

import cv2
import numpy as np

from pybot.runtime.capture.window_roi import HuntRoi
from pybot.runtime.config import PROJECT_ROOT, HuntRuntimeConfig
from pybot.runtime.control import RuntimeControl
from pybot.runtime.hunt_mode import create_hunt_mode
from pybot.runtime.hunt_policy import HuntPolicy
from pybot.runtime.hunt_tracks import HuntTracks
from pybot.runtime.input.input_backend import ShadowInputBackend
from pybot.runtime.logging import HuntLogger
from pybot.runtime.runtime_context import HuntRuntimeContext
from pybot.runtime.urgent_state import UrgentStateQueue
from pybot.runtime.validation_log import HuntValidationLogger
from pybot.runtime.detection.detector_session import DetectorSession
from pybot.runtime.workers.discovery_worker import DiscoveryWorker
from pybot.runtime.workers.tracking_worker import TrackingWorker

FIXTURE = PROJECT_ROOT / "mob-recognition" / "test-fixtures" / "game-screenshots" / "333.png"


def playfield_roi(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    return frame[
        int(height * 0.08) : int(height * 0.92),
        int(width * 0.03) : int(width * 0.97),
    ]


class FixtureDetector(DetectorSession):
    def __init__(self, frame: np.ndarray) -> None:
        super().__init__("horn", project_root=PROJECT_ROOT)
        self._fixture_frame = frame

    def discover(self, roi: HuntRoi):
        return self.discover_frame(self._fixture_frame, roi)

    def track_locals(self, roi: HuntRoi, track_snapshots):
        return self.track_locals_frame(self._fixture_frame, roi, track_snapshots)


class FakeCapture:
    def __init__(self, roi: HuntRoi) -> None:
        self._roi = roi

    def is_valid(self) -> bool:
        return True

    def get_hunt_roi(self) -> HuntRoi:
        return self._roi


def make_config() -> HuntRuntimeConfig:
    return HuntRuntimeConfig(
        config_path=PROJECT_ROOT / "config.ini",
        hwnd=12345,
        mob_name="horn",
        hunt_mode="teleport",
        skill_delay_ms=500,
        skill_button="e",
        skill_scan_code=18,
        teleport_button="q",
        teleport_scan_code=16,
        search_range_cells=16,
        cell_size_px=64,
        state_interval_ms=100,
        discovery_interval_ms=3000,
        post_attack_state_delay_ms=120,
        teleport_duration_ms=500,
        coord_stale_skip_ms=None,
        validation_enabled=False,
        validation_state_every_n=1,
        control_file=None,
    )


class ShadowPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        frame = cv2.imread(str(FIXTURE), cv2.IMREAD_COLOR)
        if frame is None:
            raise unittest.SkipTest("fixture 333.png missing")
        cls.roi_frame = playfield_roi(frame)
        cls.roi = HuntRoi(x=0, y=0, w=cls.roi_frame.shape[1], h=cls.roi_frame.shape[0])

    def test_discovery_and_local_track_shadow_tick(self) -> None:
        config = make_config()
        logger = HuntLogger(session_id="test_shadow_pipeline")
        tracks = HuntTracks()
        stop = threading.Event()
        stop.set()
        capture = FakeCapture(self.roi)
        detector = FixtureDetector(self.roi_frame)
        ctx = HuntRuntimeContext(
            config=config,
            logger=logger,
            tracks=tracks,
            policy=HuntPolicy(),
            capture=capture,
            detector=detector,
            urgent=UrgentStateQueue(),
            validation=HuntValidationLogger(logger, tracks, enabled=False),
            control=RuntimeControl(None),
            stop_event=stop,
        )
        hunt_mode = create_hunt_mode(ctx, ShadowInputBackend())

        discovery = DiscoveryWorker(ctx, hunt_mode)
        discovery._run_scan()

        self.assertGreater(tracks.get_track_count(), 0)
        self.assertTrue(hunt_mode.discovery_since_reset)

        tracking = TrackingWorker(ctx)
        tracking._run_local_track_batch()

        self.assertGreaterEqual(tracks.get_track_count(), 1)
        track = tracks.get_track_by_id(1)
        assert track is not None
        self.assertGreater(track.updated_tick, 0)
