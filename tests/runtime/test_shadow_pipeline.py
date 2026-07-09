"""End-to-end shadow pipeline on fixture frame."""

from __future__ import annotations

import threading
import unittest
from pathlib import Path

import cv2
import numpy as np

from pybot.runtime.capture.window_roi import HuntRoi
from pybot.paths import PROJECT_ROOT, RECOGNITION_FIXTURES_DIR
from pybot.recognition.fixtures import default_horn_fixture
from pybot.runtime.config import HuntRuntimeConfig
from pybot.runtime.control import RuntimeControl
from pybot.runtime.hunt_mode import create_hunt_mode
from pybot.runtime.hunt_policy import HuntPolicy
from pybot.runtime.hunt_tracks import HuntTracks
from pybot.runtime.input.input_backend import ShadowInputBackend
from pybot.runtime.logging import HuntLogger
from pybot.runtime.runtime_context import HuntRuntimeContext
from pybot.runtime.validation_log import HuntValidationLogger
from pybot.runtime.detection.detector_session import DetectorSession

FIXTURE = default_horn_fixture()


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
        discovery_interval_ms=3000,
        teleport_duration_ms=500,
        validation_enabled=False,
        control_file=None,
    )


class ShadowPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        frame = cv2.imread(str(FIXTURE), cv2.IMREAD_COLOR)
        if frame is None:
            raise unittest.SkipTest("fixture Horn/3Horn.png missing")
        cls.roi_frame = playfield_roi(frame)
        cls.roi = HuntRoi(x=0, y=0, w=cls.roi_frame.shape[1], h=cls.roi_frame.shape[0])

    def test_discovery_creates_tracks_on_fixture(self) -> None:
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
            tracker=detector,
            validation=HuntValidationLogger(logger, tracks, enabled=False),
            control=RuntimeControl(None),
            stop_event=stop,
        )
        hunt_mode = create_hunt_mode(ctx, ShadowInputBackend())

        from pybot.recognition.rules import DiscoveryDetection

        scan = detector.discover(self.roi)
        detections = [
            DiscoveryDetection(
                x=d.x, y=d.y, confidence=d.confidence, candidate_scale=d.candidate_scale, living=True
            )
            for d in scan.detections
        ]

        summary = tracks.reconcile_detections(
            detections,
            mob_name="horn",
            now_tick=int(1_000_000),
        )

        self.assertGreater(tracks.get_track_count(), 0)
        self.assertGreater(summary.added_count, 0)

        track = tracks.get_track_by_id(1)
        assert track is not None
        self.assertGreater(track.updated_tick, 0)
        self.assertEqual(track.state, "alive")


if __name__ == "__main__":
    unittest.main()
