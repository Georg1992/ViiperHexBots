"""Hunt pipeline contract tests."""

from __future__ import annotations

import unittest

import cv2
import numpy as np

from pybot.paths import PROJECT_ROOT, RECOGNITION_DIR
from pybot.recognition.cli import apply_scale_calibration
from pybot.recognition.fixtures import default_horn_fixture
from pybot.recognition.rules import MobTrack, select_target_id
from pybot.recognition.detector.detector import MobDetector, load_detector_config
from pybot.recognition.detector.tracking.local_tracker import track_local

ROOT = PROJECT_ROOT
MOB_REC = RECOGNITION_DIR


def playfield_roi(frame):
    height, width = frame.shape[:2]
    return frame[
        int(height * 0.08) : int(height * 0.92),
        int(width * 0.03) : int(width * 0.97),
    ]


def _sample_fixture_palette(
    frame: np.ndarray, cx: int, cy: int, size: int = 20,
) -> list[tuple[int, int, int]]:
    h, w = frame.shape[:2]
    half = size // 2
    x0 = max(0, cx - half)
    y0 = max(0, cy - half)
    x1 = min(w, x0 + size)
    y1 = min(h, y0 + size)
    if x1 <= x0 or y1 <= y0:
        return []
    patch = frame[y0:y1, x0:x1]
    pixels = patch.reshape(-1, 3)
    unique = np.unique(pixels, axis=0)
    return [tuple(int(v) for v in color) for color in unique]


class HuntPipelineIntegrationTests(unittest.TestCase):
    """Discovery + local tracking + track rules."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.base_config = load_detector_config()
        cls.fixture_dir = MOB_REC / "test-fixtures" / "game-screenshots"
        cls.frame = cv2.imread(str(default_horn_fixture()), cv2.IMREAD_COLOR)
        if cls.frame is None:
            raise unittest.SkipTest("fixture Horn/3Horn.png missing")
        cls.roi = playfield_roi(cls.frame)

    def _detector_at_discovery_scale(self) -> MobDetector:
        calibrated = apply_scale_calibration(self.base_config, (0.82, 0.98), True)
        detector = MobDetector(ROOT, calibrated)
        detector.apply_runtime_config(calibrated)
        return detector

    def test_discovery_to_attackable_without_waiting_for_state(self) -> None:
        detector = self._detector_at_discovery_scale()
        discovery = detector.detect(self.roi, "horn")
        living = [c for c in discovery.accepted]
        self.assertGreater(len(living), 0)

        anchor = living[0]
        now = 3_182_750_000
        track = MobTrack.from_discovery(
            1,
            anchor.center_x,
            anchor.center_y,
            anchor.final_score,
            now_tick=now,
            discovery_scale=anchor.candidate_scale,
        )
        self.assertEqual(track.state, "alive")
        self.assertEqual(select_target_id([track], now), 1)

    def test_local_tracking_keeps_track_attackable(self) -> None:
        detector = self._detector_at_discovery_scale()
        discovery = detector.detect(self.roi, "horn")
        living = [c for c in discovery.accepted]
        self.assertGreater(len(living), 0)
        anchor = living[0]

        t_create = 0
        track = MobTrack.from_discovery(
            1,
            anchor.center_x,
            anchor.center_y,
            0.65,
            now_tick=t_create,
            discovery_scale=anchor.candidate_scale,
        )
        palette = _sample_fixture_palette(self.roi, track.x, track.y)
        track_req = {
            "trackId": 1,
            "x": track.x,
            "y": track.y,
            "scale": track.discovery_scale,
            "trackPaletteBgr": palette,
        }
        result = track_local(detector, self.roi, "horn", track_req)
        self.assertTrue(result.found)
        self.assertEqual(track.state, "alive")
        self.assertEqual(select_target_id([track], t_create), 1)

    def test_local_tracking_multi_tick_attackable(self) -> None:
        detector = self._detector_at_discovery_scale()
        discovery = detector.detect(self.roi, "horn")
        living = [c for c in discovery.accepted]
        anchor = living[0]

        now = 100_000
        track = MobTrack.from_discovery(
            1,
            anchor.center_x,
            anchor.center_y,
            anchor.final_score,
            now_tick=now,
            discovery_scale=anchor.candidate_scale,
        )
        palette = _sample_fixture_palette(self.roi, track.x, track.y)
        track_req = {
            "trackId": 1,
            "x": track.x,
            "y": track.y,
            "scale": anchor.candidate_scale,
            "trackPaletteBgr": palette,
        }

        for tick in range(5):
            at = now + tick * 2_000
            result = track_local(detector, self.roi, "horn", track_req)
            self.assertTrue(result.found, f"tick={tick} local follow must keep mob visible")
            track_req["x"] = result.x
            track_req["y"] = result.y
            self.assertEqual(select_target_id([track], at), 1)


if __name__ == "__main__":
    unittest.main()
