"""Local tracker vision core tests."""

from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent.parent
MOB_REC = Path(__file__).resolve().parent.parent
SIMPLE = MOB_REC / "simple"
for path in (str(MOB_REC), str(SIMPLE)):
    if path not in sys.path:
        sys.path.insert(0, path)

from cli import apply_scale_calibration  # noqa: E402
from detector import SimpleMobDetector, load_simple_config  # noqa: E402
from tracking.local_track_recognizer import follow_track_local  # noqa: E402
from tracking.local_tracker import LocalTrackResult, track_local  # noqa: E402


def playfield_roi(frame):
    height, width = frame.shape[:2]
    return frame[
        int(height * 0.08) : int(height * 0.92),
        int(width * 0.03) : int(width * 0.97),
    ]


class LocalTrackerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.base_config = load_simple_config()
        cls.fixture_dir = MOB_REC / "test-fixtures" / "game-screenshots"
        cls.frame = cv2.imread(str(cls.fixture_dir / "333.png"), cv2.IMREAD_COLOR)
        if cls.frame is None:
            raise unittest.SkipTest("fixture 333.png missing")
        cls.roi = playfield_roi(cls.frame)

    def _detector(self) -> SimpleMobDetector:
        calibrated = apply_scale_calibration(self.base_config, (0.82, 0.98), True)
        detector = SimpleMobDetector(ROOT, calibrated)
        detector.apply_runtime_config(calibrated)
        return detector

    def _living_anchor(self, detector: SimpleMobDetector):
        discovery = detector.detect(self.roi, "horn")
        living = [c for c in discovery.accepted]
        self.assertGreater(len(living), 0)
        return living[0]

    def test_finds_mob_at_discovery_coords(self) -> None:
        detector = self._detector()
        anchor = self._living_anchor(detector)
        track = {
            "trackId": 1,
            "x": anchor.center_x,
            "y": anchor.center_y,
            "scale": anchor.candidate_scale,
        }
        result = track_local(detector, self.roi, "horn", track)
        self.assertIsInstance(result, LocalTrackResult)
        self.assertTrue(result.found)
        self.assertGreater(result.confidence, 0.4)
        self.assertEqual(result.miss_reason, "")
        dist = abs(result.x - anchor.center_x) + abs(result.y - anchor.center_y)
        self.assertLess(dist, 40)

    def test_recognizer_wrapper_matches_detector(self) -> None:
        detector = self._detector()
        anchor = self._living_anchor(detector)
        track = {
            "trackId": 2,
            "x": anchor.center_x,
            "y": anchor.center_y,
            "scale": anchor.candidate_scale,
        }
        direct = track_local(detector, self.roi, "horn", track)
        wrapped = follow_track_local(detector, self.roi, "horn", track)
        self.assertEqual(direct, wrapped)

    def test_miss_returns_reason_not_unreachable(self) -> None:
        detector = self._detector()
        track = {"trackId": 99, "x": 8, "y": 8, "scale": 0.9}
        result = track_local(detector, self.roi, "horn", track, search_radius_px=20)
        self.assertFalse(result.found)
        self.assertGreater(len(result.miss_reason), 0)
        self.assertNotIn(result.miss_reason, ("unreachable", "unknown"))

    def test_finds_mob_within_search_radius_after_offset_seed(self) -> None:
        detector = self._detector()
        anchor = self._living_anchor(detector)
        track = {
            "trackId": 3,
            "x": anchor.center_x + 12,
            "y": anchor.center_y + 8,
            "scale": anchor.candidate_scale,
        }
        result = track_local(detector, self.roi, "horn", track, search_radius_px=60)
        self.assertTrue(result.found)
        dist = abs(result.x - anchor.center_x) + abs(result.y - anchor.center_y)
        self.assertLess(dist, 50)

    def test_benchmark_one_three_six_tracks(self) -> None:
        detector = self._detector()
        discovery = detector.detect(self.roi, "horn")
        living = [c for c in discovery.accepted][:6]
        if len(living) < 3:
            self.skipTest("fixture needs at least 3 living horns")

        def bench(tracks: list[dict]) -> float:
            start = time.perf_counter()
            for track in tracks:
                track_local(detector, self.roi, "horn", track)
            return time.perf_counter() - start

        one = [
            {
                "trackId": 1,
                "x": living[0].center_x,
                "y": living[0].center_y,
                "scale": living[0].candidate_scale,
            }
        ]
        three = [
            {
                "trackId": index + 1,
                "x": candidate.center_x,
                "y": candidate.center_y,
                "scale": candidate.candidate_scale,
            }
            for index, candidate in enumerate(living[:3])
        ]
        six = [
            {
                "trackId": index + 1,
                "x": candidate.center_x,
                "y": candidate.center_y,
                "scale": candidate.candidate_scale,
            }
            for index, candidate in enumerate(living[:6])
        ]

        elapsed_one = bench(one)
        elapsed_three = bench(three)
        elapsed_six = bench(six)

        print(
            f"\nlocal_track bench: 1={elapsed_one:.3f}s "
            f"3={elapsed_three:.3f}s 6={elapsed_six:.3f}s"
        )
        self.assertLess(elapsed_one, 0.5)
        self.assertLess(elapsed_three, 1.5)
        self.assertLess(elapsed_six, 3.0)


if __name__ == "__main__":
    unittest.main()
