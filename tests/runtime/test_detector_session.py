"""DetectorSession tests using fixture frames (no live capture)."""

from __future__ import annotations

import unittest

import cv2

from pybot.paths import PROJECT_ROOT, RECOGNITION_FIXTURES_DIR
from pybot.runtime.capture.window_roi import HuntRoi
from pybot.runtime.detection.detector_session import DetectorSession, StateTrackSnapshot

ROOT = PROJECT_ROOT
FIXTURE = RECOGNITION_FIXTURES_DIR / "game-screenshots" / "333.png"


def playfield_roi(frame):
    height, width = frame.shape[:2]
    return frame[
        int(height * 0.08) : int(height * 0.92),
        int(width * 0.03) : int(width * 0.97),
    ]


class DetectorSessionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.frame = cv2.imread(str(FIXTURE), cv2.IMREAD_COLOR)
        if cls.frame is None:
            raise unittest.SkipTest("fixture 333.png missing")
        cls.roi_frame = playfield_roi(cls.frame)
        cls.roi = HuntRoi(x=0, y=0, w=cls.roi_frame.shape[1], h=cls.roi_frame.shape[0])
        cls.detector = DetectorSession("horn", project_root=ROOT)

    def test_discover_finds_living_candidates(self) -> None:
        scan = self.detector.discover_frame(self.roi_frame, self.roi)
        self.assertTrue(scan.ok)
        self.assertGreater(scan.raw_count, 0)
        living = [d for d in scan.detections if d.living]
        self.assertGreater(len(living), 0)
        self.assertGreater(scan.duration_ms, 0)

    def test_track_locals_returns_results(self) -> None:
        scan = self.detector.discover_frame(self.roi_frame, self.roi)
        anchor = next(d for d in scan.detections if d.living)
        snapshots = [
            StateTrackSnapshot(
                track_id=1,
                x=anchor.x,
                y=anchor.y,
                scale=anchor.candidate_scale,
            )
        ]
        batch = self.detector.track_locals_frame(self.roi_frame, self.roi, snapshots)
        self.assertTrue(batch.ok)
        self.assertEqual(len(batch.results), 1)
        self.assertTrue(batch.results[0].found)
        self.assertGreater(batch.duration_ms, 0)


if __name__ == "__main__":
    unittest.main()
