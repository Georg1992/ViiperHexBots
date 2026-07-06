"""Canonical state evaluator and search profiles."""

from __future__ import annotations

import sys
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
from detector import (  # noqa: E402
    STATE_PROFILE_DIRECT,
    SimpleMobDetector,
    load_simple_config,
)
from tracking.state_recognizer import evaluate_track_state, evaluate_track_states  # noqa: E402


def playfield_roi(frame):
    height, width = frame.shape[:2]
    return frame[
        int(height * 0.08) : int(height * 0.92),
        int(width * 0.03) : int(width * 0.97),
    ]


class StateCanonicalTests(unittest.TestCase):
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

    def test_full_profile_finds_living_mob_at_discovery_coords(self) -> None:
        detector = self._detector()
        discovery = detector.detect(self.roi, "horn")
        living = [c for c in discovery.accepted if not c.is_dead]
        self.assertGreater(len(living), 0)
        anchor = living[0]
        tracks = [
            {
                "trackId": 1,
                "x": anchor.center_x,
                "y": anchor.center_y,
                "scale": anchor.candidate_scale,
            }
        ]
        updates = evaluate_track_states(detector, self.roi, "horn", tracks)
        self.assertEqual(len(updates), 1)
        self.assertEqual(updates[0]["state"], "alive")
        self.assertGreater(updates[0]["confidence"], 0.4)

    def test_direct_profile_returns_alive_dead_or_gone_not_unknown(self) -> None:
        detector = self._detector()
        discovery = detector.detect(self.roi, "horn")
        living = [c for c in discovery.accepted if not c.is_dead]
        self.assertGreater(len(living), 0)
        anchor = living[0]
        update = evaluate_track_state(
            detector,
            self.roi,
            "horn",
            1,
            anchor.center_x,
            anchor.center_y,
            scale_hint=anchor.candidate_scale,
            profile=STATE_PROFILE_DIRECT,
        )
        self.assertIn(update["state"], ("alive", "dead", "gone"))
        self.assertNotEqual(update["state"], "unknown")


if __name__ == "__main__":
    unittest.main()
