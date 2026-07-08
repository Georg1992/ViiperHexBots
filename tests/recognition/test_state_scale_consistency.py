"""State must use the same calibrated scales as discovery — not a separate hardcoded set."""

from __future__ import annotations

import unittest
from pathlib import Path

import cv2

from pybot.paths import PROJECT_ROOT, RECOGNITION_DIR
from pybot.recognition.cli import apply_scale_calibration
from pybot.recognition.simple.detector import SimpleMobDetector, load_simple_config
from pybot.recognition.simple.tracking.state_recognizer import evaluate_track_states

ROOT = PROJECT_ROOT
MOB_REC = RECOGNITION_DIR


def playfield_roi(frame):
    height, width = frame.shape[:2]
    return frame[
        int(height * 0.08) : int(height * 0.92),
        int(width * 0.03) : int(width * 0.97),
    ]


class StateScaleConsistencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.base_config = load_simple_config()
        cls.fixture_dir = MOB_REC / "test-fixtures" / "game-screenshots"
        cls.frame = cv2.imread(str(cls.fixture_dir / "333.png"), cv2.IMREAD_COLOR)
        if cls.frame is None:
            raise unittest.SkipTest("fixture 333.png missing")
        cls.roi = playfield_roi(cls.frame)

    def test_state_and_discovery_share_calibrated_scales(self) -> None:
        calibrated = apply_scale_calibration(self.base_config, (0.82, 0.98), True)
        detector = SimpleMobDetector(ROOT, calibrated)
        detector.apply_runtime_config(calibrated)

        discovery_scales = [round(scale, 3) for scale in detector._candidate_scales(self.roi.shape[1])]
        self.assertEqual(discovery_scales, [0.82, 0.9, 0.98])

        hinted_scales = [round(scale, 3) for scale in detector._scales_for_track(self.roi.shape[1], 0.9)]
        self.assertEqual(hinted_scales, [0.9, 0.98, 0.82])

        legacy_track_scales = [0.45, 0.55, 0.65]
        self.assertFalse(any(scale in legacy_track_scales for scale in discovery_scales))

    def test_state_stays_alive_across_repeated_ticks_at_discovery_scale(self) -> None:
        calibrated = apply_scale_calibration(self.base_config, (0.82, 0.98), True)
        detector = SimpleMobDetector(ROOT, calibrated)
        detector.apply_runtime_config(calibrated)

        discovery = detector.detect(self.roi, "horn")
        living = [c for c in discovery.accepted]
        self.assertGreater(len(living), 0, "fixture must contain at least one living horn")

        anchor = living[0]
        track = {
            "trackId": 1,
            "x": anchor.center_x,
            "y": anchor.center_y,
            "scale": anchor.candidate_scale,
        }

        for tick in range(5):
            updates = evaluate_track_states(detector, self.roi, "horn", [track])
            self.assertEqual(len(updates), 1, f"tick={tick}")
            update = updates[0]
            self.assertEqual(
                update["state"],
                "alive",
                f"tick={tick} expected alive at scale={track.get('scale')}",
            )
            track["x"] = update["x"]
            track["y"] = update["y"]


if __name__ == "__main__":
    unittest.main()
