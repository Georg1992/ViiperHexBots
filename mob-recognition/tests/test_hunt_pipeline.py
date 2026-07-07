"""Hunt pipeline contract tests."""

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
from detector import SimpleMobDetector, load_simple_config  # noqa: E402
from hunt_track_rules import (  # noqa: E402
    MobTrack,
    select_target_id,
)
from tracking.state_recognizer import evaluate_track_states  # noqa: E402


def playfield_roi(frame):
    height, width = frame.shape[:2]
    return frame[
        int(height * 0.08) : int(height * 0.92),
        int(width * 0.03) : int(width * 0.97),
    ]


class HuntPipelineIntegrationTests(unittest.TestCase):
    """Discovery + state vision + track rules — reproduces live-session failure modes."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.base_config = load_simple_config()
        cls.fixture_dir = MOB_REC / "test-fixtures" / "game-screenshots"
        cls.frame = cv2.imread(str(cls.fixture_dir / "333.png"), cv2.IMREAD_COLOR)
        if cls.frame is None:
            raise unittest.SkipTest("fixture 333.png missing")
        cls.roi = playfield_roi(cls.frame)

    def _detector_at_discovery_scale(self) -> SimpleMobDetector:
        calibrated = apply_scale_calibration(self.base_config, (0.82, 0.98), True)
        detector = SimpleMobDetector(ROOT, calibrated)
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

    def test_live_session_timeline_unreachable_ignored_still_attacks(self) -> None:
        """Replays log pattern: create @19:54:30, unreachable ignored @19:54:37 — must stay attackable."""
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

        state_req = {
            "trackId": 1,
            "x": track.x,
            "y": track.y,
            "scale": track.discovery_scale,
        }
        updates = evaluate_track_states(detector, self.roi, "horn", [state_req])
        self.assertEqual(len(updates), 1)

        self.assertEqual(track.state, "alive", "must stay alive after state eval")
        self.assertEqual(select_target_id([track], t_create), 1)

    def test_state_alive_then_multi_tick_attackable(self) -> None:
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
        state_track = {
            "trackId": 1,
            "x": track.x,
            "y": track.y,
            "scale": anchor.candidate_scale,
        }

        for tick in range(5):
            at = now + tick * 2_000
            updates = evaluate_track_states(detector, self.roi, "horn", [state_track])
            obs = updates[0]
            if obs["state"] == "alive":
                state_track["x"] = obs["x"]
                state_track["y"] = obs["y"]

            self.assertEqual(
                track.state, "alive",
                f"tick={tick} state={obs['state']} must stay attackable before first attack",
            )

        self.assertEqual(select_target_id([track], now + 20_000), 1)


if __name__ == "__main__":
    unittest.main()
