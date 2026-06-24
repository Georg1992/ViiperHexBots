from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
MOB_REC = Path(__file__).resolve().parent
SIMPLE = MOB_REC / "simple"
for path in (str(MOB_REC), str(SIMPLE)):
    if path not in sys.path:
        sys.path.insert(0, path)

from cli import parse_request_tracks, run_detect_request  # noqa: E402
from detector import SimpleMobDetector, load_simple_config  # noqa: E402
from state_recognizer import evaluate_track_states  # noqa: E402


def playfield_roi(frame):
    height, width = frame.shape[:2]
    return frame[
        int(height * 0.08) : int(height * 0.92),
        int(width * 0.03) : int(width * 0.97),
    ]


class TrackIdStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_simple_config()
        cls.detector = SimpleMobDetector(ROOT, cls.config)
        cls.fixture_dir = MOB_REC / "test-fixtures" / "user-corpse"
        cls.manifest = json.loads((cls.fixture_dir / "manifest.json").read_text(encoding="utf-8"))

    def test_parse_request_tracks(self) -> None:
        tracks = parse_request_tracks(
            [
                {"trackId": 12, "x": 100, "y": 200},
                {"trackId": 5, "x": 50, "y": 60},
            ]
        )
        self.assertEqual(len(tracks), 2)
        self.assertEqual(tracks[0]["trackId"], 12)
        self.assertEqual(tracks[1]["y"], 60)

    def test_state_response_echoes_track_ids(self) -> None:
        entry = self.manifest["images"][0]
        image_path = self.fixture_dir / entry["file"]
        frame = cv2.imread(str(image_path))
        self.assertIsNotNone(frame)
        roi = playfield_roi(frame)
        watch_points = [tuple(point) for point in entry["watchPoints"]]
        tracks = [{"trackId": index + 1, "x": point[0], "y": point[1]} for index, point in enumerate(watch_points)]
        updates = evaluate_track_states(self.detector, roi, "horn", tracks)
        self.assertEqual(len(updates), len(tracks))
        for request, update in zip(tracks, updates):
            self.assertEqual(update["trackId"], request["trackId"])
            self.assertIn(update["state"], {"alive", "dead", "gone"})

    def test_close_mobs_keep_distinct_track_ids(self) -> None:
        entry = next(item for item in self.manifest["images"] if len(item["watchPoints"]) >= 2)
        image_path = self.fixture_dir / entry["file"]
        frame = cv2.imread(str(image_path))
        self.assertIsNotNone(frame)
        roi = playfield_roi(frame)
        watch_points = [tuple(point) for point in entry["watchPoints"][:2]]
        tracks = [
            {"trackId": 101, "x": watch_points[0][0], "y": watch_points[0][1]},
            {"trackId": 202, "x": watch_points[1][0], "y": watch_points[1][1]},
        ]
        updates = evaluate_track_states(self.detector, roi, "horn", tracks)
        ids = {update["trackId"] for update in updates}
        self.assertEqual(ids, {101, 202})


if __name__ == "__main__":
    unittest.main()
