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

from detector import SimpleMobDetector, load_simple_config  # noqa: E402


def playfield_roi(frame):
    height, width = frame.shape[:2]
    return frame[
        int(height * 0.08) : int(height * 0.92),
        int(width * 0.03) : int(width * 0.97),
    ]


class CorpseDetectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_simple_config()
        cls.detector = SimpleMobDetector(ROOT, cls.config)
        cls.fixture_dir = MOB_REC / "test-fixtures" / "user-corpse"
        cls.manifest = json.loads((cls.fixture_dir / "manifest.json").read_text(encoding="utf-8"))

    def test_user_corpse_screenshots(self) -> None:
        for entry in self.manifest["images"]:
            with self.subTest(image=entry["file"]):
                image_path = self.fixture_dir / entry["file"]
                frame = cv2.imread(str(image_path))
                self.assertIsNotNone(frame, entry["file"])
                attack_slots = [tuple(slot) for slot in entry["attackSlots"]]
                result = self.detector.detect(playfield_roi(frame), "horn", attack_slots=attack_slots)
                dead = sum(1 for candidate in result.accepted if candidate.is_dead)
                living = sum(1 for candidate in result.accepted if not candidate.is_dead)
                self.assertEqual(dead, entry["expectDead"], result.accepted)
                self.assertEqual(living, entry["expectLiving"], result.accepted)
                self.assertTrue(all(candidate.is_dead for candidate in result.accepted))


if __name__ == "__main__":
    unittest.main()
