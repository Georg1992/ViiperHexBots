"""Thara Frog screenshot fixture regression tests."""

from __future__ import annotations

import json
import re
import unittest

import cv2

from pybot.paths import PROJECT_ROOT, RECOGNITION_DIR
from pybot.recognition.detector.detector import MobDetector, load_detector_config

ROOT = PROJECT_ROOT
FIXTURES_ROOT = RECOGNITION_DIR / "test-fixtures"
THARA_MANIFEST = FIXTURES_ROOT / "thara_frog" / "manifest.json"
THARA_FIXTURE_PATTERN = re.compile(r"^(\d+)Tharas?\.png$", re.IGNORECASE)


def expected_count_from_filename(file_name: str) -> int:
    match = THARA_FIXTURE_PATTERN.match(file_name)
    if match is None:
        raise ValueError(f"not a thara fixture filename: {file_name}")
    return int(match.group(1))


class TharaFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not THARA_MANIFEST.is_file():
            raise unittest.SkipTest("thara_frog/manifest.json missing")
        cls.manifest = json.loads(THARA_MANIFEST.read_text(encoding="utf-8"))
        cls.image_dir = (THARA_MANIFEST.parent / cls.manifest["imageDir"]).resolve()
        cls.detector = MobDetector(ROOT, load_detector_config())
        cls.entries = list(cls.manifest["images"])

    def test_manifest_lists_four_thara_fixtures(self) -> None:
        self.assertEqual(len(self.entries), 4)
        for entry in self.entries:
            expected_count_from_filename(entry["file"])

    def test_fixture_accept_counts(self) -> None:
        strict = bool(self.manifest.get("strictAcceptCount", True))
        for entry in self.entries:
            file_name = entry["file"]
            image_path = self.image_dir / file_name
            frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            self.assertIsNotNone(frame, f"missing or unreadable fixture: {image_path}")

            result = self.detector.detect(frame, "thara_frog")
            expected = expected_count_from_filename(file_name)
            self.assertEqual(
                int(entry["expectCount"]),
                expected,
                f"{file_name}: manifest expectCount must match filename",
            )
            accepted = len(result.accepted)

            if strict:
                self.assertEqual(
                    accepted,
                    expected,
                    f"{file_name}: expected {expected} accepted, got {accepted}",
                )
            else:
                self.assertGreaterEqual(
                    accepted,
                    expected,
                    f"{file_name}: expected at least {expected} accepted, got {accepted}",
                )


if __name__ == "__main__":
    unittest.main()
