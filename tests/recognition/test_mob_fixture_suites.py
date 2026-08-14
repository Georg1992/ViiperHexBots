"""Mob screenshot fixture regression tests (Horn, TharaFrog, Alligator, Noxious, Creamy, Wolf, WildRose)."""

from __future__ import annotations

import unittest

import cv2

from pybot.paths import PROJECT_ROOT
from pybot.recognition.detector.detector import MobDetector, load_detector_config
from pybot.recognition.fixtures import (
    MOB_FIXTURE_SUITES,
    MobFixtureImage,
    MobFixtureSuite,
    fixture_search_frame,
)


class MobFixtureSuiteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        config = load_detector_config()
        cls.detector = MobDetector(PROJECT_ROOT, config)
        cls.grf_detector = MobDetector(PROJECT_ROOT, config, use_sprite_grf=True)

    def test_each_suite_has_expected_fixtures(self) -> None:
        for suite in MOB_FIXTURE_SUITES:
            images = suite.images()
            self.assertEqual(
                len(images),
                suite.expected_fixture_count,
                f"{suite.folder}: expected {suite.expected_fixture_count} PNG fixtures, found {len(images)}",
            )
            normal = [image for image in images if not image.gray_world]
            gray = [image for image in images if image.gray_world]
            self.assertEqual(
                len(normal),
                suite.expected_normal_count,
                f"{suite.folder}: expected {suite.expected_normal_count} normal-world fixtures",
            )
            self.assertEqual(
                len(gray),
                suite.expected_gray_count,
                f"{suite.folder}: expected {suite.expected_gray_count} gray-world fixtures",
            )

    def test_fixture_accept_counts(self) -> None:
        for suite in MOB_FIXTURE_SUITES:
            if not suite.recognition_regression:
                continue
            with self.subTest(suite=suite.folder):
                self._assert_suite_counts(suite)

    def _assert_suite_counts(self, suite: MobFixtureSuite) -> None:
        for image in suite.images():
            with self.subTest(suite=suite.folder, file=image.file_name):
                self._assert_image_count(suite, image)

    def _assert_image_count(self, suite: MobFixtureSuite, image: MobFixtureImage) -> None:
        frame = cv2.imread(str(image.path), cv2.IMREAD_COLOR)
        self.assertIsNotNone(frame, f"missing or unreadable fixture: {image.path}")

        detector = self.grf_detector if suite.use_sprite_grf else self.detector
        result = detector.detect(fixture_search_frame(frame), suite.mob_name)

        self.assertEqual(
            len(result.accepted),
            image.expected_count,
            f"{suite.folder}/{image.file_name}: expected {image.expected_count} "
            f"accepted, got {len(result.accepted)}",
        )


if __name__ == "__main__":
    unittest.main()
