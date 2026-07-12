"""Mob screenshot fixture regression tests (Horn, TharaFrog, Alligator, Noxious)."""

from __future__ import annotations

import unittest

import cv2

from pybot.paths import PROJECT_ROOT
from pybot.recognition.detector.detector import MobDetector, load_detector_config
from pybot.recognition.fixtures import MOB_FIXTURE_SUITES, fixture_search_frame, MobFixtureImage, MobFixtureSuite


class MobFixtureSuiteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.detector = MobDetector(PROJECT_ROOT, load_detector_config())

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
            with self.subTest(suite=suite.folder):
                self._assert_suite_counts(suite)

    def _assert_suite_counts(self, suite: MobFixtureSuite) -> None:
        for image in suite.images():
            with self.subTest(suite=suite.folder, file=image.file_name):
                self._assert_image_count(suite, image)

    def _assert_image_count(self, suite: MobFixtureSuite, image: MobFixtureImage) -> None:
        frame = cv2.imread(str(image.path), cv2.IMREAD_COLOR)
        self.assertIsNotNone(frame, f"missing or unreadable fixture: {image.path}")

        frame = fixture_search_frame(frame)

        result = self.detector.detect(frame, suite.mob_name)
        accepted = len(result.accepted)
        expected = image.expected_count
        world = "gray" if image.gray_world else "normal"

        self.assertEqual(
            accepted,
            expected,
            f"{suite.folder}/{image.file_name} ({world}): expected {expected} accepted, got {accepted}",
        )


if __name__ == "__main__":
    unittest.main()
