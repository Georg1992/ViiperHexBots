"""Danger assessment: foreign sprites near the character."""

from __future__ import annotations

import unittest
from pathlib import Path

import cv2
import numpy as np

from pybot.recognition.danger import assess_danger, count_near_objects

_TESTS = Path(__file__).resolve().parents[1]
_FIXTURES = [
    _TESTS / "sit1.png",
    _TESTS / "sit2.png",
    _TESTS / "Stand1.png",
    _TESTS / "Stand2.png",
]


class DangerNearObjectsTests(unittest.TestCase):
    def test_fixtures_have_no_near_foreign_objects(self) -> None:
        for path in _FIXTURES:
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            self.assertIsNotNone(img, msg=str(path))
            report = assess_danger(img, cell_size_px=64)
            self.assertFalse(
                report.in_danger,
                msg=f"{path.name}: {report.reasons}",
            )
            self.assertEqual(report.near_object_count, 0)

    def test_synthetic_nearby_blob_is_danger(self) -> None:
        img = cv2.imread(str(_TESTS / "sit1.png"), cv2.IMREAD_COLOR)
        self.assertIsNotNone(img)
        assert img is not None
        h, w = img.shape[:2]
        cx, cy = w // 2, h // 2 + 8
        # Opaque mob-sized blob ~70px to the right (inside 1.5 cells).
        cv2.rectangle(
            img,
            (cx + 50, cy - 30),
            (cx + 90, cy + 30),
            (40, 80, 200),
            -1,
        )
        report = assess_danger(img, cell_size_px=64)
        self.assertTrue(report.in_danger)
        self.assertGreaterEqual(report.near_object_count, 1)
        self.assertTrue(any(r.startswith("near_objects:") for r in report.reasons))

    def test_distant_blob_is_not_danger(self) -> None:
        img = cv2.imread(str(_TESTS / "sit1.png"), cv2.IMREAD_COLOR)
        self.assertIsNotNone(img)
        assert img is not None
        h, w = img.shape[:2]
        cx, cy = w // 2, h // 2 + 8
        # Far outside 1.5 cells (~96px).
        cv2.rectangle(
            img,
            (cx + 200, cy - 30),
            (cx + 240, cy + 30),
            (40, 80, 200),
            -1,
        )
        near = count_near_objects(img, cell_size_px=64, near_cells=1.5)
        self.assertEqual(near.count, 0)

    def test_hp_drop_is_danger(self) -> None:
        img = cv2.imread(str(_TESTS / "sit1.png"), cv2.IMREAD_COLOR)
        self.assertIsNotNone(img)
        assert img is not None
        report = assess_danger(img, cell_size_px=64, hp=900, previous_hp=1000)
        self.assertTrue(report.in_danger)
        self.assertTrue(report.hp_dropped)
        self.assertTrue(any(r.startswith("hp_drop:") for r in report.reasons))

    def test_stable_hp_alone_is_not_danger(self) -> None:
        img = cv2.imread(str(_TESTS / "sit1.png"), cv2.IMREAD_COLOR)
        self.assertIsNotNone(img)
        assert img is not None
        report = assess_danger(img, cell_size_px=64, hp=1000, previous_hp=1000)
        self.assertFalse(report.in_danger)
        self.assertFalse(report.hp_dropped)


if __name__ == "__main__":
    unittest.main()
