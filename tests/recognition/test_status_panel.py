"""Status panel SP/Weight parsing tests."""

from __future__ import annotations

import unittest

import cv2
import numpy as np

from pybot.paths import PROJECT_ROOT
from pybot.recognition.ui.status_panel import (
    clear_template_cache,
    find_status_panel,
    read_status_panel,
)

FIXTURES_DIR = PROJECT_ROOT / "tests"
# (filename, expected_sp, expected_sp_max, expected_weight, expected_weight_max)
FIXTURE_CASES: tuple[tuple[str, int, int, int, int], ...] = (
    ("StatusPanel.png", 1229, 1229, 587, 2630),
    ("StatusPanel2.png", 430, 430, 280, 2730),
    ("StatusPanel3.png", 430, 430, 280, 2730),
)


class StatusPanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        clear_template_cache()

    def _load(self, name: str) -> np.ndarray:
        path = FIXTURES_DIR / name
        if not path.is_file():
            self.skipTest(f"missing fixture {path}")
        frame = cv2.imread(str(path))
        if frame is None:
            self.skipTest(f"unreadable fixture {path}")
        return frame

    def test_finds_panel_near_top_left(self) -> None:
        frame = self._load("StatusPanel.png")
        origin = find_status_panel(frame)
        self.assertIsNotNone(origin)
        assert origin is not None
        self.assertLessEqual(origin[0], 20)
        self.assertLessEqual(origin[1], 20)

    def test_reads_sp_weight_from_fixtures(self) -> None:
        for name, sp, sp_max, weight, weight_max in FIXTURE_CASES:
            with self.subTest(fixture=name):
                frame = self._load(name)
                values = read_status_panel(frame)
                self.assertIsNotNone(values, f"failed to read {name}")
                assert values is not None
                self.assertEqual(values.sp, sp)
                self.assertEqual(values.sp_max, sp_max)
                self.assertEqual(values.weight, weight)
                self.assertEqual(values.weight_max, weight_max)
                self.assertEqual(values.panel_origin, (0, 0))


if __name__ == "__main__":
    unittest.main()
