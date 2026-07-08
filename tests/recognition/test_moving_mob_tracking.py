"""Tests for moving-mob tracking: drift search geometry."""

from __future__ import annotations

import math
import unittest
from pathlib import Path

from pybot.paths import PROJECT_ROOT
from pybot.recognition.simple.detector import SimpleMobDetector, load_simple_config

ROOT = PROJECT_ROOT


def max_search_distance(cx: int, cy: int, frame_shape: tuple[int, ...], detector: SimpleMobDetector) -> float:
    points = detector._track_search_centers(cx, cy, frame_shape)
    return max(math.hypot(px - cx, py - cy) for px, py in points)


class MovingMobTrackingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_simple_config()
        cls.detector = SimpleMobDetector(ROOT, cls.config)
        cls.drift_radius = int(cls.config["watchDriftRadiusPx"])
        cls.frame_shape = (900, 1600, 3)

    def test_config_drift_radius_supports_ro_walk_speed(self) -> None:
        px_per_tick = 15
        self.assertGreaterEqual(self.drift_radius, px_per_tick * 2)

    def test_state_search_grid_covers_walk_drift_per_tick(self) -> None:
        cx, cy = 800, 400
        max_dist = max_search_distance(cx, cy, self.frame_shape, self.detector)
        self.assertGreaterEqual(
            max_dist,
            30,
            msg="state drift grid must cover typical per-tick mob movement",
        )

    def test_state_search_reacquires_after_stale_coord_shift(self) -> None:
        cx, cy = 800, 400
        for shift in (12, 24, 36):
            stale_x = cx + shift
            stale_y = cy
            max_dist = max_search_distance(stale_x, stale_y, self.frame_shape, self.detector)
            self.assertGreaterEqual(
                max_dist,
                shift,
                msg=f"drift search from stale +{shift}px must reach true mob position",
            )


if __name__ == "__main__":
    unittest.main()
