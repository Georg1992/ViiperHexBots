"""Tests for moving-mob tracking: drift search geometry + discovery dedup slack."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
MOB_REC = Path(__file__).resolve().parent.parent
SIMPLE = MOB_REC / "simple"
for path in (str(MOB_REC), str(SIMPLE)):
    if path not in sys.path:
        sys.path.insert(0, path)

from detector import SimpleMobDetector, load_simple_config  # noqa: E402


def discovery_match_radius_px(
    coord_age_ms: int,
    *,
    base_radius: int = 90,
    state_interval_ms: int = 100,
    slack_per_tick: int = 30,
    pending: bool = False,
) -> float:
    """Mirror HuntTracks discovery match radius calculation."""
    movement_slack = (coord_age_ms / state_interval_ms) * slack_per_tick
    radius = base_radius + movement_slack
    if pending:
        radius *= 1.5
    return radius


def simulate_moving_track_dedup(
    track_xy: tuple[float, float],
    detection_xy: tuple[float, float],
    coord_age_ms: int,
    pending: bool = False,
) -> bool:
    dx = detection_xy[0] - track_xy[0]
    dy = detection_xy[1] - track_xy[1]
    dist = math.hypot(dx, dy)
    return dist <= discovery_match_radius_px(coord_age_ms, pending=pending)


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

    def test_discovery_dedup_while_mob_moves_between_state_ticks(self) -> None:
        track_pos = (400.0, 300.0)
        detection_pos = (435.0, 300.0)
        self.assertTrue(simulate_moving_track_dedup(track_pos, detection_pos, coord_age_ms=100))

    def test_discovery_dedup_after_one_second_scan_while_walking(self) -> None:
        track_pos = (400.0, 300.0)
        detection_pos = (470.0, 310.0)
        self.assertTrue(simulate_moving_track_dedup(track_pos, detection_pos, coord_age_ms=1000))

    def test_discovery_dedup_pending_track_wider_slack(self) -> None:
        track_pos = (400.0, 300.0)
        detection_pos = (800.0, 300.0)
        self.assertFalse(simulate_moving_track_dedup(track_pos, detection_pos, coord_age_ms=1000, pending=False))
        self.assertTrue(simulate_moving_track_dedup(track_pos, detection_pos, coord_age_ms=1000, pending=True))


if __name__ == "__main__":
    unittest.main()
