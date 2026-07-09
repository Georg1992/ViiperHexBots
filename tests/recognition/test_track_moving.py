"""Tests for track movement state used by death detection."""

from __future__ import annotations

import unittest

from pybot.recognition.rules import (
    MobTrack,
    apply_movement_observation,
    death_movement_thresholds,
    evaluate_track_moving,
)


class TrackMovingTests(unittest.TestCase):
    def test_enters_moving_above_enter_threshold(self) -> None:
        self.assertTrue(
            evaluate_track_moving(
                was_moving=False,
                displacement_sq=13 * 13,
                move_threshold_px=12,
                stop_threshold_px=6,
            )
        )

    def test_stays_stationary_below_enter_threshold(self) -> None:
        self.assertFalse(
            evaluate_track_moving(
                was_moving=False,
                displacement_sq=10 * 10,
                move_threshold_px=12,
                stop_threshold_px=6,
            )
        )

    def test_hysteresis_keeps_moving_until_stop_threshold(self) -> None:
        self.assertTrue(
            evaluate_track_moving(
                was_moving=True,
                displacement_sq=8 * 8,
                move_threshold_px=12,
                stop_threshold_px=6,
            )
        )
        self.assertFalse(
            evaluate_track_moving(
                was_moving=True,
                displacement_sq=5 * 5,
                move_threshold_px=12,
                stop_threshold_px=6,
            )
        )

    def test_apply_movement_observation_updates_track(self) -> None:
        track = MobTrack(id=1, x=100, y=100)
        move_px, stop_px = death_movement_thresholds(
            {
                "deathOpacityMoveThresholdPx": 12,
                "deathOpacityStopThresholdPx": 6,
            }
        )
        apply_movement_observation(
            track,
            x=120,
            y=100,
            move_threshold_px=move_px,
            stop_threshold_px=stop_px,
        )
        self.assertTrue(track.moving)


if __name__ == "__main__":
    unittest.main()
