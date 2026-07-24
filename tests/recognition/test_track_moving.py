"""Tests for track movement state."""

from __future__ import annotations

import unittest

from pybot.recognition.rules import (
    MobTrack,
    apply_movement_observation,
    evaluate_track_moving,
    movement_thresholds,
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
        move_px, stop_px = movement_thresholds(
            {
                "movementMoveThresholdPx": 12,
                "movementStopThresholdPx": 6,
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

    def test_hit_updates_velocity_and_miss_coasts_when_moving(self) -> None:
        from pybot.recognition.rules import apply_track_observation

        track = MobTrack(id=1, x=100, y=100, moving=True)
        apply_track_observation(
            track, found=True, x=120, y=104, confidence=0.8, now_tick=10
        )
        self.assertGreater(track.vel_x, 0.0)
        self.assertGreater(track.vel_y, 0.0)
        before_x, before_y = track.x, track.y
        vel_x, vel_y = track.vel_x, track.vel_y
        apply_track_observation(
            track, found=False, x=before_x, y=before_y, confidence=0.0, now_tick=20
        )
        self.assertEqual(track.lost_count, 1)
        self.assertEqual(track.x, before_x + int(round(vel_x)))
        self.assertEqual(track.y, before_y + int(round(vel_y)))

    def test_stationary_miss_does_not_coast(self) -> None:
        from pybot.recognition.rules import apply_track_observation

        track = MobTrack(id=1, x=100, y=100, moving=False, vel_x=8.0, vel_y=3.0)
        apply_track_observation(
            track, found=False, x=100, y=100, confidence=0.0, now_tick=20
        )
        self.assertEqual(track.lost_count, 1)
        self.assertEqual(track.x, 100)
        self.assertEqual(track.y, 100)
        self.assertLess(track.vel_x, 8.0)


if __name__ == "__main__":
    unittest.main()
