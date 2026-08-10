"""Local tracker vision core tests."""

from __future__ import annotations

import time
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import cv2
import numpy as np

from pybot.paths import PROJECT_ROOT, RECOGNITION_DIR
from pybot.recognition.cli import apply_scale_calibration
from pybot.recognition.fixtures import (
    MOB_FIXTURE_SUITES,
    default_horn_fixture,
    fixture_search_frame,
)
from pybot.recognition.detector.detector import MobDetector, load_detector_config
from pybot.recognition.detector.tracking.local_tracker import (
    LocalTrackResult,
    clear_track_states,
    transfer_track_state,
    _local_follow_scales,
    track_local,
)

ROOT = PROJECT_ROOT
MOB_REC = RECOGNITION_DIR


def playfield_roi(frame):
    height, width = frame.shape[:2]
    return frame[
        int(height * 0.08) : int(height * 0.92),
        int(width * 0.03) : int(width * 0.97),
    ]


class LocalTrackerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.base_config = load_detector_config()
        cls.fixture_dir = MOB_REC / "test-fixtures" / "game-screenshots"
        cls.frame = cv2.imread(str(default_horn_fixture()), cv2.IMREAD_COLOR)
        if cls.frame is None:
            raise unittest.SkipTest("fixture Horn/3Horn.png missing")
        cls.roi = playfield_roi(cls.frame)

    def _detector(self) -> MobDetector:
        calibrated = apply_scale_calibration(self.base_config, (0.82, 0.98), True)
        detector = MobDetector(ROOT, calibrated)
        detector.apply_runtime_config(calibrated)
        return detector

    def _living_anchor(self, detector: MobDetector):
        discovery = detector.detect(self.roi, "horn")
        living = [c for c in discovery.accepted]
        self.assertGreater(len(living), 0)
        return living[0]

    def _build_track_dict(self, anchor, track_id=1, **overrides) -> dict:
        """Build a track dict from a discovery anchor."""
        track = {
            "trackId": track_id,
            "x": anchor.center_x,
            "y": anchor.center_y,
            "scale": anchor.candidate_scale,
        }
        track.update(overrides)
        return track

    def test_finds_mob_at_discovery_coords(self) -> None:
        detector = self._detector()
        anchor = self._living_anchor(detector)
        track = self._build_track_dict(anchor, trackId=-1)
        result = track_local(detector, self.roi, "horn", track)
        self.assertIsInstance(result, LocalTrackResult)
        self.assertTrue(result.found)
        self.assertGreater(result.confidence, 0.0)
        self.assertGreater(result.opacity_score, 0.0)
        self.assertEqual(result.miss_reason, "")
        dist = abs(result.x - anchor.center_x) + abs(result.y - anchor.center_y)
        self.assertLess(dist, 40)

    def test_zero_track_id_is_rejected(self) -> None:
        detector = self._detector()
        result = track_local(
            detector,
            self.roi,
            "horn",
            {"trackId": 0, "x": 100, "y": 100, "scale": 0.9},
        )
        self.assertFalse(result.found)
        self.assertEqual(result.miss_reason, "invalid_track_id")

    def test_local_acquisition_recovers_from_offset_seed(self) -> None:
        """A provisional Track searches locally from its own seed position."""
        detector = self._detector()
        anchor = self._living_anchor(detector)
        track = self._build_track_dict(
            anchor,
            trackId=-101,
            x=anchor.center_x - 80,
            y=anchor.center_y,
        )
        result = track_local(detector, self.roi, "horn", track, search_radius_px=120)
        self.assertTrue(result.found, result.miss_reason)
        self.assertLess(abs(result.x - anchor.center_x), 40)
        self.assertLess(abs(result.y - anchor.center_y), 40)

    def test_miss_returns_meaningful_reason(self) -> None:
        detector = self._detector()
        # -1000,-1000 is well off-screen → heatmap peak search returns miss.
        track = {"trackId": -99, "x": -1000, "y": -1000, "scale": 0.9}
        result = track_local(detector, self.roi, "horn", track, search_radius_px=20)
        self.assertFalse(result.found)
        self.assertIn(result.miss_reason, ("no_peak", "below_threshold"))

    def test_finds_mob_within_search_radius_after_offset_seed(self) -> None:
        detector = self._detector()
        anchor = self._living_anchor(detector)
        track = self._build_track_dict(anchor, trackId=-3,
            x=anchor.center_x + 12, y=anchor.center_y + 8)
        result = track_local(detector, self.roi, "horn", track, search_radius_px=60)
        self.assertTrue(result.found)
        dist = abs(result.x - anchor.center_x) + abs(result.y - anchor.center_y)
        self.assertLess(dist, 50)

    def test_real_fast_frame_shift_recovery_stays_on_same_anchor(self) -> None:
        """A fast one-cycle displacement is recovered without generic reacquire."""
        detector = self._detector()
        anchor = self._living_anchor(detector)
        provisional = self._build_track_dict(anchor, track_id=-92)
        first = track_local(detector, self.roi, "horn", provisional)
        self.assertTrue(first.found, first.miss_reason)
        self.assertTrue(transfer_track_state(detector, -92, 92))

        shift_x, shift_y = 30, -8
        matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        shifted = cv2.warpAffine(
            self.roi,
            matrix,
            (self.roi.shape[1], self.roi.shape[0]),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        result = track_local(
            detector,
            shifted,
            "horn",
            self._build_track_dict(
                anchor,
                track_id=92,
                velX=shift_x,
                velY=shift_y,
                lost_count=1,
                prediction_valid=True,
                anchor_required=True,
            ),
        )

        self.assertTrue(result.found, result.miss_reason)
        self.assertLessEqual(abs(result.x - (first.x + shift_x)), 12)
        self.assertLessEqual(abs(result.y - (first.y + shift_y)), 12)

    def test_prediction_leads_cached_follow_after_a_miss(self) -> None:
        """Recovery searches ahead of the held coordinate for a fast mob."""
        detector = self._detector()
        anchor = self._living_anchor(detector)
        track = self._build_track_dict(
            anchor,
            track_id=91,
            velX=18.0,
            velY=-4.0,
            lost_count=1,
            prediction_valid=True,
            anchor_required=True,
        )
        # The visual anchor is already committed; replace the expensive matcher
        # only to assert the exact predicted center passed to the warm follower.
        from pybot.recognition.detector.tracking import local_tracker
        local_tracker._template_store(detector)[91] = SimpleNamespace(
            image_gray=np.ones((4, 4), dtype=np.uint8),
        )
        with patch.object(
            local_tracker,
            "_follow_cached_template",
            return_value=LocalTrackResult(
                track_id=91,
                found=True,
                x=anchor.center_x + 36,
                y=anchor.center_y - 8,
                confidence=0.9,
                miss_reason="",
            ),
        ) as follow:
            result = track_local(detector, self.roi, "horn", track)

        self.assertTrue(result.found)
        self.assertEqual(follow.call_args.kwargs["cx"], anchor.center_x + 36)
        self.assertEqual(follow.call_args.kwargs["cy"], anchor.center_y - 8)

    def test_wide_radius_finds_mob_lagged_behind_last_known(self) -> None:
        """No velocity needed — fixed wide disk around last-known catches runners."""
        detector = self._detector()
        anchor = self._living_anchor(detector)
        lag = 70
        self.assertLess(lag, detector.local_track_moving_search_radius_px)
        track = self._build_track_dict(
            anchor,
            trackId=-5,
            x=anchor.center_x - lag,
            y=anchor.center_y,
        )
        result = track_local(detector, self.roi, "horn", track)
        self.assertTrue(result.found, result.miss_reason)
        dist = abs(result.x - anchor.center_x) + abs(result.y - anchor.center_y)
        self.assertLess(dist, 50)

    def test_finds_mob_at_center_no_cache_state(self) -> None:
        detector = self._detector()
        anchor = self._living_anchor(detector)
        track = self._build_track_dict(anchor, trackId=-4)
        result = track_local(detector, self.roi, "horn", track)
        self.assertTrue(result.found)
        self.assertGreater(result.confidence, 0.0)

    def test_local_follow_scales_prefers_track_scale(self) -> None:
        self.assertEqual(
            _local_follow_scales([0.35, 0.45, 0.55, 0.82, 0.98], 0.90),
            [0.90, 0.98],
        )
        self.assertEqual(_local_follow_scales([0.35, 0.45], 0.35), [0.35, 0.45])
        self.assertEqual(_local_follow_scales([], 0.90), [0.90])

    def test_anubis_modified_sprite_tracking_is_fast_and_centered(self) -> None:
        """Large Anubis follows from a nearby center without prediction state."""
        detector = MobDetector(ROOT, self.base_config, use_sprite_grf=True)
        suite = next(
            suite for suite in MOB_FIXTURE_SUITES if suite.mob_name == "anubis"
        )
        image = next(
            image for image in suite.images()
            if "ModifiedSprite" in image.file_name
        )
        frame = cv2.imread(str(image.path), cv2.IMREAD_COLOR)
        self.assertIsNotNone(frame)
        assert frame is not None
        frame = fixture_search_frame(frame)
        discovery = detector.detect(frame, "anubis")
        self.assertGreater(len(discovery.accepted), 0)
        anchor = discovery.accepted[0]
        track = self._build_track_dict(
            anchor,
            trackId=-77,
            x=anchor.center_x - 60,
            y=anchor.center_y - 20,
        )
        started = time.perf_counter()
        result = track_local(detector, frame, "anubis", track)
        elapsed = time.perf_counter() - started
        self.assertTrue(result.found, result.miss_reason)
        self.assertLess(elapsed, 0.5)

    def test_area_reset_has_no_temporal_state_to_clear(self) -> None:
        """The centered follower keeps no stale screen-local cache."""
        detector = self._detector()
        clear_track_states(detector)
        self.assertFalse(hasattr(detector, "_local_track_states"))

    def test_centered_follow_reacquires_current_bbox_center(self) -> None:
        """A fresh-frame local hit publishes the accepted bbox center directly."""
        detector = self._detector()
        anchor = self._living_anchor(detector)
        track = self._build_track_dict(anchor, track_id=88)
        result = track_local(detector, self.roi, "horn", track)
        self.assertTrue(result.found, result.miss_reason)
        self.assertLessEqual(abs(result.x - anchor.center_x), 12)
        self.assertLessEqual(abs(result.y - anchor.center_y), 12)
        self.assertFalse(hasattr(detector, "_local_track_states"))

    def test_benchmark_one_three_six_tracks(self) -> None:
        detector = self._detector()
        discovery = detector.detect(self.roi, "horn")
        living = [c for c in discovery.accepted][:6]
        if len(living) < 3:
            self.skipTest("fixture needs at least 3 living horns")

        def bench(tracks: list[dict]) -> float:
            start = time.perf_counter()
            for track in tracks:
                track_local(detector, self.roi, "horn", track)
            return time.perf_counter() - start

        one = [
            {
                "trackId": -1,
                "x": living[0].center_x,
                "y": living[0].center_y,
                "scale": living[0].candidate_scale,
            }
        ]
        three = [
            {
                "trackId": -(index + 1),
                "x": candidate.center_x,
                "y": candidate.center_y,
                "scale": candidate.candidate_scale,
            }
            for index, candidate in enumerate(living[:3])
        ]
        six = [
            {
                "trackId": -(index + 1),
                "x": candidate.center_x,
                "y": candidate.center_y,
                "scale": candidate.candidate_scale,
            }
            for index, candidate in enumerate(living[:6])
        ]

        elapsed_one = bench(one)
        elapsed_three = bench(three)
        elapsed_six = bench(six)

        print(
            f"\nlocal_track bench: 1={elapsed_one:.3f}s "
            f"3={elapsed_three:.3f}s 6={elapsed_six:.3f}s"
        )
        self.assertLess(elapsed_one, 0.5)
        self.assertLess(elapsed_three, 1.5)
        self.assertLess(elapsed_six, 3.0)


if __name__ == "__main__":
    unittest.main()
