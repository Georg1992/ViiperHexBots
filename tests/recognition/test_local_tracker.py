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
from pybot.recognition.fixtures import default_horn_fixture
from pybot.recognition.detector.detector import MobDetector, load_detector_config
from pybot.recognition.detector.tracking.local_tracker import (
    LocalTrackResult,
    _local_follow_scales,
    _refine_hit_to_sprite_center,
    clear_track_states,
    track_local,
    transfer_track_state,
    _TrackAnchor,
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

    def test_center_projection_fails_closed_without_current_sprite_mask(self) -> None:
        """A non-sprite frame cannot preserve/publish the old coordinate."""
        detector = self._detector()
        descriptor = detector.ensure_descriptor("horn")
        blank = np.zeros_like(self.roi)
        self.assertIsNone(
            _refine_hit_to_sprite_center(detector, blank, descriptor, 100, 100, 0.9),
        )

    def test_tracking_fails_closed_when_current_center_is_unavailable(self) -> None:
        """A raw heat/template hit is never published without a sprite center."""
        detector = self._detector()
        anchor = self._living_anchor(detector)
        from pybot.recognition.detector.tracking import local_tracker

        with patch.object(
            local_tracker,
            "_find_local_peak",
            return_value=(anchor.center_x, anchor.center_y, 1.0, 0.9, (
                anchor.center_x - 10, anchor.center_y - 10, 20, 20,
            )),
        ), patch.object(
            local_tracker,
            "_refine_hit_to_sprite_center",
            return_value=None,
        ):
            result = track_local(
                detector,
                self.roi,
                "horn",
                self._build_track_dict(anchor, track_id=-404),
            )

        self.assertFalse(result.found)
        self.assertEqual(result.miss_reason, "center_projection_failed")

    def test_finds_mob_at_discovery_coords(self) -> None:
        detector = self._detector()
        anchor = self._living_anchor(detector)
        track = self._build_track_dict(anchor, trackId=-1)
        from pybot.recognition.detector.tracking import local_tracker

        with patch.object(
            local_tracker,
            "measure_opacity_score",
            return_value=0.77,
        ):
            result = track_local(detector, self.roi, "horn", track)

        self.assertIsInstance(result, LocalTrackResult)
        self.assertTrue(result.found)
        self.assertGreater(result.confidence, 0.0)
        self.assertEqual(result.opacity_score, 0.77)
        # Warm tracking intentionally does not run opacity probing; it must
        # keep the attack-coordinate path lightweight.
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
            anchors=(_TrackAnchor(np.ones((4, 4), dtype=np.uint8), 0, 0),),
            width=8,
            height=8,
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

    def test_fast_velocity_expands_tracking_roi_ahead_of_last_coordinate(self) -> None:
        """Fast runners get a wider predicted ROI before they can leave it."""
        detector = self._detector()
        anchor = self._living_anchor(detector)
        track_id = 902
        from pybot.recognition.detector.tracking import local_tracker

        local_tracker._template_store(detector)[track_id] = SimpleNamespace(
            anchors=(_TrackAnchor(np.ones((4, 4), dtype=np.uint8), 0, 0),),
            width=8,
            height=8,
        )
        with patch.object(
            local_tracker,
            "_follow_cached_template",
            return_value=LocalTrackResult(
                track_id=track_id,
                found=True,
                x=anchor.center_x + 360,
                y=anchor.center_y,
                confidence=0.9,
                miss_reason="",
            ),
        ) as follow:
            result = track_local(
                detector,
                self.roi,
                "horn",
                self._build_track_dict(
                    anchor,
                    track_id=track_id,
                    velX=180.0,
                    velY=0.0,
                    lost_count=1,
                    prediction_valid=True,
                    anchor_required=True,
                ),
            )

        self.assertTrue(result.found)
        kwargs = follow.call_args.kwargs
        self.assertGreater(
            kwargs["search_radius_px"],
            detector.local_track_moving_search_radius_px,
        )
        self.assertEqual(kwargs["cx"], anchor.center_x + 360)
        self.assertEqual(kwargs["cy"], anchor.center_y)
        self.assertEqual(kwargs["identity_cx"], anchor.center_x + 360)
        self.assertLess(kwargs["identity_radius_px"], kwargs["search_radius_px"])

    def test_template_recovery_prefers_valid_corridor_over_identical_neighbor(self) -> None:
        """A stronger neighbor outside the corridor cannot steal the Track."""
        detector = self._detector()
        descriptor = detector.ensure_descriptor("horn")
        from pybot.recognition.detector.tracking import local_tracker

        rng = np.random.default_rng(7)
        template = rng.integers(0, 255, size=(12, 12), dtype=np.uint8)
        frame_gray = rng.integers(0, 35, size=(180, 260), dtype=np.uint8)
        desired = cv2.resize(template, (24, 24), interpolation=cv2.INTER_NEAREST)
        # The desired target is deliberately a little weaker than the exact
        # identical neighbor, so an unmasked global minMaxLoc would choose the
        # wrong mob. The corridor is centered on the desired target.
        weakened = desired.copy()
        weakened[::3, ::3] = weakened[::3, ::3] // 2
        frame_gray[68:92, 103:127] = weakened
        frame_gray[68:92, 183:207] = desired
        frame = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
        track_id = 901
        local_tracker._template_store(detector)[track_id] = SimpleNamespace(
            anchors=(
                _TrackAnchor(template, 0, 0),
                _TrackAnchor(template, 0, 0),
            ),
            width=24,
            height=24,
            center_x=115,
            center_y=80,
            scale=0.9,
        )

        with patch.object(
            local_tracker,
            "_refine_hit_to_sprite_center",
            side_effect=lambda _detector, _frame, _descriptor, cx, cy, _scale: (cx, cy),
        ), patch.object(
            local_tracker,
            "measure_opacity_score",
            return_value=0.66,
        ):
            result = local_tracker._follow_cached_template(
                detector,
                frame,
                descriptor,
                track_id=track_id,
                cx=115,
                cy=80,
                scale=0.9,
                search_radius_px=110,
                suppress_positions=None,
                offset_x=0,
                offset_y=0,
                identity_cx=115,
                identity_cy=80,
                identity_radius_px=28,
            )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.opacity_score, 0.66)
        self.assertLessEqual(abs(result.x - 115), 4)
        self.assertLessEqual(abs(result.y - 80), 4)

    def test_acquired_multi_anchor_follow_survives_cursor_occlusion(self) -> None:
        """Real acquisition stores corner anchors that survive a cursor block."""
        detector = self._detector()
        anchor = self._living_anchor(detector)
        track_id = 946
        provisional = self._build_track_dict(anchor, track_id=-track_id)
        first = track_local(detector, self.roi, "horn", provisional)
        self.assertTrue(first.found, first.miss_reason)
        template = detector._local_track_templates[-track_id]
        self.assertGreaterEqual(len(template.anchors), 2)
        self.assertTrue(transfer_track_state(detector, -track_id, track_id))

        shifted = cv2.warpAffine(
            self.roi,
            np.float32([[1, 0, 12], [0, 1, 8]]),
            (self.roi.shape[1], self.roi.shape[0]),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        # Cover the translated sprite center with a cursor-sized block. Corner
        # anchors remain available; center refinement must use the dominant
        # current sprite component rather than the black hole pixel.
        cursor_x = first.x + 12
        cursor_y = first.y + 8
        desc = detector.ensure_descriptor("horn")
        cursor_w = max(8, int(round(desc.avg_width * 0.22)))
        cursor_h = max(8, int(round(desc.avg_height * 0.22)))
        shifted[
            cursor_y - cursor_h // 2:cursor_y + cursor_h // 2,
            cursor_x - cursor_w // 2:cursor_x + cursor_w // 2,
        ] = 0
        result = track_local(
            detector,
            shifted,
            "horn",
            self._build_track_dict(
                anchor,
                track_id=track_id,
                x=anchor.center_x,
                y=anchor.center_y,
                velX=12,
                velY=8,
                prediction_valid=True,
                anchor_required=True,
            ),
        )
        self.assertTrue(result.found, result.miss_reason)
        self.assertLessEqual(abs(result.x - (first.x + 12)), 16)
        self.assertLessEqual(abs(result.y - (first.y + 8)), 16)

    def test_multi_anchor_follow_survives_game_cursor_occluding_one_quadrant(self) -> None:
        """Remaining sprite anchors keep the Track on the same moving mob."""
        detector = self._detector()
        descriptor = detector.ensure_descriptor("horn")
        from pybot.recognition.detector.tracking import local_tracker

        rng = np.random.default_rng(41)
        sprite = rng.integers(20, 235, size=(48, 48), dtype=np.uint8)
        previous = np.zeros((190, 260), dtype=np.uint8)
        previous[56:104, 82:130] = sprite
        current = np.zeros_like(previous)
        current[64:112, 96:144] = sprite
        # The large game cursor covers the upper-left sprite quadrant. The
        # other independent anchors remain visible in the current frame.
        current[64:84, 96:118] = 0

        anchors = tuple(
            _TrackAnchor(
                cv2.resize(
                    sprite[y:y + 12, x:x + 12],
                    (6, 6),
                    interpolation=cv2.INTER_AREA,
                ),
                x + 6 - 24,
                y + 6 - 24,
            )
            for x, y in ((2, 2), (34, 2), (2, 34), (34, 34))
        )
        track_id = 944
        local_tracker._template_store(detector)[track_id] = SimpleNamespace(
            anchors=anchors,
            width=48,
            height=48,
        )
        frame_bgr = cv2.cvtColor(current, cv2.COLOR_GRAY2BGR)
        with patch.object(
            local_tracker,
            "_refine_hit_to_sprite_center",
            side_effect=lambda _detector, _frame, _descriptor, cx, cy, _scale: (cx, cy),
        ):
            result = local_tracker._follow_cached_template(
                detector,
                frame_bgr,
                descriptor,
                track_id=track_id,
                cx=120,
                cy=88,
                scale=0.9,
                search_radius_px=35,
                suppress_positions=None,
                offset_x=0,
                offset_y=0,
                identity_cx=120,
                identity_cy=88,
                identity_radius_px=35,
            )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertLessEqual(abs(result.x - 120), 4)
        self.assertLessEqual(abs(result.y - 88), 4)
        self.assertGreaterEqual(result.confidence, 0.42)

    def test_large_displacement_recovery_stays_on_predicted_target(self) -> None:
        """A fast target is recovered from a large jump without neighbor theft."""
        detector = self._detector()
        from pybot.recognition.detector.tracking import local_tracker

        rng = np.random.default_rng(123)
        template = rng.integers(0, 255, size=(12, 12), dtype=np.uint8)
        desired = cv2.resize(template, (24, 24), interpolation=cv2.INTER_NEAREST)
        frame_gray = np.zeros((220, 700), dtype=np.uint8)
        # The target moved 300px from the last coordinate. An identical neighbor
        # is inside the broad search ROI but outside the predicted identity band.
        frame_gray[88:112, 388:412] = desired
        frame_gray[88:112, 548:572] = desired
        frame = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
        track_id = 903
        local_tracker._template_store(detector)[track_id] = SimpleNamespace(
            anchors=(
                _TrackAnchor(template, 0, 0),
                _TrackAnchor(template, 0, 0),
            ),
            width=24,
            height=24,
            center_x=100,
            center_y=100,
            scale=0.9,
        )

        with patch.object(
            local_tracker,
            "_refine_hit_to_sprite_center",
            side_effect=lambda _detector, _frame, _descriptor, cx, cy, _scale: (cx, cy),
        ):
            result = track_local(
                detector,
                frame,
                "horn",
                {
                    "trackId": track_id,
                    "x": 100,
                    "y": 100,
                    "scale": 0.9,
                    "velX": 150.0,
                    "velY": 0.0,
                    "lost_count": 1,
                    "prediction_valid": True,
                    "anchor_required": True,
                },
            )

        self.assertTrue(result.found, result.miss_reason)
        self.assertLessEqual(abs(result.x - 400), 4)
        self.assertLessEqual(abs(result.y - 100), 4)

    def test_expanded_recovery_keeps_same_template_identity_bound(self) -> None:
        """A wider miss recovery also widens its bound without generic reacquire."""
        detector = self._detector()
        anchor = self._living_anchor(detector)
        track_id = 93
        from pybot.recognition.detector.tracking import local_tracker

        local_tracker._template_store(detector)[track_id] = SimpleNamespace(
            anchors=(_TrackAnchor(np.ones((4, 4), dtype=np.uint8), 0, 0),),
            width=8,
            height=8,
        )
        calls: list[dict] = []

        def follow(_detector, _frame, _descriptor, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                return None
            return LocalTrackResult(
                track_id=track_id,
                found=True,
                x=anchor.center_x + 120,
                y=anchor.center_y,
                confidence=0.9,
                miss_reason="",
            )

        with patch.object(local_tracker, "_follow_cached_template", side_effect=follow):
            result = track_local(
                detector,
                self.roi,
                "horn",
                self._build_track_dict(
                    anchor,
                    track_id=track_id,
                    velX=60.0,
                    velY=0.0,
                    lost_count=1,
                    prediction_valid=True,
                    anchor_required=True,
                ),
            )

        self.assertTrue(result.found)
        self.assertEqual(len(calls), 2)
        self.assertGreater(calls[1]["search_radius_px"], calls[0]["search_radius_px"])
        # Recovery searches a wider crop, but identity stays in the normal
        # motion corridor around the predicted center. This is the sticky
        # boundary that prevents an identical neighbor at the far edge from
        # winning the expanded template search.
        self.assertEqual(
            calls[1]["identity_radius_px"], calls[0]["identity_radius_px"],
        )
        self.assertEqual(calls[0]["identity_cx"], anchor.center_x + 120)
        self.assertEqual(calls[1]["identity_cx"], anchor.center_x + 120)
        self.assertEqual(calls[1]["identity_cy"], anchor.center_y)
        self.assertLess(calls[0]["identity_radius_px"], calls[0]["search_radius_px"])

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

    def test_warm_follow_benchmark_stays_cheap_per_track(self) -> None:
        """Warm multi-anchor follow must stay cheap as track count grows.

        The full-crop matchTemplate path cost ~7 ms per warm track, so a
        3-mob batch alone blew the 20 ms tracking budget and mobs fell behind.
        Per-anchor local windows cut this below 5 ms per track; assert a bound
        that separates the two implementations without being CI-flaky.
        """
        detector = self._detector()
        discovery = detector.detect(self.roi, "horn")
        living = [c for c in discovery.accepted][:6]
        if len(living) < 2:
            self.skipTest("fixture needs at least 2 living horns")

        warm: list[dict] = []
        for index, candidate in enumerate(living):
            provisional = self._build_track_dict(candidate, track_id=-(index + 1))
            first = track_local(detector, self.roi, "horn", provisional)
            if not first.found:
                self.skipTest("fixture acquisition failed")
            track_id = index + 1
            self.assertTrue(transfer_track_state(detector, -(index + 1), track_id))
            warm.append({
                "trackId": track_id,
                "x": first.x,
                "y": first.y,
                "scale": candidate.candidate_scale,
                "prediction_valid": True,
                "lost_count": 0,
                "anchor_required": True,
            })

        def bench(tracks: list[dict]) -> float:
            start = time.perf_counter()
            for _ in range(5):
                for track in tracks:
                    result = track_local(detector, self.roi, "horn", track)
                    self.assertTrue(result.found, result.miss_reason)
            return (time.perf_counter() - start) / 5.0 * 1000.0

        elapsed = bench(warm)
        print(
            f"\nwarm follow: {len(warm)} track(s) = {elapsed:.2f} ms/frame "
            f"({elapsed / max(1, len(warm)):.2f} ms/track)"
        )
        # ~2 ms/track after the local-window change; 12 ms/track is a
        # generous CI-safe ceiling that the old full-crop path exceeded.
        self.assertLess(elapsed / max(1, len(warm)), 12.0)

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
