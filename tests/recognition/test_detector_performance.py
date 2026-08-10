"""Detector performance regressions for bounded silhouette work."""

from __future__ import annotations

import threading
import time
import unittest

import cv2
import numpy as np

from pybot.paths import PROJECT_ROOT
from pybot.recognition.detector.detector import (
    MobDetector,
    configure_opencv_runtime,
    load_detector_config,
)
from pybot.runtime.detection.detector_session import DetectorSession, StateTrackSnapshot
from pybot.runtime.capture.window_roi import HuntRoi
from pybot.recognition.detector.scoring.heatmap_detector import (
    HeatmapDetector,
    _palette_dist_sq,
    _pixel_dot,
    _pixel_palette_dot,
)


class DetectorPerformanceTests(unittest.TestCase):
    @staticmethod
    def _fixture_dir():
        return (
            PROJECT_ROOT
            / "pybot"
            / "recognition"
            / "test-fixtures"
            / "game-screenshots"
            / "Anubis"
        )

    def test_oversized_component_recovers_two_close_sprite_peaks(self) -> None:
        """A close non-overlapping pair must not collapse into one center."""
        detector = HeatmapDetector({
            "topCandidateCenters": 32,
            "minCenterHeat": 0.015,
            "peakRelativeThreshold": 0.25,
            "centerScales": [0.35, 0.45, 0.65, 0.85, 0.95, 1.1],
            "smallScaleMinFrameWidth": 512,
            "smallScaleCutoff": 0.75,
            "minBodyClusterStrong": 0.03,
            "minRequiredPaletteGroups": 2,
        })

        # A compact synthetic heatmap models the production shape: two strong
        # vertically separated peaks joined by a low bridge after blur. The
        # descriptor dimensions are Miyabi-like but no local screenshot/assets
        # are required for this unit-level blob test.
        class _Descriptor:
            avg_width = 114.0
            avg_height = 144.0

        yy, xx = np.mgrid[0:512, 0:512]
        heatmap = (
            0.95 * np.exp(-((xx - 120) ** 2 + (yy - 160) ** 2) / (2 * 35**2))
            + 0.85 * np.exp(-((xx - 120) ** 2 + (yy - 320) ** 2) / (2 * 35**2))
        ).astype(np.float32)
        blobs = detector.top_centers(heatmap, _Descriptor())

        self.assertEqual(len(blobs), 2)
        centers = sorted((x, y) for x, y, _score, _bbox in blobs)
        self.assertLess(abs(centers[0][0] - 120), 5)
        self.assertLess(abs(centers[1][0] - 120), 5)
        self.assertLess(abs(centers[0][1] - 160), 12)
        self.assertLess(abs(centers[1][1] - 320), 12)
        self.assertGreaterEqual(centers[1][1] - centers[0][1], 140)

    def test_clearly_oversized_single_lobe_is_not_split(self) -> None:
        """A single broad response with a weak shoulder stays one blob."""
        class _Descriptor:
            avg_width = 114.0
            avg_height = 144.0

        yy, xx = np.mgrid[0:512, 0:512]
        heatmap = (
            0.98 * np.exp(-((xx - 120) ** 2 + (yy - 220) ** 2) / (2 * 70**2))
            + 0.45 * np.exp(-((xx - 120) ** 2 + (yy - 335) ** 2) / (2 * 28**2))
        ).astype(np.float32)

        blobs = HeatmapDetector({
            "topCandidateCenters": 32,
            "minCenterHeat": 0.015,
            "peakRelativeThreshold": 0.25,
            "centerScales": [0.35, 0.45, 0.65, 0.85, 0.95, 1.1],
            "smallScaleMinFrameWidth": 512,
            "smallScaleCutoff": 0.75,
            "minBodyClusterStrong": 0.03,
            "minRequiredPaletteGroups": 2,
        }).top_centers(heatmap, _Descriptor())

        self.assertEqual(len(blobs), 1)

    def test_moderately_tall_single_component_is_not_split(self) -> None:
        """Two heat lobes below the oversized threshold remain one blob.

        The split is intended for a component that is clearly larger than one
        sprite, not for every tall or multi-lobed heat response. This protects
        single mobs whose palette response happens to have two local maxima.
        """
        class _Descriptor:
            avg_width = 114.0
            avg_height = 144.0

        yy, xx = np.mgrid[0:512, 0:512]
        heatmap = (
            0.95 * np.exp(-((xx - 120) ** 2 + (yy - 206) ** 2) / (2 * 40**2))
            + 0.90 * np.exp(-((xx - 120) ** 2 + (yy - 306) ** 2) / (2 * 40**2))
        ).astype(np.float32)

        blobs = HeatmapDetector({
            "topCandidateCenters": 32,
            "minCenterHeat": 0.015,
            "peakRelativeThreshold": 0.25,
            "centerScales": [0.35, 0.45, 0.65, 0.85, 0.95, 1.1],
            "smallScaleMinFrameWidth": 512,
            "smallScaleCutoff": 0.75,
            "minBodyClusterStrong": 0.03,
            "minRequiredPaletteGroups": 2,
        }).top_centers(heatmap, _Descriptor())

        self.assertEqual(len(blobs), 1)

    def test_pixel_dot_matches_numpy_dot(self) -> None:
        """The bounded pixel arithmetic must preserve detector scores."""
        rng = np.random.default_rng(7)
        pixels = rng.random((257, 3), dtype=np.float32)
        palette = rng.random((9, 3), dtype=np.float32)
        np.testing.assert_allclose(
            _pixel_dot(pixels, palette[0]),
            np.dot(pixels, palette[0]),
            rtol=2e-6,
            atol=2e-6,
        )
        np.testing.assert_allclose(
            _pixel_palette_dot(pixels, palette),
            np.dot(pixels, palette.T),
            rtol=2e-6,
            atol=2e-6,
        )
        np.testing.assert_allclose(
            _palette_dist_sq(pixels, palette),
            np.sum((pixels[:, None, :] - palette[None, :, :]) ** 2, axis=2),
            rtol=2e-6,
            atol=2e-6,
        )

    def test_runtime_explicitly_bounds_native_opencv_parallelism(self) -> None:
        # Detector construction is intentionally side-effect free; the live
        # runtime owns this process-wide setting. Restore the global OpenCV
        # setting so this test cannot affect unrelated recognition tests.
        previous = cv2.getNumThreads()
        try:
            cv2.setNumThreads(4)
            configure_opencv_runtime()
            self.assertEqual(cv2.getNumThreads(), 1)
        finally:
            cv2.setNumThreads(previous)

    def test_observer_sessions_run_in_parallel_without_shared_busy_state(self) -> None:
        """Independent discovery/tracking sessions both execute concurrently."""
        previous_threads = cv2.getNumThreads()
        configure_opencv_runtime()
        frame = cv2.imread(
            str(self._fixture_dir() / "3Anubis_Gray_ModifiedSprite.png"),
            cv2.IMREAD_COLOR,
        )
        self.assertIsNotNone(frame)
        assert frame is not None
        roi = HuntRoi(0, 0, frame.shape[1], frame.shape[0])
        discovery = DetectorSession("anubis", PROJECT_ROOT, use_sprite_grf=True)
        tracking = DetectorSession("anubis", PROJECT_ROOT, use_sprite_grf=True)
        snapshot = StateTrackSnapshot(
            track_id=1,
            x=frame.shape[1] // 2,
            y=frame.shape[0] // 2,
            scale=1.0,
        )
        barrier = threading.Barrier(3)
        heavy_entered = threading.Barrier(2)
        active = 0
        max_active = 0
        active_lock = threading.Lock()
        errors: list[BaseException] = []
        results: list[object] = []

        def enter_heavy_work() -> None:
            nonlocal active, max_active
            with active_lock:
                active += 1
                max_active = max(max_active, active)
            heavy_entered.wait(timeout=5.0)

        def leave_heavy_work() -> None:
            nonlocal active
            with active_lock:
                active -= 1

        def run_discovery() -> None:
            try:
                barrier.wait(timeout=5.0)
                original = discovery._detector.detect
                def detect(*args, **kwargs):
                    enter_heavy_work()
                    try:
                        return original(*args, **kwargs)
                    finally:
                        leave_heavy_work()
                discovery._detector.detect = detect
                results.append(discovery.discover_frame(frame, roi))
            except BaseException as exc:
                errors.append(exc)

        def run_tracking() -> None:
            try:
                barrier.wait(timeout=5.0)
                original = tracking._detector.track_local
                def track(*args, **kwargs):
                    enter_heavy_work()
                    try:
                        return original(*args, **kwargs)
                    finally:
                        leave_heavy_work()
                tracking._detector.track_local = track
                # A missing warm template is a valid independent-session call;
                # the overlap barrier is the property under test, not tracking
                # recognition quality.
                results.append(tracking.track_locals_frame(frame, roi, [snapshot]))
            except BaseException as exc:
                errors.append(exc)

        try:
            discovery_thread = threading.Thread(target=run_discovery)
            tracking_thread = threading.Thread(target=run_tracking)
            discovery_thread.start()
            tracking_thread.start()
            barrier.wait(timeout=5.0)
            discovery_thread.join(timeout=10.0)
            tracking_thread.join(timeout=10.0)
            self.assertFalse(discovery_thread.is_alive())
            self.assertFalse(tracking_thread.is_alive())
            self.assertEqual(errors, [])
            self.assertEqual(len(results), 2)
            self.assertTrue(all(getattr(result, "ok", False) for result in results))
            self.assertTrue(all(
                getattr(result, "fail_reason", "") != "detector_busy"
                for result in results
            ))
            self.assertEqual(max_active, 2)
        finally:
            cv2.setNumThreads(previous_threads)

    def test_body_diversity_heatmap_does_not_materialize_full_similarity_tensor(self) -> None:
        """Discovery diversity must stay 2-D on the live large-ROI path."""
        detector = MobDetector(PROJECT_ROOT, load_detector_config(), use_sprite_grf=False)
        descriptor = detector.ensure_descriptor("anubis")
        self.assertTrue(descriptor.use_body_cluster_diversity)
        frame_path = next(self._fixture_dir().glob("1Anubis*_Gray_ModifiedSprite.png"), None)
        self.assertIsNotNone(frame_path)
        assert frame_path is not None
        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        self.assertIsNotNone(frame)
        assert frame is not None

        from unittest.mock import patch
        from pybot.recognition.detector.scoring import heatmap_detector

        calls: list[bool] = []
        original = heatmap_detector.weighted_sprite_palette_heatmap

        def wrapped(*args, **kwargs):
            calls.append(bool(kwargs.get("return_similarity", False)))
            return original(*args, **kwargs)

        with patch.object(
            heatmap_detector,
            "weighted_sprite_palette_heatmap",
            side_effect=wrapped,
        ):
            heatmap = detector.heatmap_detector.build_sprite_heatmap(
                frame,
                descriptor,
                downscale=2,
            )

        # The work image is half-resolution internally, but the public heatmap
        # is upscaled back to frame coordinates for blob matching. Odd source
        # dimensions can lose one pixel in each axis during integer resize.
        self.assertLessEqual(frame.shape[0] - heatmap.shape[0], 1)
        self.assertLessEqual(frame.shape[1] - heatmap.shape[1], 1)
        self.assertGreaterEqual(heatmap.shape[0], frame.shape[0] - 1)
        self.assertGreaterEqual(heatmap.shape[1], frame.shape[1] - 1)
        self.assertTrue(calls)
        self.assertNotIn(True, calls)

    def test_silhouette_palette_heatmap_is_reused_for_multiple_gates(self) -> None:
        # The generic animated-sprite path still reuses one full-frame palette
        # map when several candidates need the same gate input. Static GRF mode
        # intentionally skips that optimization in favor of local palette gates.
        detector = MobDetector(PROJECT_ROOT, load_detector_config(), use_sprite_grf=False)
        # This is a normal animated-sprite multi-mob fixture, so the generic
        # path should build one shared full-frame palette map. Modified GRF mode
        # has a separate static fast path and is tested in test_grf_detector_mode.py.
        horn_dir = PROJECT_ROOT / "pybot" / "recognition" / "test-fixtures" / "game-screenshots" / "Horn"
        frame = cv2.imread(str(horn_dir / "2Horn.png"), cv2.IMREAD_COLOR)
        self.assertIsNotNone(frame)
        assert frame is not None

        calls = 0
        heatmap_ids: set[int] = set()
        original = detector._evaluate_silhouette_gate

        def wrapped(*args, **kwargs):
            nonlocal calls
            calls += 1
            heatmap = kwargs.get("palette_heatmap_full")
            self.assertIsNotNone(heatmap)
            heatmap_ids.add(id(heatmap))
            return original(*args, **kwargs)

        detector._evaluate_silhouette_gate = wrapped  # type: ignore[method-assign]
        result = detector.detect(frame, "horn")
        self.assertGreaterEqual(calls, 2)
        self.assertEqual(len(heatmap_ids), 1)
        self.assertEqual(calls, int(result.timing["silhouetteCheckCount"]))
        self.assertGreater(result.timing["silhouettePaletteHeatmap"], 0.0)

    def test_single_candidate_keeps_local_palette_heatmap_path(self) -> None:
        detector = MobDetector(PROJECT_ROOT, load_detector_config(), use_sprite_grf=True)
        single_fixture = next(self._fixture_dir().glob("1Anubis*_ModifiedSprite.png"), None)
        self.assertIsNotNone(single_fixture)
        assert single_fixture is not None
        frame = cv2.imread(str(single_fixture), cv2.IMREAD_COLOR)
        self.assertIsNotNone(frame)
        assert frame is not None

        received: list[object] = []
        original = detector._evaluate_silhouette_gate

        def wrapped(*args, **kwargs):
            received.append(kwargs.get("palette_heatmap_full"))
            return original(*args, **kwargs)

        detector._evaluate_silhouette_gate = wrapped  # type: ignore[method-assign]
        result = detector.detect(frame, "anubis")
        self.assertEqual(result.timing["blobCount"], 1.0)
        self.assertEqual(len(received), int(result.timing["silhouetteCheckCount"]))
        self.assertTrue(received)
        self.assertTrue(all(item is None for item in received))
        self.assertLess(result.timing["silhouettePaletteHeatmap"], 0.001)

    def test_oversized_deformation_is_descriptor_bounded(self) -> None:
        detector = MobDetector(PROJECT_ROOT, load_detector_config(), use_sprite_grf=False)
        descriptor = detector.ensure_descriptor("anubis")
        ref = detector._descriptor_silhouette_references(descriptor.silhouette_masks)[0]
        ref_avg, ref_stable = ref

        # A large noisy palette component is the production failure shape. The
        # deformation result must remain bounded even when the crop is wide.
        region = np.zeros((900, 1400, 3), dtype=np.uint8)
        region[:, :] = (80, 80, 80)
        cv2.rectangle(region, (200, 100), (1200, 800), (100, 130, 180), thickness=-1)

        started = time.perf_counter()
        result = detector._deform_silhouette_occupancy(
            region,
            descriptor,
            ref_avg,
            ref_stable,
        )
        elapsed = time.perf_counter() - started

        self.assertEqual(result.shape, region.shape[:2])
        self.assertLess(elapsed, 0.75)


if __name__ == "__main__":
    unittest.main()
