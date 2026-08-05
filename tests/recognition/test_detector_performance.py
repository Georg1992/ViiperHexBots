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

    def test_independent_observer_sessions_can_run_concurrently(self) -> None:
        """Regression for the post-sit observer oversubscription path.

        Production deliberately has separate discovery/tracking sessions, so
        their Python locks do not serialize native OpenCV work. Both calls must
        be able to finish when launched together after runtime initialization;
        this guards the topology that previously produced multi-second spikes.
        """
        previous_threads = cv2.getNumThreads()
        configure_opencv_runtime()
        frame = cv2.imread(
            str(self._fixture_dir() / "3Anubis_Gray_ModifiedSprite.png"),
            cv2.IMREAD_COLOR,
        )
        self.assertIsNotNone(frame)
        assert frame is not None
        roi = HuntRoi(0, 0, frame.shape[1], frame.shape[0])
        discovery = DetectorSession(
            "anubis", PROJECT_ROOT, use_sprite_grf=True,
        )
        tracking = DetectorSession(
            "anubis", PROJECT_ROOT, use_sprite_grf=True,
        )
        snapshot = StateTrackSnapshot(
            track_id=1,
            x=frame.shape[1] // 2,
            y=frame.shape[0] // 2,
            scale=1.0,
            now_tick=1,
        )
        barrier = threading.Barrier(3)
        errors: list[BaseException] = []
        durations: list[float] = []
        duration_lock = threading.Lock()

        def run_discovery() -> None:
            try:
                barrier.wait(timeout=5.0)
                started = time.perf_counter()
                discovery.discover_frame(frame, roi)
                elapsed = time.perf_counter() - started
                with duration_lock:
                    durations.append(elapsed)
            except BaseException as exc:  # noqa: BLE001 - test thread transport
                errors.append(exc)

        def run_tracking() -> None:
            try:
                barrier.wait(timeout=5.0)
                started = time.perf_counter()
                tracking.track_locals_frame(frame, roi, [snapshot])
                elapsed = time.perf_counter() - started
                with duration_lock:
                    durations.append(elapsed)
            except BaseException as exc:  # noqa: BLE001 - test thread transport
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
            self.assertEqual(len(durations), 2)
            # The old post-sit trace was 2.9–13.8s on this fixture. Keep a
            # generous ceiling to catch a recurrence without making the test
            # depend on sub-second hardware timing.
            self.assertLess(max(durations), 5.0)
        finally:
            cv2.setNumThreads(previous_threads)

    def test_silhouette_palette_heatmap_is_reused_for_multiple_gates(self) -> None:
        detector = MobDetector(PROJECT_ROOT, load_detector_config(), use_sprite_grf=True)
        frame = cv2.imread(
            str(self._fixture_dir() / "3Anubis_Gray_ModifiedSprite.png"),
            cv2.IMREAD_COLOR,
        )
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
        result = detector.detect(frame, "anubis")
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
