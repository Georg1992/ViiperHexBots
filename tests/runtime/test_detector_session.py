"""DetectorSession tests using fixture frames (no live capture)."""

from __future__ import annotations

import threading
import time
import unittest
from unittest.mock import patch

import cv2

from pybot.recognition.detector.tracking.local_tracker import LocalTrackResult

from pybot.paths import PROJECT_ROOT
from pybot.recognition.fixtures import default_horn_fixture
from pybot.runtime.capture.window_roi import HuntRoi
from pybot.runtime.detection.detector_session import DetectorSession, StateTrackSnapshot
from pybot.recognition.detector.detector import load_detector_config

ROOT = PROJECT_ROOT
FIXTURE = default_horn_fixture()


def playfield_roi(frame):
    height, width = frame.shape[:2]
    return frame[
        int(height * 0.08) : int(height * 0.92),
        int(width * 0.03) : int(width * 0.97),
    ]


class DetectorSessionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.frame = cv2.imread(str(FIXTURE), cv2.IMREAD_COLOR)
        if cls.frame is None:
            raise unittest.SkipTest("fixture Horn/3Horn.png missing")
        cls.roi_frame = playfield_roi(cls.frame)
        cls.roi = HuntRoi(x=0, y=0, w=cls.roi_frame.shape[1], h=cls.roi_frame.shape[0])
        cls.detector = DetectorSession("horn", project_root=ROOT)

    def test_discover_finds_living_candidates(self) -> None:
        scan = self.detector.discover_frame(self.roi_frame, self.roi)
        self.assertTrue(scan.ok)
        self.assertGreater(scan.raw_count, 0)
        living = [d for d in scan.detections if d.living]
        self.assertGreater(len(living), 0)
        self.assertGreater(scan.duration_ms, 0)
        # Stage timing splits into session-lock wait and detector work. An
        # uncontended call should spend almost all time in the detector itself.
        self.assertEqual(scan.duration_ms, scan.lock_wait_ms + scan.detect_ms)
        self.assertGreater(scan.detect_ms, 0)
        self.assertLess(scan.lock_wait_ms, 50)

    def test_parallel_tracking_is_bounded_and_preserves_snapshot_order(self) -> None:
        """Independent jobs overlap without making result order nondeterministic."""
        config = load_detector_config()
        config["localTrackingWorkerCount"] = 3
        session = DetectorSession(
            "horn", project_root=ROOT, detector_config=config,
        )
        snapshots = [
            StateTrackSnapshot(
                track_id=index + 1,
                x=100 + index * 40,
                y=100,
                scale=1.0,
            )
            for index in range(7)
        ]
        active = 0
        maximum = 0
        lock = threading.Lock()
        completed: list[int] = []

        def fake_track(_frame, _mob_name, track, **_kwargs):
            nonlocal active, maximum
            with lock:
                active += 1
                maximum = max(maximum, active)
            time.sleep(0.01)
            with lock:
                active -= 1
                completed.append(int(track["trackId"]))
            return LocalTrackResult(
                track_id=int(track["trackId"]),
                found=True,
                x=int(track["x"]),
                y=int(track["y"]),
                confidence=1.0,
                miss_reason="",
            )

        try:
            with patch.object(session._detector, "ensure_descriptor"):
                with patch.object(session._detector, "track_local", side_effect=fake_track):
                    batch = session.track_locals_frame(
                        self.roi_frame,
                        self.roi,
                        snapshots,
                    )
        finally:
            session.close()

        self.assertEqual(maximum, 3)
        self.assertEqual(
            [result.track_id for result in batch.results],
            [snapshot.track_id for snapshot in snapshots],
        )
        self.assertEqual(set(completed), {snapshot.track_id for snapshot in snapshots})
        self.assertEqual(batch.found_count, len(snapshots))
        self.assertLess(batch.duration_ms, 60)
        self.assertEqual(set(batch.track_durations_ms), set(completed))

    def test_async_tracking_drops_completion_after_reset(self) -> None:
        """A completion from an invalidated generation is never delivered."""
        config = load_detector_config()
        config["localTrackingWorkerCount"] = 2
        session = DetectorSession("horn", project_root=ROOT, detector_config=config)
        started = threading.Event()
        release = threading.Event()
        callbacks: list[int] = []
        snapshots = [
            StateTrackSnapshot(track_id=1, x=200, y=200, scale=1.0),
        ]

        def fake_track(_frame, _mob_name, track, **_kwargs):
            started.set()
            release.wait(timeout=2.0)
            return LocalTrackResult(
                track_id=int(track["trackId"]),
                found=True,
                x=int(track["x"]),
                y=int(track["y"]),
                confidence=1.0,
                miss_reason="",
            )

        try:
            with patch.object(session._detector, "ensure_descriptor"):
                with patch.object(session._detector, "track_local", side_effect=fake_track):
                    accepted = session.submit_track_locals_frame(
                        self.roi_frame,
                        self.roi,
                        snapshots,
                        on_result=lambda result: callbacks.append(result.track_id),
                    )
                    self.assertEqual(accepted, [1])
                    self.assertTrue(started.wait(timeout=1.0))
                    session.clear_track_templates()
                    release.set()
                    deadline = time.monotonic() + 2.0
                    while time.monotonic() < deadline and session._tracking_inflight:
                        time.sleep(0.01)
            self.assertEqual(callbacks, [])
            self.assertEqual(session._detector._local_track_templates, {})
        finally:
            release.set()
            session.close()

    def test_clear_templates_invalidates_generation_and_leaves_cache_empty(self) -> None:
        """A reset during tracking cannot leave old temporal patches behind."""
        session = DetectorSession("horn", project_root=ROOT)
        started = threading.Event()
        release = threading.Event()
        snapshots = [
            StateTrackSnapshot(track_id=1, x=200, y=200, scale=1.0),
            StateTrackSnapshot(track_id=2, x=300, y=200, scale=1.0),
        ]

        def fake_track(_frame, _mob_name, track, **_kwargs):
            started.set()
            release.wait(timeout=2.0)
            return LocalTrackResult(
                track_id=int(track["trackId"]),
                found=True,
                x=int(track["x"]),
                y=int(track["y"]),
                confidence=1.0,
                miss_reason="",
            )

        worker = threading.Thread(
            target=lambda: session.track_locals_frame(
                self.roi_frame, self.roi, snapshots,
            ),
        )
        try:
            with patch.object(session._detector, "ensure_descriptor"):
                with patch.object(session._detector, "track_local", side_effect=fake_track):
                    worker.start()
                    self.assertTrue(started.wait(timeout=1.0))
                    session.clear_track_templates()
                    release.set()
                    worker.join(timeout=2.0)
            self.assertFalse(worker.is_alive())
            self.assertEqual(session._detector._local_track_templates, {})
        finally:
            release.set()
            if worker.is_alive():
                worker.join(timeout=2.0)
            session.close()

    def test_track_locals_returns_results(self) -> None:
        scan = self.detector.discover_frame(self.roi_frame, self.roi)
        anchor = next(d for d in scan.detections if d.living)
        snapshots = [
            StateTrackSnapshot(
                track_id=1,
                x=anchor.x,
                y=anchor.y,
                scale=anchor.candidate_scale,
            )
        ]
        batch = self.detector.track_locals_frame(self.roi_frame, self.roi, snapshots)
        self.assertTrue(batch.ok)
        self.assertEqual(len(batch.results), 1)
        self.assertTrue(batch.results[0].found)
        self.assertGreater(batch.duration_ms, 0)
        # Stage timing splits into session-lock wait and local-follow compute.
        self.assertEqual(batch.duration_ms, batch.lock_wait_ms + batch.compute_ms)
        self.assertGreater(batch.compute_ms, 0)
        self.assertLess(batch.lock_wait_ms, 50)


if __name__ == "__main__":
    unittest.main()
