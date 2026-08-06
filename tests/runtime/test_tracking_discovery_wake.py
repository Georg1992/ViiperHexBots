"""Tracking wakes discovery on local miss; death removal is death-worker-owned."""

from __future__ import annotations

import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from pybot.recognition.detector.detector import load_detector_config
from pybot.runtime.hunt_tracks import HuntTracks
from pybot.runtime.workers.coord_tracking_worker import CoordTrackingWorker


class TrackingDiscoveryWakeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tracks = HuntTracks(load_detector_config())
        self.ctx = MagicMock()
        self.ctx.tracks = self.tracks
        self.ctx.discovery_suspend = threading.Event()
        self.ctx.discovery_wake = threading.Event()
        self.ctx.capture.is_valid.return_value = True
        self.ctx.capture.get_hunt_roi.return_value = MagicMock(
            x=0, y=0, w=200, h=200
        )
        self.ctx.capture.capture_roi.return_value = MagicMock(size=1)
        self.ctx.tracker.track_locals_frame.return_value = SimpleNamespace(
            results=[]
        )
        self.worker = CoordTrackingWorker(self.ctx)

    def test_discovery_timeout_does_not_consume_post_teleport_wake(self) -> None:
        """A wake arriving at the cadence boundary must survive for the next scan."""
        from pybot.runtime.workers.discovery_worker import DiscoveryWorker

        ctx = MagicMock()
        ctx.stop_event = threading.Event()
        ctx.config.discovery_interval_ms = 1
        ctx.should_run_discovery.return_value = True
        ctx.discovery_wake = threading.Event()
        # Simulate a teleport setting the wake while the cadence wait reports
        # timeout. The worker must not clear that signal before scanning.
        ctx.discovery_wake.set()
        worker = DiscoveryWorker(ctx, MagicMock())
        worker._wait_for_discovery_wake = MagicMock(return_value=False)
        worker._scan = lambda: ctx.stop_event.set()

        worker.run()

        self.assertTrue(ctx.discovery_wake.is_set())

    def test_coord_worker_treats_dead_flag_as_miss(self) -> None:
        """Coord worker no longer special-cases dead=True — it's just a miss."""
        track = self.tracks.create_track(
            "horn", 100, 100, 0.8, 0.9, now_tick=1
        )
        self.ctx.tracker.track_locals_frame.return_value = SimpleNamespace(
            results=[
                SimpleNamespace(
                    track_id=track.id,
                    found=False,
                    x=100,
                    y=100,
                    confidence=0.8,
                    dead=True,
                    opacity_baseline=0.6,
                    opacity_baseline_samples=4,
                    opacity_decay_streak=0,
                )
            ]
        )
        self.worker._tick()
        # Coord worker only tracks — dead=True is just a miss, so discovery wakes.
        self.assertTrue(self.ctx.discovery_wake.is_set())
        # Coord worker never removes tracks; death worker owns removal.
        self.assertIsNotNone(self.tracks.get_track_by_id(track.id))

    def test_critical_escape_gate_does_not_ingest_old_candidates(self) -> None:
        """A candidate queued before danger TP must not become a stale track."""
        from pybot.recognition.rules import DiscoveryDetection

        self.ctx.config.mob_name = "horn"
        self.ctx.config.use_sprite_grf = True
        self.ctx.critical_danger_requested = threading.Event()
        self.ctx.should_run_tracking.side_effect = (
            lambda: not self.ctx.critical_danger_requested.is_set()
        )
        self.tracks.process_discovery_scan(
            [
                DiscoveryDetection(
                    x=100,
                    y=100,
                    confidence=0.8,
                    candidate_scale=0.9,
                    living=True,
                )
            ],
            mob_name="horn",
            now_tick=1,
        )
        self.assertTrue(self.tracks.has_pending_discovery_candidates())

        self.ctx.critical_danger_requested.set()
        self.worker._tick()

        self.assertEqual(self.tracks.get_track_count(), 0)
        self.assertTrue(self.tracks.has_pending_discovery_candidates())

    def test_local_miss_wakes_discovery_and_keeps_track(self) -> None:
        track = self.tracks.create_track(
            "horn", 100, 100, 0.8, 0.9, now_tick=1
        )
        self.ctx.tracker.track_locals_frame.return_value = SimpleNamespace(
            results=[
                SimpleNamespace(
                    track_id=track.id,
                    found=False,
                    x=0,
                    y=0,
                    confidence=0.0,
                    dead=False,
                    opacity_baseline=0.0,
                    opacity_baseline_samples=0,
                    opacity_decay_streak=0,
                )
            ]
        )
        self.worker._tick()
        self.assertTrue(self.ctx.discovery_wake.is_set())
        kept = self.tracks.get_track_by_id(track.id)
        assert kept is not None
        self.assertEqual(kept.lost_count, 1)


if __name__ == "__main__":
    unittest.main()
