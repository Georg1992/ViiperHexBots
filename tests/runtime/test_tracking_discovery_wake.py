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
        self.tracks = HuntTracks(load_detector_config(), skill_delay_ms=5000)
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
