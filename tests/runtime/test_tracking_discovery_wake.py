"""Tracking wakes discovery for lost/unreachable, not for confirmed deaths."""

from __future__ import annotations

import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from pybot.recognition.detector.detector import load_detector_config
from pybot.runtime.hunt_tracks import HuntTracks
from pybot.runtime.workers.tracking_worker import TrackingWorker


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
        self.worker = TrackingWorker(self.ctx)

    def test_death_does_not_wake_discovery(self) -> None:
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
        self.assertFalse(self.ctx.discovery_wake.is_set())
        self.assertIsNone(self.tracks.get_track_by_id(track.id))
        # Death site still blocks rediscovery.
        from pybot.recognition.rules import DiscoveryDetection

        summary = self.tracks.reconcile_detections(
            [
                DiscoveryDetection(
                    x=100, y=100, confidence=0.8, candidate_scale=0.9, living=True
                )
            ],
            mob_name="horn",
            now_tick=50,
        )
        self.assertEqual(summary.added_count, 0)

    def test_lost_wakes_discovery(self) -> None:
        track = self.tracks.create_track(
            "horn", 100, 100, 0.8, 0.9, now_tick=1
        )
        miss_limit = int(load_detector_config()["trackLostMissLimit"])
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
        for _ in range(miss_limit):
            self.ctx.discovery_wake.clear()
            self.worker._tick()
        self.assertTrue(self.ctx.discovery_wake.is_set())
        self.assertIsNone(self.tracks.get_track_by_id(track.id))


if __name__ == "__main__":
    unittest.main()
