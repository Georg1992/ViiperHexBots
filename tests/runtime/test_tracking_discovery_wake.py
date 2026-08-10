"""Tracking wakes discovery on local miss; death removal is death-worker-owned."""

from __future__ import annotations

import threading
import unittest
from types import SimpleNamespace
from unittest.mock import ANY, MagicMock

from pybot.recognition.detector.detector import load_detector_config
from pybot.runtime.hunt_tracks import HuntTracks
from pybot.runtime.workers.attack_loop import AttackLoop
from pybot.runtime.workers.coord_tracking_worker import CoordTrackingWorker


class TrackingDiscoveryWakeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tracks = HuntTracks(load_detector_config())
        self.ctx = MagicMock()
        self.ctx.tracks = self.tracks
        self.ctx.discovery_suspend = threading.Event()
        self.ctx.discovery_wake = threading.Event()
        self.ctx.attack_wake = threading.Event()
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

    def test_created_track_wakes_attack_after_template_commit(self) -> None:
        """Tracking signals attack only after the live track is fully committed."""
        from pybot.recognition.rules import DiscoveryDetection

        self.ctx.config.mob_name = "horn"
        self.ctx.config.use_sprite_grf = True
        self.tracks.process_discovery_scan(
            [DiscoveryDetection(
                x=100, y=100, confidence=0.8,
                candidate_scale=0.9, living=True,
            )],
            mob_name="horn",
            now_tick=1,
        )
        self.ctx.tracker.track_locals_frame.return_value = SimpleNamespace(
            ok=True,
            results=[SimpleNamespace(
                track_id=-1, found=True, x=101, y=102,
                confidence=0.9,
            )],
        )
        self.ctx.tracker.transfer_track_template.return_value = True

        self.worker._tick()

        self.assertTrue(self.ctx.attack_wake.is_set())
        created = self.tracks.snapshot_alive(2)
        self.assertEqual(len(created), 1)
        self.assertEqual((created[0].x, created[0].y), (101, 102))
        self.ctx.tracker.transfer_track_template.assert_called_once()

    def test_created_track_is_followed_in_same_tick(self) -> None:
        """The first committed track is followed without a second tick."""
        from pybot.recognition.rules import DiscoveryDetection

        self.ctx.config.mob_name = "horn"
        self.ctx.config.use_sprite_grf = True
        self.tracks.process_discovery_scan(
            [DiscoveryDetection(
                x=100, y=100, confidence=0.8,
                candidate_scale=0.9, living=True,
            )],
            mob_name="horn",
            now_tick=1,
        )
        self.ctx.tracker.transfer_track_template.return_value = True
        calls: list[int] = []

        def follow(_frame, _roi, snapshots, *, on_result=None):
            for snapshot in snapshots:
                calls.append(snapshot.track_id)
                result = SimpleNamespace(
                    track_id=snapshot.track_id,
                    found=True,
                    x=101 if snapshot.track_id < 0 else 106,
                    y=102 if snapshot.track_id < 0 else 107,
                    confidence=0.9,
                    opacity_score=0.0,
                )
                if on_result is not None:
                    on_result(result)
            return SimpleNamespace(ok=True, results=[
                SimpleNamespace(
                    track_id=snapshot.track_id,
                    found=True,
                    x=101 if snapshot.track_id < 0 else 106,
                    y=102 if snapshot.track_id < 0 else 107,
                    confidence=0.9,
                    opacity_score=0.0,
                )
                for snapshot in snapshots
            ])

        self.ctx.tracker.track_locals_frame.side_effect = follow
        self.worker._tick()

        self.assertEqual(calls, [-1, 1])
        created = self.tracks.snapshot_alive(2)
        self.assertEqual(len(created), 1)
        self.assertEqual((created[0].x, created[0].y), (106, 107))
        self.assertTrue(self.ctx.attack_wake.is_set())

    def test_attack_wake_interrupts_idle_attack_poll(self) -> None:
        """A producer wake interrupts the idle wait immediately."""
        ctx = MagicMock()
        ctx.stop_event = threading.Event()
        ctx.attack_wake = threading.Event()
        ctx.attack_wake.set()
        ctx.should_run_combat.return_value = True
        ctx.in_post_teleport_heal_window.return_value = False
        ctx.tracks.tracks_for_policy.return_value = []
        ctx.policy.select_target.return_value = 0
        ctx.hunt_mode = MagicMock()
        ctx.hunt_mode.on_no_attackable_targets.return_value = False
        attack = AttackLoop(ctx, ctx.hunt_mode, MagicMock())

        self.assertFalse(attack.process_pending())
        self.assertFalse(ctx.attack_wake.is_set())

    def test_attack_wake_rechecks_store_before_area_clear(self) -> None:
        """A track committed during the empty read is attacked immediately."""
        ctx = MagicMock()
        ctx.stop_event = threading.Event()
        ctx.attack_wake = threading.Event()
        ctx.attack_wake.set()
        ctx.should_run_combat.return_value = True
        ctx.in_post_teleport_heal_window.return_value = False
        ctx.tracks.tracks_for_policy.side_effect = [
            [],
            [SimpleNamespace(id=7, area_epoch=0)],
        ]
        ctx.policy.select_target.side_effect = [0, 7]
        ctx.hunt_mode = MagicMock()
        ctx.hunt_mode.on_no_attackable_targets.return_value = False
        attack = AttackLoop(ctx, ctx.hunt_mode, MagicMock())
        attack._attack_one = MagicMock()  # type: ignore[method-assign]

        self.assertTrue(attack.process_pending())
        attack._attack_one.assert_called_once_with(
            7, ANY, expected_epoch=0,
        )
        ctx.hunt_mode.on_no_attackable_targets.assert_not_called()

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
