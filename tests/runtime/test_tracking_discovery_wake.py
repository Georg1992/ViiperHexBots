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
        self.ctx.tracking_wake = threading.Event()
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


    def test_tracking_worker_consumes_discovery_wake_before_next_cadence(self) -> None:
        """A discovery wake is consumed immediately between tracking ticks."""
        ctx = MagicMock()
        ctx.stop_event = threading.Event()
        ctx.should_run_tracking.return_value = True
        ctx.tracking_wake = MagicMock()
        ctx.tracking_wake.is_set.return_value = True
        ctx.tracking_wake.wait.return_value = False
        worker = CoordTrackingWorker(ctx)
        calls = {"count": 0}

        def tick() -> None:
            calls["count"] += 1
            if calls["count"] == 2:
                ctx.stop_event.set()

        worker._tick = MagicMock(side_effect=tick)

        worker.run()

        self.assertEqual(calls["count"], 2)
        ctx.tracking_wake.wait.assert_called_once()
        ctx.tracking_wake.clear.assert_called_once_with()

    def test_discovery_candidate_wakes_tracking_immediately(self) -> None:
        """A positive discovery scan wakes the coordinator before its cadence."""
        from pybot.recognition.rules import DiscoveryDetection
        from pybot.runtime.workers.discovery_worker import DiscoveryWorker

        ctx = MagicMock()
        ctx.stop_event = threading.Event()
        ctx.config.discovery_interval_ms = 250
        ctx.config.mob_name = "horn"
        ctx.config.use_sprite_grf = True
        ctx.should_run_discovery.return_value = True
        ctx.discovery_wake = threading.Event()
        ctx.tracking_wake = threading.Event()
        ctx.area_transition_lock = threading.RLock()
        ctx.observation_publication_lock = threading.Lock()
        ctx.hunt_generation = 0
        ctx.capture.is_valid.return_value = True
        ctx.capture.get_hunt_roi.return_value = SimpleNamespace(
            x=0, y=0, w=200, h=200,
            center_x=100, center_y=100,
        )
        ctx.capture.capture_roi.return_value = MagicMock(size=1)
        ctx.tracks = self.tracks
        ctx.detector.discover_frame.return_value = SimpleNamespace(
            ok=True,
            fail_reason="",
            raw_count=1,
            accepted_count=1,
            duration_ms=1,
            timing={},
            detections=[SimpleNamespace(
                x=100,
                y=100,
                confidence=0.9,
                candidate_scale=0.9,
                bbox=(90, 90, 20, 20),
                living=True,
            )],
        )
        ctx.overlay = MagicMock()
        ctx.validation = MagicMock()
        ctx.mark_startup_area_clear = MagicMock()
        hunt_mode = MagicMock()
        worker = DiscoveryWorker(ctx, hunt_mode)

        worker._scan()

        self.assertTrue(ctx.tracking_wake.is_set())
        self.assertTrue(self.tracks.has_pending_discovery_candidates())
        hunt_mode.note_discovery_scan_completed.assert_called_once()

    def test_discovery_accepts_legacy_detector_signature(self) -> None:
        """Old two-argument detector doubles still complete a discovery pass."""
        from pybot.runtime.workers.discovery_worker import DiscoveryWorker

        self.ctx.stop_event = threading.Event()
        self.ctx.config.discovery_interval_ms = 250
        self.ctx.config.mob_name = "horn"
        self.ctx.config.use_sprite_grf = False
        self.ctx.should_run_discovery.return_value = True
        self.ctx.discovery_wake = threading.Event()
        self.ctx.area_transition_lock = threading.RLock()
        self.ctx.observation_publication_lock = threading.Lock()
        self.ctx.hunt_generation = 0
        self.ctx.capture.get_hunt_roi.return_value = SimpleNamespace(
            x=0, y=0, w=200, h=200, center_x=100, center_y=100,
        )
        self.ctx.capture.capture_roi.return_value = MagicMock(size=1)
        self.ctx.detector.detector_config.return_value = {}
        self.ctx.detector.discover_frame.side_effect = (
            lambda _frame, _roi: SimpleNamespace(
                ok=True,
                fail_reason="",
                raw_count=0,
                accepted_count=0,
                duration_ms=1,
                timing={},
                detections=[],
            )
        )
        self.ctx.overlay = MagicMock()
        self.ctx.validation = MagicMock()
        self.ctx.mark_startup_area_clear = MagicMock()
        worker = DiscoveryWorker(self.ctx, MagicMock())
        worker._scan()
        self.assertEqual(self.tracks.get_track_count(), 0)
        self.assertTrue(self.ctx.detector.discover_frame.called)

    def test_created_track_wakes_attack_after_state_commit(self) -> None:
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
        self.ctx.tracker.transfer_track_state.return_value = True

        self.worker._tick()

        self.assertTrue(self.ctx.attack_wake.is_set())
        created = self.tracks.snapshot_alive(2)
        self.assertEqual(len(created), 1)
        self.assertEqual((created[0].x, created[0].y), (101, 102))
        self.ctx.tracker.transfer_track_state.assert_called_once()

    def test_nearby_distinct_candidates_each_create_a_track(self) -> None:
        """Candidate dedup keeps the discovery cluster boundary.

        The wider existing-track dedup radius is for matching a new scan to a
        known mob; it must not merge two distinct blobs that Discovery already
        separated in the same scan.
        """
        from pybot.recognition.rules import DiscoveryDetection

        self.ctx.config.mob_name = "horn"
        self.ctx.config.use_sprite_grf = True
        self.tracks.process_discovery_scan(
            [
                DiscoveryDetection(
                    x=100, y=100, confidence=0.8,
                    candidate_scale=0.9, living=True,
                ),
                DiscoveryDetection(
                    x=160, y=100, confidence=0.79,
                    candidate_scale=0.9, living=True,
                ),
            ],
            mob_name="horn",
            now_tick=1,
        )
        self.ctx.tracker.transfer_track_state.return_value = True

        def acquire(_frame, _roi, snapshots, *, on_result=None):
            results = []
            for snapshot in snapshots:
                result = SimpleNamespace(
                    track_id=snapshot.track_id,
                    found=True,
                    x=snapshot.x,
                    y=snapshot.y,
                    confidence=0.9,
                    opacity_score=0.0,
                )
                results.append(result)
                if on_result is not None:
                    on_result(result)
            return SimpleNamespace(ok=True, results=results)

        self.ctx.tracker.track_locals_frame.side_effect = acquire
        self.worker._tick()

        created = self.tracks.snapshot_alive(2)
        self.assertEqual(len(created), 2)
        self.assertEqual(
            {(track.x, track.y) for track in created},
            {(100, 100), (160, 100)},
        )

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
        self.ctx.tracker.transfer_track_state.return_value = True
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

    def test_all_stale_tracks_are_excluded_but_remain_alive(self) -> None:
        """Stale combat input never deletes Tracks needed for local recovery."""
        ctx = MagicMock()
        ctx.stop_event = threading.Event()
        ctx.attack_wake = threading.Event()
        ctx.should_run_combat.return_value = True
        ctx.in_post_teleport_heal_window.return_value = False
        now = 10_000
        stale = SimpleNamespace(id=1, last_found_tick=now - 1_000, area_epoch=0)
        ctx.tracks.tracks_for_policy.return_value = [stale]
        ctx.policy.select_target.return_value = 0
        ctx.hunt_mode = MagicMock()
        attack = AttackLoop(ctx, ctx.hunt_mode, MagicMock())

        with unittest.mock.patch(
            "pybot.runtime.workers.attack_loop.monotonic_ms", return_value=now,
        ):
            self.assertFalse(attack.process_pending())

        ctx.policy.select_target.assert_called_with([], now)
        ctx.hunt_mode.on_no_attackable_targets.assert_called_once_with()

    def test_stale_track_is_not_selected_for_attack(self) -> None:
        """Held coordinates do not monopolize attack selection."""
        ctx = MagicMock()
        ctx.stop_event = threading.Event()
        ctx.attack_wake = threading.Event()
        ctx.should_run_combat.return_value = True
        ctx.in_post_teleport_heal_window.return_value = False
        now = 10_000
        stale = SimpleNamespace(id=1, last_found_tick=now - 1_000, area_epoch=0)
        fresh = SimpleNamespace(id=2, last_found_tick=now, area_epoch=0)
        ctx.tracks.tracks_for_policy.return_value = [stale, fresh]
        ctx.policy.select_target.return_value = 2
        ctx.hunt_mode = MagicMock()
        attack = AttackLoop(ctx, ctx.hunt_mode, MagicMock())

        with unittest.mock.patch(
            "pybot.runtime.workers.attack_loop.monotonic_ms", return_value=now,
        ):
            attack._attack_one = MagicMock()  # type: ignore[method-assign]
            self.assertTrue(attack.process_pending())

        ctx.policy.select_target.assert_called_once_with([fresh], now)
        attack._attack_one.assert_called_once_with(2, now, expected_epoch=0)
        ctx.hunt_mode.on_no_attackable_targets.assert_not_called()

    def test_danger_wake_interrupts_gameplay_idle_wait(self) -> None:
        """A newly observed critical hit interrupts the idle wait immediately."""
        ctx = MagicMock()
        ctx.stop_event = threading.Event()
        ctx.danger_wake = threading.Event()
        ctx.attack_wake = threading.Event()
        attack = AttackLoop(ctx, MagicMock(), MagicMock())
        wake = threading.Timer(0.01, ctx.danger_wake.set)
        wake.start()
        try:
            attack._wait_for_gameplay_delay(1.0)
        finally:
            wake.join(timeout=1.0)

        self.assertFalse(ctx.danger_wake.is_set())
        self.assertFalse(ctx.stop_event.is_set())

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

    def test_tracking_snapshot_contains_prediction_inputs(self) -> None:
        """The hot path receives center, scale, and bounded motion prediction."""
        track = self.tracks.create_track(
            "horn", 100, 100, 0.8, 0.9, now_tick=1
        )
        self.ctx.tracker.track_locals_frame.return_value = SimpleNamespace(
            results=[SimpleNamespace(
                track_id=track.id,
                found=True,
                x=100,
                y=100,
                confidence=0.9,
                opacity_score=0.0,
            )]
        )

        self.worker._tick()

        snapshots = self.ctx.tracker.track_locals_frame.call_args.args[2]
        self.assertEqual(len(snapshots), 1)
        snapshot = snapshots[0]
        self.assertEqual((snapshot.x, snapshot.y), (100, 100))
        self.assertEqual(snapshot.scale, 0.9)
        self.assertEqual(snapshot.vel_x, 0.0)
        self.assertEqual(snapshot.vel_y, 0.0)
        self.assertTrue(snapshot.prediction_valid)
        self.assertEqual(snapshot.lost_count, 0)

    def test_fast_track_keeps_prediction_through_first_local_miss(self) -> None:
        """A single weak frame must not turn off the runner's motion lead."""
        track = self.tracks.create_track(
            "horn", 100, 100, 0.8, 0.9, now_tick=1
        )
        track.vel_x = 42.0
        track.vel_y = -6.0
        track.lost_count = 1
        self.ctx.tracker.track_locals_frame.return_value = SimpleNamespace(
            results=[SimpleNamespace(
                track_id=track.id,
                found=False,
                x=track.x,
                y=track.y,
                confidence=0.0,
                tracking_lost=False,
            )]
        )

        self.worker._tick()

        snapshot = self.ctx.tracker.track_locals_frame.call_args.args[2][0]
        self.assertTrue(snapshot.prediction_valid)
        self.assertEqual(snapshot.lost_count, 1)
        self.assertEqual(snapshot.vel_x, 42.0)
        self.assertEqual(snapshot.vel_y, -6.0)

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
