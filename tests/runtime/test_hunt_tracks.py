"""Tests for thread-safe HuntTracks"""

from __future__ import annotations

import threading
import unittest
from types import SimpleNamespace

from pybot.recognition.detector.detector import load_detector_config
from pybot.recognition.rules import DiscoveryDetection
from pybot.runtime.constants import HUNT_TRACK_LOST_LIMIT
from pybot.runtime.hunt_policy import HuntPolicy
from pybot.runtime.hunt_tracks import HuntTracks


def _hit(track_id: int, x: int, y: int, confidence: float = 0.8) -> SimpleNamespace:
    return SimpleNamespace(track_id=track_id, found=True, x=x, y=y, confidence=confidence)


def _miss(track_id: int) -> SimpleNamespace:
    return SimpleNamespace(track_id=track_id, found=False, x=0, y=0, confidence=0.0)


def _dead(track_id: int, x: int = 0, y: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        track_id=track_id,
        found=False,
        x=x,
        y=y,
        confidence=0.8,
        dead=True,
        opacity_baseline=0.6,
        opacity_baseline_samples=4,
        opacity_decay_streak=0,
    )


def det(x: int, y: int, confidence: float = 0.71, scale: float = 0.9) -> DiscoveryDetection:
    return DiscoveryDetection(
        x=x,
        y=y,
        confidence=confidence,
        candidate_scale=scale,
        living=True,
    )


class HuntTracksRulesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tracks = HuntTracks(load_detector_config())
        self.policy = HuntPolicy()
        self.now = 1_000_000

    def _create(self, x: int, y: int) -> int:
        summary = self.tracks.reconcile_detections(
            [det(x, y)],
            mob_name="horn",
            now_tick=self.now,
        )
        self.assertEqual(summary.added_count, 1)
        created = summary.created_ids or []
        self.assertEqual(len(created), 1)
        return created[0]

    def test_newly_discovered_track_is_alive(self) -> None:
        track_id = self._create(874, 578)
        track = self.tracks.get_track_by_id(track_id)
        assert track is not None
        self.assertEqual(track.state, "alive")

    def test_select_target_rotates_three_targets(self) -> None:
        for x, y in ((874, 578), (900, 610), (820, 520)):
            self.tracks.create_track("horn", x, y, 0.65, 0.9, now_tick=self.now)
        tracks = self.tracks.tracks_for_policy(self.now)
        self.assertEqual(self.policy.select_target(tracks, self.now), 1)
        self.policy.note_attack_target(1)
        self.assertEqual(self.policy.select_target(tracks, self.now), 2)
        self.policy.note_attack_target(2)
        self.assertEqual(self.policy.select_target(tracks, self.now), 3)
        self.policy.note_attack_target(3)
        self.assertEqual(self.policy.select_target(tracks, self.now), 1)

    def test_discovery_dedups_existing_track_without_moving_it(self) -> None:
        # Discovery is create-only: a detection near an existing track is
        # recognised as the same object (no duplicate) but does NOT move it —
        # tracking owns position.
        track_id = self._create(874, 578)
        matched = self.tracks.reconcile_detections(
            [det(900, 610, 0.71, 0.9)],
            mob_name="horn",
            now_tick=self.now + 500,
        )
        self.assertEqual(matched.added_count, 0)
        self.assertEqual(matched.matched_count, 1)
        self.assertEqual(self.tracks.get_track_count(), 1)
        track = self.tracks.get_track_by_id(track_id)
        assert track is not None
        self.assertEqual(track.x, 874)
        self.assertEqual(track.y, 578)

    def test_tracking_refreshes_coords(self) -> None:
        track_id = self._create(874, 578)
        self.tracks.apply_tracking([_hit(track_id, 900, 610)], now_tick=self.now + 50)
        track = self.tracks.get_track_by_id(track_id)
        assert track is not None
        self.assertEqual(track.x, 900)
        self.assertEqual(track.y, 610)

    def test_round_robin_includes_stale_coords(self) -> None:
        first = self.tracks.create_track("horn", 874, 578, 0.65, 0.9, now_tick=self.now)
        stale = self.tracks.create_track("horn", 900, 610, 0.65, 0.9, now_tick=self.now)
        stale_track = self.tracks.get_track_by_id(stale.id)
        assert stale_track is not None
        stale_track.updated_tick = self.now - 60_000
        tracks = self.tracks.tracks_for_policy(self.now)
        self.assertEqual(self.policy.select_target(tracks, self.now), first.id)
        self.policy.note_attack_target(first.id)
        self.assertEqual(self.policy.select_target(tracks, self.now), stale.id)

    def test_tracking_miss_removes_track(self) -> None:
        track_id = self._create(874, 578)
        lost_ids: list[int] = []
        for i in range(HUNT_TRACK_LOST_LIMIT):
            _, lost_ids = self.tracks.apply_tracking([_miss(track_id)], now_tick=self.now + i)
        self.assertIn(track_id, lost_ids)
        self.assertIsNone(self.tracks.get_track_by_id(track_id))

    def test_tracking_death_removes_track_immediately(self) -> None:
        track_id = self._create(874, 578)
        dead_ids, lost_ids = self.tracks.apply_tracking(
            [_dead(track_id, 874, 578)],
            now_tick=self.now + 1,
        )
        self.assertEqual(dead_ids, [track_id])
        self.assertEqual(lost_ids, [])
        self.assertIsNone(self.tracks.get_track_by_id(track_id))

    def test_death_site_blocks_discovery_rediscovery(self) -> None:
        track_id = self._create(874, 578)
        self.tracks.apply_tracking([_dead(track_id, 874, 578)], now_tick=self.now + 1)
        summary = self.tracks.reconcile_detections(
            [det(874, 578, 0.75, 0.9)],
            mob_name="horn",
            now_tick=self.now + 100,
        )
        self.assertEqual(summary.added_count, 0)
        self.assertEqual(summary.matched_count, 1)
        self.assertEqual(self.tracks.get_track_count(), 0)

    def test_death_site_expires_after_cooldown(self) -> None:
        config = {**load_detector_config(), "deathRediscoveryCooldownMs": 1000}
        tracks = HuntTracks(config)
        track_id = tracks.create_track("horn", 874, 578, 0.65, 0.9, now_tick=self.now).id
        tracks.apply_tracking([_dead(track_id, 874, 578)], now_tick=self.now + 1)
        blocked = tracks.reconcile_detections(
            [det(874, 578)],
            mob_name="horn",
            now_tick=self.now + 500,
        )
        self.assertEqual(blocked.added_count, 0)
        allowed = tracks.reconcile_detections(
            [det(874, 578)],
            mob_name="horn",
            now_tick=self.now + 2000,
        )
        self.assertEqual(allowed.added_count, 1)

    def test_tracking_hit_resets_miss_streak(self) -> None:
        track_id = self._create(874, 578)
        for i in range(HUNT_TRACK_LOST_LIMIT - 1):
            self.tracks.apply_tracking([_miss(track_id)], now_tick=self.now + i)
        # A hit clears the streak, so the track survives further misses.
        self.tracks.apply_tracking([_hit(track_id, 880, 580)], now_tick=self.now + 100)
        self.tracks.apply_tracking([_miss(track_id)], now_tick=self.now + 101)
        self.assertIsNotNone(self.tracks.get_track_by_id(track_id))

    def test_area_reset_clears_tracks(self) -> None:
        self._create(874, 578)
        self.tracks.area_reset()
        self.assertEqual(self.tracks.get_track_count(), 0)
        self.assertEqual(self.tracks.area_epoch, 1)

    def test_thread_safe_concurrent_reads(self) -> None:
        self._create(874, 578)
        errors: list[str] = []

        def reader() -> None:
            try:
                for _ in range(50):
                    self.tracks.snapshot_alive(self.now)
                    self.tracks.tracks_for_policy(self.now)
            except Exception as exc:  # pragma: no cover
                errors.append(str(exc))

        threads = [threading.Thread(target=reader) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
