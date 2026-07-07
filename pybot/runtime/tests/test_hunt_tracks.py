"""Tests for thread-safe HuntTracks"""

from __future__ import annotations

import threading
import unittest

from pybot.runtime.hunt_policy import HuntPolicy
from pybot.runtime.hunt_tracks import HuntTracks
from pybot.runtime._mob_rec_path import import_hunt_track_rules
_hunt = import_hunt_track_rules()

HUNT_ATTACK_RESULT_WINDOW_MS = _hunt.HUNT_ATTACK_RESULT_WINDOW_MS
HUNT_LOCAL_TRACK_MISS_LIMIT = _hunt.HUNT_LOCAL_TRACK_MISS_LIMIT
HUNT_UNREACHABLE_CONFIRM_STREAK = _hunt.HUNT_UNREACHABLE_CONFIRM_STREAK
DiscoveryDetection = _hunt.DiscoveryDetection
LocalTrackObservation = _hunt.LocalTrackObservation
StateObservation = _hunt.StateObservation
is_attackable = _hunt.is_attackable
is_pending = _hunt.is_pending


def det(x: int, y: int, confidence: float = 0.71, scale: float = 0.9) -> DiscoveryDetection:
    return DiscoveryDetection(
        x=x,
        y=y,
        confidence=confidence,
        candidate_scale=scale,
        living=True,
        dead=False,
    )


class HuntTracksRulesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tracks = HuntTracks()
        self.policy = HuntPolicy()
        self.now = 1_000_000

    def _create(self, x: int, y: int, track_id_hint: int | None = None) -> int:
        summary = self.tracks.reconcile_detections(
            [det(x, y)],
            mob_name="horn",
            now_tick=self.now,
        )
        self.assertEqual(summary.added_count, 1)
        created = summary.created_ids or []
        self.assertEqual(len(created), 1)
        return created[0]

    def test_newly_discovered_track_is_attackable(self) -> None:
        track_id = self._create(874, 578)
        track = self.tracks.get_track_by_id(track_id)
        assert track is not None
        self.assertTrue(is_attackable(track, self.now))

    def test_ignored_unreachable_does_not_block_first_attack(self) -> None:
        track_id = self._create(874, 578)
        self.tracks.apply_state_observations(
            [StateObservation(track_id, "unreachable")],
            now_tick=self.now + 7_000,
        )
        track = self.tracks.get_track_by_id(track_id)
        assert track is not None
        self.assertEqual(track.state_unreachable_count, 1)
        self.assertEqual(track.last_state_tick, 0)
        self.assertEqual(track.state, "alive")
        self.assertTrue(is_attackable(track, self.now + 7_000))

    def test_stale_coords_still_attackable(self) -> None:
        track_id = self._create(874, 578)
        track = self.tracks.get_track_by_id(track_id)
        assert track is not None
        track.updated_tick = self.now - 60_000
        self.assertTrue(is_attackable(track, self.now))

    def test_local_track_miss_does_not_remove_track(self) -> None:
        track_id = self._create(874, 578)
        for _ in range(HUNT_LOCAL_TRACK_MISS_LIMIT):
            self.tracks.apply_local_track_observations(
                [LocalTrackObservation(track_id, False, miss_reason="no_peak")],
                now_tick=self.now,
            )
        track = self.tracks.get_track_by_id(track_id)
        assert track is not None
        self.assertEqual(track.local_track_miss_count, HUNT_LOCAL_TRACK_MISS_LIMIT)
        self.assertEqual(track.state, "alive")
        self.assertTrue(self.tracks.select_state_confirm_track_id(self.now) == track_id)

    def test_local_track_found_updates_coords(self) -> None:
        track_id = self._create(874, 578)
        self.tracks.apply_local_track_observations(
            [LocalTrackObservation(track_id, True, x=880, y=582, confidence=0.7)],
            now_tick=self.now + 100,
        )
        track = self.tracks.get_track_by_id(track_id)
        assert track is not None
        self.assertEqual(track.x, 880)
        self.assertEqual(track.y, 582)
        self.assertEqual(track.local_track_miss_count, 0)

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

    def test_select_target_skips_pending_in_rotation(self) -> None:
        ids = []
        for x, y in ((874, 578), (900, 610), (820, 520)):
            track = self.tracks.create_track("horn", x, y, 0.65, 0.9, now_tick=self.now)
            ids.append(track.id)
        self.tracks.apply_attack_event(ids[1], now_tick=self.now + 100)
        self.policy.note_attack_target(ids[0])
        tracks = self.tracks.tracks_for_policy(self.now + 200)
        self.assertEqual(self.policy.select_target(tracks, self.now + 200), ids[2])

    def test_select_target_picks_unattacked_track(self) -> None:
        track_id = self._create(874, 578)
        self.tracks.apply_state_observations(
            [StateObservation(track_id, "unreachable")],
            now_tick=self.now + 1_000,
        )
        tracks = self.tracks.tracks_for_policy(self.now + 1_000)
        self.assertEqual(self.policy.select_target(tracks, self.now + 1_000), track_id)

    def test_after_attack_pending_blocks(self) -> None:
        track_id = self._create(874, 578)
        self.tracks.apply_attack_event(track_id, now_tick=self.now + 100)
        track = self.tracks.get_track_by_id(track_id)
        assert track is not None
        self.assertTrue(is_pending(track, self.now + 200))
        self.assertFalse(is_attackable(track, self.now + 200))

    def test_all_alive_tracks_request_state(self) -> None:
        track_id = self._create(874, 578)
        reqs = self.tracks.collect_state_requests(now_tick=self.now)
        self.assertEqual(len(reqs), 1)
        self.assertEqual(reqs[0]["id"], track_id)
        self.tracks.apply_attack_event(track_id, now_tick=self.now + 100)
        reqs = self.tracks.collect_state_requests(now_tick=self.now + 200)
        self.assertEqual(len(reqs), 1)
        self.assertEqual(reqs[0]["id"], track_id)

    def test_discovery_match_refreshes_coords_before_attack(self) -> None:
        track_id = self._create(874, 578)
        matched = self.tracks.reconcile_detections(
            [det(900, 610, 0.71, 0.9)],
            mob_name="horn",
            now_tick=self.now + 500,
        )
        self.assertEqual(matched.matched_count, 1)
        track = self.tracks.get_track_by_id(track_id)
        assert track is not None
        self.assertEqual(track.x, 900)
        self.assertEqual(track.y, 610)
        self.assertEqual(track.attack_count, 0)

    def test_pending_timeout_clears_pending(self) -> None:
        track_id = self._create(874, 578)
        self.tracks.apply_attack_event(track_id, now_tick=self.now + 100)
        track = self.tracks.get_track_by_id(track_id)
        assert track is not None
        track.pending_result_until_tick = self.now
        self.assertFalse(is_pending(track, self.now + 1))
        self.assertEqual(track.state, "alive")
        self.assertTrue(track.pending_result_resolved)

    def test_attacked_unreachable_keeps_track(self) -> None:
        track_id = self._create(874, 578)
        self.tracks.apply_attack_event(track_id, now_tick=self.now + 100)
        track = self.tracks.get_track_by_id(track_id)
        assert track is not None
        track.pending_result_resolved = True
        track.state = "alive"
        # First "unreachable" after attack should keep track, mark unreachable
        self.tracks.apply_state_observations(
            [StateObservation(track_id, "unreachable")],
            now_tick=self.now + 200,
        )
        track = self.tracks.get_track_by_id(track_id)
        self.assertIsNotNone(track)
        self.assertEqual(track.state, "unreachable")

    def test_attacked_unreachable_marks_even_when_pending(self) -> None:
        """Attacked then unreachable marks it even during pending window."""
        track_id = self._create(874, 578)
        self.tracks.apply_attack_event(track_id, now_tick=self.now + 100)
        track = self.tracks.get_track_by_id(track_id)
        assert track is not None
        self.tracks.apply_state_observations(
            [StateObservation(track_id, "unreachable")],
            now_tick=self.now + 200,
        )
        track = self.tracks.get_track_by_id(track_id)
        self.assertIsNotNone(track)
        self.assertEqual(track.state, "unreachable")

    def test_attacked_unreachable_marks_after_timeout_too(self) -> None:
        """Even after pending timeout, attacked+unreachable marks unreachable."""
        track_id = self._create(874, 578)
        self.tracks.apply_attack_event(track_id, now_tick=self.now + 100)
        timeout_at = self.now + 100 + HUNT_ATTACK_RESULT_WINDOW_MS + 1
        self.tracks.apply_state_observations(
            [StateObservation(track_id, "unreachable")],
            now_tick=timeout_at,
        )
        track = self.tracks.get_track_by_id(track_id)
        self.assertIsNotNone(track)
        self.assertEqual(track.state, "unreachable")

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

    def test_discovery_miss_removes_track(self) -> None:
        track_id = self._create(874, 578)
        for _ in range(3):
            self.tracks.reconcile_detections([], mob_name="horn", now_tick=self.now + 500)
        self.assertIsNone(self.tracks.get_track_by_id(track_id))

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
                    self.tracks.snapshot_attackable(self.now)
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
