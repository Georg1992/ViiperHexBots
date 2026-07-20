"""Tests for thread-safe HuntTracks"""

from __future__ import annotations

import threading
import unittest
from types import SimpleNamespace

from pybot.recognition.detector.detector import load_detector_config
from pybot.recognition.rules import DiscoveryDetection
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
        self.tracks = HuntTracks(load_detector_config(), skill_delay_ms=5000)
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
        # A detection near an existing track is recognised as the same object
        # (no duplicate) but does NOT move it — tracking owns position.
        track_id = self._create(874, 578)
        matched = self.tracks.reconcile_detections(
            [det(900, 610, 0.71, 0.9)],
            mob_name="horn",
            now_tick=self.now + 500,
        )
        self.assertEqual(matched.added_count, 0)
        self.assertEqual(matched.matched_count, 1)
        self.assertEqual(matched.removed_count, 0)
        self.assertEqual(self.tracks.get_track_count(), 1)
        track = self.tracks.get_track_by_id(track_id)
        assert track is not None
        self.assertEqual(track.x, 874)
        self.assertEqual(track.y, 578)

    def test_discovery_removes_track_absent_from_scan(self) -> None:
        kept = self._create(874, 578)
        gone = self.tracks.create_track(
            "horn", 200, 200, 0.65, 0.9, now_tick=self.now
        ).id
        summary = self.tracks.reconcile_detections(
            [det(874, 578, 0.75, 0.9)],
            mob_name="horn",
            now_tick=self.now + 100,
        )
        self.assertEqual(summary.added_count, 0)
        self.assertEqual(summary.matched_count, 1)
        self.assertEqual(summary.removed_count, 1)
        self.assertEqual(summary.removed_ids, [gone])
        self.assertIsNotNone(self.tracks.get_track_by_id(kept))
        self.assertIsNone(self.tracks.get_track_by_id(gone))

    def test_discovery_empty_scan_clears_all_tracks(self) -> None:
        first = self._create(874, 578)
        second = self.tracks.create_track(
            "horn", 200, 200, 0.65, 0.9, now_tick=self.now
        ).id
        summary = self.tracks.reconcile_detections(
            [],
            mob_name="horn",
            now_tick=self.now + 100,
        )
        self.assertEqual(summary.removed_count, 2)
        self.assertEqual(set(summary.removed_ids or []), {first, second})
        self.assertEqual(self.tracks.get_track_count(), 0)

    def test_try_claim_clear_for_teleport_rejects_alive_tracks(self) -> None:
        self._create(874, 578)
        self.assertFalse(self.tracks.try_claim_clear_for_teleport())
        self.assertEqual(self.tracks.get_track_count(), 1)
        self.assertEqual(self.tracks.area_epoch, 0)

    def test_try_claim_clear_for_teleport_advances_epoch(self) -> None:
        self.assertTrue(self.tracks.try_claim_clear_for_teleport())
        self.assertEqual(self.tracks.area_epoch, 1)
        self.assertEqual(self.tracks.get_track_count(), 0)

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
        miss_limit = int(load_detector_config()["trackLostMissLimit"])
        lost_ids: list[int] = []
        for i in range(miss_limit):
            _, lost_ids, _ = self.tracks.apply_tracking([_miss(track_id)], now_tick=self.now + i)
        self.assertIn(track_id, lost_ids)
        self.assertIsNone(self.tracks.get_track_by_id(track_id))

    def test_tracking_death_removes_track_immediately(self) -> None:
        track_id = self._create(874, 578)
        dead_ids, lost_ids, unreachable_ids = self.tracks.apply_tracking(
            [_dead(track_id, 874, 578)],
            now_tick=self.now + 1,
        )
        self.assertEqual(dead_ids, [track_id])
        self.assertEqual(lost_ids, [])
        self.assertEqual(unreachable_ids, [])
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

    def test_attack_limit_removes_track_as_unreachable(self) -> None:
        tracks = HuntTracks(load_detector_config(), skill_delay_ms=1500)
        track_id = tracks.create_track("horn", 874, 578, 0.65, 0.9, now_tick=self.now).id
        for i in range(2):
            self.assertTrue(
                tracks.apply_attack_event(track_id, now_tick=self.now + i + 1)
            )
        self.assertFalse(tracks.apply_attack_event(track_id, now_tick=self.now + 3))
        _, _, unreachable_ids = tracks.apply_tracking([], now_tick=self.now + 3)
        self.assertEqual(unreachable_ids, [track_id])
        self.assertIsNone(tracks.get_track_by_id(track_id))

    def test_attack_limit_is_per_mob_not_shared(self) -> None:
        tracks = HuntTracks(load_detector_config(), skill_delay_ms=1500)
        first = tracks.create_track("horn", 874, 578, 0.65, 0.9, now_tick=self.now).id
        second = tracks.create_track("horn", 980, 640, 0.65, 0.9, now_tick=self.now).id
        for i in range(3):
            tracks.apply_attack_event(first, now_tick=self.now + i + 1)
        _, _, unreachable_ids = tracks.apply_tracking([], now_tick=self.now + 3)
        self.assertEqual(unreachable_ids, [first])
        self.assertIsNone(tracks.get_track_by_id(first))
        self.assertIsNotNone(tracks.get_track_by_id(second))
        second_track = tracks.get_track_by_id(second)
        assert second_track is not None
        self.assertEqual(second_track.attack_count, 0)

    def test_unreachable_site_blocks_discovery_rediscovery(self) -> None:
        tracks = HuntTracks(load_detector_config(), skill_delay_ms=3000)
        track_id = tracks.create_track("horn", 874, 578, 0.65, 0.9, now_tick=self.now).id
        tracks.apply_attack_event(track_id, now_tick=self.now + 1)
        tracks.apply_attack_event(track_id, now_tick=self.now + 2)
        tracks.apply_tracking([], now_tick=self.now + 2)
        summary = tracks.reconcile_detections(
            [det(874, 578, 0.75, 0.9)],
            mob_name="horn",
            now_tick=self.now + 100,
        )
        self.assertEqual(summary.added_count, 0)
        self.assertEqual(summary.matched_count, 1)
        self.assertEqual(tracks.get_track_count(), 0)

    def test_mob_attack_count_inherits_after_track_recreation(self) -> None:
        config = {
            **load_detector_config(),
            "deathRediscoveryCooldownMs": 1000,
        }
        tracks = HuntTracks(config, skill_delay_ms=5000)
        track_id = tracks.create_track("horn", 874, 578, 0.65, 0.9, now_tick=self.now).id
        for i in range(2):
            tracks.apply_attack_event(track_id, now_tick=self.now + i + 1)
        tracks.apply_tracking([], now_tick=self.now + 2)
        self.assertIsNone(tracks.get_track_by_id(track_id))
        summary = tracks.reconcile_detections(
            [det(874, 578, 0.75, 0.9)],
            mob_name="horn",
            now_tick=self.now + 500,
        )
        self.assertEqual(summary.added_count, 0)
        summary = tracks.reconcile_detections(
            [det(874, 578, 0.75, 0.9)],
            mob_name="horn",
            now_tick=self.now + 2000,
        )
        self.assertEqual(summary.added_count, 1)
        created = summary.created_ids or []
        self.assertEqual(len(created), 1)
        track = tracks.get_track_by_id(created[0])
        assert track is not None
        self.assertEqual(track.attack_count, 2)
        self.assertFalse(tracks.apply_attack_event(created[0], now_tick=self.now + 2001))

    def test_death_records_attacks_till_death_sample(self) -> None:
        tracks = HuntTracks(load_detector_config(), skill_delay_ms=500)
        track_id = tracks.reconcile_detections(
            [det(874, 578)],
            mob_name="horn",
            now_tick=self.now,
        ).created_ids[0]
        for i in range(4):
            tracks.apply_attack_event(track_id, now_tick=self.now + i + 1)
        tracks.apply_tracking([_dead(track_id, 874, 578)], now_tick=self.now + 10)
        self.assertEqual(tracks.kill_sample_count, 1)
        self.assertEqual(tracks.average_attacks_till_death, 4.0)
        self.assertEqual(tracks.max_attacks_per_mob_before_unreachable, 10)

    def test_kill_history_builds_rolling_average(self) -> None:
        tracks = HuntTracks(load_detector_config(), skill_delay_ms=500)
        first = tracks.create_track("horn", 874, 578, 0.65, 0.9, now_tick=self.now).id
        second = tracks.create_track("horn", 980, 640, 0.65, 0.9, now_tick=self.now).id
        for i in range(4):
            tracks.apply_attack_event(first, now_tick=self.now + i + 1)
        tracks.apply_tracking([_dead(first, 874, 578)], now_tick=self.now + 10)
        for i in range(2):
            tracks.apply_attack_event(second, now_tick=self.now + 20 + i)
        tracks.apply_tracking([_dead(second, 980, 640)], now_tick=self.now + 30)
        self.assertEqual(tracks.kill_sample_count, 2)
        self.assertEqual(tracks.average_attacks_till_death, 3.0)
        self.assertEqual(tracks.max_attacks_per_mob_before_unreachable, 9)

    def test_kill_history_survives_area_reset(self) -> None:
        tracks = HuntTracks(load_detector_config(), skill_delay_ms=500)
        track_id = tracks.create_track("horn", 874, 578, 0.65, 0.9, now_tick=self.now).id
        for i in range(3):
            tracks.apply_attack_event(track_id, now_tick=self.now + i + 1)
        tracks.apply_tracking([_dead(track_id, 874, 578)], now_tick=self.now + 10)
        tracks.area_reset()
        self.assertEqual(tracks.kill_sample_count, 1)
        self.assertEqual(tracks.average_attacks_till_death, 3.0)

    def test_kill_history_caps_at_configured_window(self) -> None:
        config = {**load_detector_config(), "attacksTillDeathHistoryWindow": 3}
        tracks = HuntTracks(config, skill_delay_ms=500)
        for n in range(4):
            track_id = tracks.create_track(
                "horn",
                874 + n * 10,
                578,
                0.65,
                0.9,
                now_tick=self.now + n,
            ).id
            for i in range(2):
                tracks.apply_attack_event(track_id, now_tick=self.now + n * 10 + i + 1)
            tracks.apply_tracking(
                [_dead(track_id, 874 + n * 10, 578)],
                now_tick=self.now + n * 10 + 5,
            )
        self.assertEqual(tracks.kill_sample_count, 3)
        self.assertEqual(tracks.average_attacks_till_death, 2.0)

    def test_pending_attack_credits_killing_blow_sample(self) -> None:
        tracks = HuntTracks(load_detector_config(), skill_delay_ms=500)
        track_id = tracks.create_track("horn", 874, 578, 0.65, 0.9, now_tick=self.now).id
        tracks.apply_attack_event(track_id, now_tick=self.now + 1)
        tracks.apply_attack_event(track_id, now_tick=self.now + 2)
        tracks.mark_attack_pending(track_id)
        tracks.apply_tracking([_dead(track_id, 874, 578)], now_tick=self.now + 3)
        self.assertEqual(tracks.kill_sample_count, 1)
        self.assertEqual(tracks.average_attacks_till_death, 3.0)

    def test_default_max_attacks_before_any_kills(self) -> None:
        tracks = HuntTracks(load_detector_config(), skill_delay_ms=5000)
        self.assertEqual(tracks.average_attacks_till_death, 1.0)
        self.assertEqual(tracks.max_attacks_per_mob_before_unreachable, 2)

    def test_faster_attack_delay_allows_more_attacks(self) -> None:
        slow = HuntTracks(load_detector_config(), skill_delay_ms=5000)
        fast = HuntTracks(load_detector_config(), skill_delay_ms=500)
        self.assertLess(slow.max_attacks_per_mob_before_unreachable, fast.max_attacks_per_mob_before_unreachable)

    def test_tracking_hit_resets_miss_streak(self) -> None:
        track_id = self._create(874, 578)
        miss_limit = int(load_detector_config()["trackLostMissLimit"])
        for i in range(miss_limit - 1):
            self.tracks.apply_tracking([_miss(track_id)], now_tick=self.now + i)
        # A hit clears the streak, so the track survives further misses.
        self.tracks.apply_tracking([_hit(track_id, 880, 580)], now_tick=self.now + 100)
        self.tracks.apply_tracking([_miss(track_id)], now_tick=self.now + 101)
        self.assertIsNotNone(self.tracks.get_track_by_id(track_id))

    def test_attack_event_resets_lost_streak(self) -> None:
        track_id = self._create(874, 578)
        miss_limit = int(load_detector_config()["trackLostMissLimit"])
        for i in range(miss_limit - 1):
            self.tracks.apply_tracking([_miss(track_id)], now_tick=self.now + i)
        track = self.tracks.get_track_by_id(track_id)
        assert track is not None
        self.assertEqual(track.lost_count, miss_limit - 1)
        self.tracks.apply_attack_event(track_id, now_tick=self.now + 50)
        track = self.tracks.get_track_by_id(track_id)
        assert track is not None
        self.assertEqual(track.lost_count, 0)

    def test_lost_site_blocks_discovery_rediscovery(self) -> None:
        track_id = self._create(874, 578)
        miss_limit = int(load_detector_config()["trackLostMissLimit"])
        for i in range(miss_limit):
            self.tracks.apply_tracking([_miss(track_id)], now_tick=self.now + i)
        summary = self.tracks.reconcile_detections(
            [det(874, 578, 0.75, 0.9)],
            mob_name="horn",
            now_tick=self.now + 100,
        )
        self.assertEqual(summary.added_count, 0)
        self.assertEqual(summary.matched_count, 1)
        self.assertEqual(self.tracks.get_track_count(), 0)

    def test_area_reset_clears_tracks(self) -> None:
        self._create(874, 578)
        self.tracks.area_reset()
        self.assertEqual(self.tracks.get_track_count(), 0)
        self.assertEqual(self.tracks.area_epoch, 1)

    def test_reconcile_aborts_when_area_epoch_advanced(self) -> None:
        epoch = self.tracks.area_epoch
        self.tracks.area_reset()
        summary = self.tracks.reconcile_detections(
            [det(100, 200)],
            mob_name="horn",
            now_tick=self.now,
            area_epoch=epoch,
        )
        self.assertEqual(summary.added_count, 0)
        self.assertEqual(self.tracks.get_track_count(), 0)
        self.assertEqual(self.tracks.area_epoch, epoch + 1)

    def test_clear_attack_pending_drops_inflight_mark(self) -> None:
        track_id = self._create(874, 578)
        self.tracks.mark_attack_pending(track_id)
        self.tracks.clear_attack_pending(track_id)
        track = self.tracks.get_track_by_id(track_id)
        assert track is not None
        # Death sample must not credit a phantom pending click.
        self.tracks.apply_tracking([_dead(track_id, x=874, y=578)], now_tick=self.now + 1)
        self.assertIsNone(self.tracks.get_track_by_id(track_id))

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
