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


def _death_result(track_id: int) -> tuple[int, float, int, int, bool]:
    """Return a death-result tuple for apply_death_results()."""
    return (track_id, 0.6, 4, 0, True)


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
        self.config = load_detector_config()
        self.tracks = HuntTracks(self.config, skill_delay_ms=5000)
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
        # (no duplicate) but does NOT move authoritative x/y — tracking owns
        # position. Discovery publishes a soft prior instead.
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
        self.assertEqual(track.discovery_obs_x, 900)
        self.assertEqual(track.discovery_obs_y, 610)
        self.assertEqual(track.discovery_obs_tick, self.now + 500)
        self.assertFalse(track.discovery_absent)

    def test_tracking_miss_snaps_to_discovery_obs(self) -> None:
        track_id = self._create(874, 578)
        self.tracks.reconcile_detections(
            [det(900, 610, 0.71, 0.9)],
            mob_name="horn",
            now_tick=self.now + 500,
        )
        missed_ids = self.tracks.apply_tracking(
            [_miss(track_id)],
            now_tick=self.now + 600,
        )
        self.assertEqual(missed_ids, [])  # reanchor succeeded — not a miss
        track = self.tracks.get_track_by_id(track_id)
        assert track is not None
        self.assertEqual((track.x, track.y), (900, 610))
        self.assertEqual(track.lost_count, 0)
        # Prior kept until a real local hit confirms.
        self.assertEqual(track.discovery_obs_tick, self.now + 500)

    def test_tracking_miss_at_discovery_obs_advances_lost_count(self) -> None:
        track_id = self._create(874, 578)
        self.tracks.reconcile_detections(
            [det(874, 578, 0.71, 0.9)],
            mob_name="horn",
            now_tick=self.now + 500,
        )
        self.tracks.apply_tracking([_miss(track_id)], now_tick=self.now + 600)
        track = self.tracks.get_track_by_id(track_id)
        assert track is not None
        # Already at prior — no snap; normal miss accounting.
        self.assertEqual(track.lost_count, 1)
        self.assertEqual((track.x, track.y), (874, 578))

    def test_tracking_hit_clears_discovery_obs(self) -> None:
        track_id = self._create(874, 578)
        self.tracks.reconcile_detections(
            [det(900, 610, 0.71, 0.9)],
            mob_name="horn",
            now_tick=self.now + 500,
        )
        self.tracks.apply_tracking(
            [_hit(track_id, 905, 615)],
            now_tick=self.now + 600,
        )
        track = self.tracks.get_track_by_id(track_id)
        assert track is not None
        self.assertEqual((track.x, track.y), (905, 615))
        self.assertEqual(track.discovery_obs_tick, 0)
        self.assertFalse(track.discovery_absent)

    def test_outside_roi_unmatched_marks_absent_for_tracker(self) -> None:
        from pybot.runtime.capture.window_roi import HuntRoi

        # Capture-time position is outside ROI; live track has since coasted in.
        # Discovery only notifies absent — tracking owns the drop.
        track_id = self.tracks.create_track(
            "horn", 50, 50, 0.65, 0.9, now_tick=self.now
        ).id
        track = self.tracks.get_track_by_id(track_id)
        assert track is not None
        track.x = 900
        track.y = 600
        roi = HuntRoi(x=800, y=500, w=200, h=200)
        summary = self.tracks.reconcile_detections(
            [],
            mob_name="horn",
            now_tick=self.now + 100,
            existing_track_positions=[(track_id, 50, 50)],
            existing_positions=[],
            hunt_roi=roi,
        )
        self.assertEqual(summary.removed_count, 0)
        track = self.tracks.get_track_by_id(track_id)
        assert track is not None
        self.assertTrue(track.discovery_absent)

    def test_discovery_marks_absent_inside_hunt_roi_without_removing(self) -> None:
        # In-ROI discovery miss marks the track; tracking removes on joint miss.
        from pybot.runtime.capture.window_roi import HuntRoi

        kept = self._create(874, 578)
        also_inside = self.tracks.create_track(
            "horn", 900, 600, 0.65, 0.9, now_tick=self.now
        ).id
        roi = HuntRoi(x=0, y=0, w=2000, h=2000)
        summary = self.tracks.reconcile_detections(
            [det(874, 578, 0.75, 0.9)],
            mob_name="horn",
            now_tick=self.now + 100,
            hunt_roi=roi,
        )
        self.assertEqual(summary.added_count, 0)
        self.assertEqual(summary.matched_count, 1)
        self.assertEqual(summary.removed_count, 0)
        kept_track = self.tracks.get_track_by_id(kept)
        absent_track = self.tracks.get_track_by_id(also_inside)
        assert kept_track is not None
        assert absent_track is not None
        self.assertFalse(kept_track.discovery_absent)
        self.assertTrue(absent_track.discovery_absent)

    def test_joint_discovery_tracking_miss_removes_track(self) -> None:
        track_id = self._create(874, 578)
        absent_at = self.now + 50
        self.tracks.reconcile_detections(
            [],
            mob_name="horn",
            now_tick=absent_at,
        )
        track = self.tracks.get_track_by_id(track_id)
        assert track is not None
        self.assertTrue(track.discovery_absent)
        self.assertEqual(track.discovery_absent_tick, absent_at)
        confirm_ms = int(load_detector_config()["trackJointAbsentConfirmMs"])
        # First miss while absent — still searching (clock from absent_at).
        self.tracks.apply_tracking(
            [_miss(track_id)],
            now_tick=absent_at + 100,
        )
        lost_ids, unreachable_ids = self.tracks.apply_tracking_cleanup(
            now_tick=absent_at + 100,
        )
        self.assertEqual(lost_ids, [])
        self.assertEqual(unreachable_ids, [])
        self.assertIsNotNone(self.tracks.get_track_by_id(track_id))
        # Sustained miss past confirm window from discovery_absent_tick.
        self.tracks.apply_tracking(
            [_miss(track_id)],
            now_tick=absent_at + confirm_ms,
        )
        lost_ids, unreachable_ids = self.tracks.apply_tracking_cleanup(
            now_tick=absent_at + confirm_ms,
        )
        self.assertEqual(lost_ids, [track_id])
        self.assertEqual(unreachable_ids, [])
        self.assertIsNone(self.tracks.get_track_by_id(track_id))

    def test_new_track_not_joint_absent_dropped_immediately(self) -> None:
        """Joint-absence clock starts at discovery miss, not create time."""
        track_id = self._create(874, 578)
        confirm_ms = int(load_detector_config()["trackJointAbsentConfirmMs"])
        # Even if create was long ago relative to confirm_ms, a brand-new
        # discovery_absent mark must wait the full confirm window.
        self.tracks.reconcile_detections(
            [],
            mob_name="horn",
            now_tick=self.now + confirm_ms + 1000,
        )
        self.tracks.apply_tracking(
            [_miss(track_id)],
            now_tick=self.now + confirm_ms + 1000,
        )
        lost_ids, unreachable_ids = self.tracks.apply_tracking_cleanup(
            now_tick=self.now + confirm_ms + 1000,
        )
        self.assertEqual(lost_ids, [])
        self.assertEqual(unreachable_ids, [])
        self.assertIsNotNone(self.tracks.get_track_by_id(track_id))


    def test_discovery_absent_cleared_when_tracking_hits(self) -> None:
        track_id = self._create(874, 578)
        self.tracks.reconcile_detections([], mob_name="horn", now_tick=self.now + 50)
        track = self.tracks.get_track_by_id(track_id)
        assert track is not None
        self.assertTrue(track.discovery_absent)
        self.assertGreater(track.discovery_absent_tick, 0)
        self.tracks.apply_tracking(
            [_hit(track_id, 880, 580)],
            now_tick=self.now + 100,
        )
        track = self.tracks.get_track_by_id(track_id)
        assert track is not None
        self.assertFalse(track.discovery_absent)
        self.assertEqual(track.discovery_absent_tick, 0)


    def test_discovery_marks_absent_outside_hunt_roi_without_removing(self) -> None:
        from pybot.runtime.capture.window_roi import HuntRoi

        kept = self._create(874, 578)
        gone = self.tracks.create_track(
            "horn", 50, 50, 0.65, 0.9, now_tick=self.now
        ).id
        # ROI covers the kept mob but not (50,50).
        roi = HuntRoi(x=800, y=500, w=200, h=200)
        summary = self.tracks.reconcile_detections(
            [det(874, 578, 0.75, 0.9)],
            mob_name="horn",
            now_tick=self.now + 100,
            hunt_roi=roi,
        )
        self.assertEqual(summary.added_count, 0)
        self.assertEqual(summary.matched_count, 1)
        self.assertEqual(summary.removed_count, 0)
        self.assertEqual(summary.removed_ids, [])
        self.assertIsNotNone(self.tracks.get_track_by_id(kept))
        gone_track = self.tracks.get_track_by_id(gone)
        assert gone_track is not None
        self.assertTrue(gone_track.discovery_absent)

    def test_discovery_without_roi_does_not_remove_absent_tracks(self) -> None:
        first = self._create(874, 578)
        second = self.tracks.create_track(
            "horn", 200, 200, 0.65, 0.9, now_tick=self.now
        ).id
        summary = self.tracks.reconcile_detections(
            [],
            mob_name="horn",
            now_tick=self.now + 100,
        )
        self.assertEqual(summary.removed_count, 0)
        self.assertEqual(self.tracks.get_track_count(), 2)
        self.assertIsNotNone(self.tracks.get_track_by_id(first))
        self.assertIsNotNone(self.tracks.get_track_by_id(second))

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

    def test_tracking_miss_keeps_track(self) -> None:
        track_id = self._create(874, 578)
        missed_ids = self.tracks.apply_tracking(
            [_miss(track_id)],
            now_tick=self.now + 5_000,
        )
        self.assertEqual(missed_ids, [track_id])
        track = self.tracks.get_track_by_id(track_id)
        assert track is not None
        self.assertEqual(track.lost_count, 1)

    def test_joint_absent_drop_allows_discovery_recreate(self) -> None:
        track_id = self._create(874, 578)
        confirm_ms = int(self.config["trackJointAbsentConfirmMs"])
        absent_at = self.now + 50
        self.tracks.reconcile_detections(
            [],
            mob_name="horn",
            now_tick=absent_at,
        )
        drop_at = absent_at + confirm_ms
        self.tracks.apply_tracking(
            [_miss(track_id)],
            now_tick=drop_at,
        )
        self.tracks.apply_tracking_cleanup(now_tick=drop_at)
        self.assertIsNone(self.tracks.get_track_by_id(track_id))
        summary = self.tracks.reconcile_detections(
            [det(874, 578, 0.75, 0.9)],
            mob_name="horn",
            now_tick=drop_at + 1,
        )
        self.assertEqual(summary.added_count, 1)
        self.assertEqual(summary.alive_after, 1)

    def test_tracking_death_removes_track_immediately(self) -> None:
        track_id = self._create(874, 578)
        dead_ids = self.tracks.apply_death_results(
            [_death_result(track_id)],
            now_tick=self.now + 1,
        )
        self.assertEqual(dead_ids, [track_id])
        self.assertIsNone(self.tracks.get_track_by_id(track_id))

    def test_death_site_blocks_discovery_rediscovery(self) -> None:
        track_id = self._create(874, 578)
        death_at = self.now + 1
        self.tracks.apply_death_results(
            [_death_result(track_id)], now_tick=death_at,
        )
        summary = self.tracks.reconcile_detections(
            [det(874, 578, 0.75, 0.9)],
            mob_name="horn",
            now_tick=death_at + 100,
        )
        self.assertEqual(summary.added_count, 0)
        self.assertEqual(summary.matched_count, 1)
        self.assertEqual(self.tracks.get_track_count(), 0)

    def test_death_site_expires_after_cooldown(self) -> None:
        config = {**self.config, "deathRediscoveryCooldownMs": 1000}
        tracks = HuntTracks(config)
        track_id = tracks.create_track("horn", 874, 578, 0.65, 0.9, now_tick=self.now).id
        death_at = self.now + 1
        tracks.apply_death_results([_death_result(track_id)], now_tick=death_at)
        blocked = tracks.reconcile_detections(
            [det(874, 578)],
            mob_name="horn",
            now_tick=death_at + 500,
        )
        self.assertEqual(blocked.added_count, 0)
        allowed = tracks.reconcile_detections(
            [det(874, 578)],
            mob_name="horn",
            now_tick=death_at + 2000,
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
        _, unreachable_ids = tracks.apply_tracking_cleanup(now_tick=self.now + 3)
        self.assertEqual(unreachable_ids, [track_id])
        self.assertIsNone(tracks.get_track_by_id(track_id))

    def test_attack_limit_is_per_mob_not_shared(self) -> None:
        tracks = HuntTracks(load_detector_config(), skill_delay_ms=1500)
        first = tracks.create_track("horn", 874, 578, 0.65, 0.9, now_tick=self.now).id
        second = tracks.create_track("horn", 980, 640, 0.65, 0.9, now_tick=self.now).id
        for i in range(3):
            tracks.apply_attack_event(first, now_tick=self.now + i + 1)
        _, unreachable_ids = tracks.apply_tracking_cleanup(now_tick=self.now + 3)
        self.assertEqual(unreachable_ids, [first])
        self.assertIsNone(tracks.get_track_by_id(first))
        self.assertIsNotNone(tracks.get_track_by_id(second))
        second_track = tracks.get_track_by_id(second)
        assert second_track is not None
        self.assertEqual(second_track.attack_count, 0)

    def test_unreachable_blocks_discovery_rediscovery(self) -> None:
        tracks = HuntTracks(load_detector_config(), skill_delay_ms=3000)
        track_id = tracks.create_track("horn", 874, 578, 0.65, 0.9, now_tick=self.now).id
        tracks.apply_attack_event(track_id, now_tick=self.now + 1)
        tracks.apply_attack_event(track_id, now_tick=self.now + 2)
        tracks.apply_tracking_cleanup(now_tick=self.now + 2)
        self.assertIsNone(tracks.get_track_by_id(track_id))
        summary = tracks.reconcile_detections(
            [det(874, 578, 0.75, 0.9)],
            mob_name="horn",
            now_tick=self.now + 100,
        )
        self.assertEqual(summary.added_count, 0)
        self.assertEqual(summary.matched_count, 1)
        self.assertEqual(summary.alive_after, 0)

    def test_stale_tracking_after_area_reset_is_ignored(self) -> None:
        track_id = self._create(874, 578)
        epoch = self.tracks.area_epoch
        self.tracks.area_reset()
        new_id = self.tracks.create_track(
            "horn", 900, 600, 0.7, 0.9, now_tick=self.now + 1
        ).id
        self.assertEqual(new_id, track_id)  # ids reuse after reset
        missed_ids = self.tracks.apply_tracking(
            [_miss(track_id)],
            now_tick=self.now + 2,
            area_epoch=epoch,
        )
        self.assertEqual(missed_ids, [])
        surviving = self.tracks.get_track_by_id(new_id)
        assert surviving is not None
        self.assertEqual((surviving.x, surviving.y), (900, 600))

    def test_mob_attack_count_inherits_after_death_recreation(self) -> None:
        config = {
            **self.config,
            "deathRediscoveryCooldownMs": 1000,
        }
        tracks = HuntTracks(config, skill_delay_ms=5000)
        track_id = tracks.create_track("horn", 874, 578, 0.65, 0.9, now_tick=self.now).id
        for i in range(2):
            tracks.apply_attack_event(track_id, now_tick=self.now + i + 1)
        death_at = self.now + 1
        tracks.apply_death_results(
            [_death_result(track_id)],
            now_tick=death_at,
        )
        self.assertIsNone(tracks.get_track_by_id(track_id))
        summary = tracks.reconcile_detections(
            [det(874, 578, 0.75, 0.9)],
            mob_name="horn",
            now_tick=death_at + 500,
        )
        self.assertEqual(summary.added_count, 0)
        summary = tracks.reconcile_detections(
            [det(874, 578, 0.75, 0.9)],
            mob_name="horn",
            now_tick=death_at + 2000,
        )
        self.assertEqual(summary.added_count, 1)
        created = summary.created_ids or []
        self.assertEqual(len(created), 1)
        track = tracks.get_track_by_id(created[0])
        assert track is not None
        self.assertEqual(track.attack_count, 2)
        self.assertFalse(
            tracks.apply_attack_event(created[0], now_tick=death_at + 2001)
        )

    def test_death_records_attacks_till_death_sample(self) -> None:
        tracks = HuntTracks(self.config, skill_delay_ms=500)
        track_id = tracks.reconcile_detections(
            [det(874, 578)],
            mob_name="horn",
            now_tick=self.now,
        ).created_ids[0]
        for i in range(4):
            tracks.apply_attack_event(track_id, now_tick=self.now + i + 1)
        tracks.apply_death_results(
            [_death_result(track_id)],
            now_tick=self.now + 1,
        )
        self.assertEqual(tracks.kill_sample_count, 1)
        self.assertEqual(tracks.average_attacks_till_death, 4.0)
        self.assertEqual(tracks.max_attacks_per_mob_before_unreachable, 10)

    def test_kill_history_builds_rolling_average(self) -> None:
        tracks = HuntTracks(self.config, skill_delay_ms=500)
        first = tracks.create_track("horn", 874, 578, 0.65, 0.9, now_tick=self.now).id
        second = tracks.create_track("horn", 980, 640, 0.65, 0.9, now_tick=self.now).id
        for i in range(4):
            tracks.apply_attack_event(first, now_tick=self.now + i + 1)
        first_death = self.now + 1
        tracks.apply_death_results([_death_result(first)], now_tick=first_death)
        for i in range(2):
            tracks.apply_attack_event(second, now_tick=first_death + 20 + i)
        tracks.apply_death_results(
            [_death_result(second)], now_tick=first_death + 30,
        )
        self.assertEqual(tracks.kill_sample_count, 2)
        self.assertEqual(tracks.average_attacks_till_death, 3.0)
        self.assertEqual(tracks.max_attacks_per_mob_before_unreachable, 9)

    def test_kill_history_survives_area_reset(self) -> None:
        tracks = HuntTracks(self.config, skill_delay_ms=500)
        track_id = tracks.create_track("horn", 874, 578, 0.65, 0.9, now_tick=self.now).id
        for i in range(3):
            tracks.apply_attack_event(track_id, now_tick=self.now + i + 1)
        tracks.apply_death_results(
            [_death_result(track_id)],
            now_tick=self.now + 1,
        )
        tracks.area_reset()
        self.assertEqual(tracks.kill_sample_count, 1)
        self.assertEqual(tracks.average_attacks_till_death, 3.0)

    def test_kill_history_caps_at_configured_window(self) -> None:
        config = {**self.config, "attacksTillDeathHistoryWindow": 3}
        tracks = HuntTracks(config, skill_delay_ms=500)
        step = 20
        for n in range(4):
            created_at = self.now + n * step
            track_id = tracks.create_track(
                "horn",
                874 + n * 10,
                578,
                0.65,
                0.9,
                now_tick=created_at,
            ).id
            for i in range(2):
                tracks.apply_attack_event(track_id, now_tick=created_at + i + 1)
            tracks.apply_death_results(
                [_death_result(track_id)],
                now_tick=created_at + 1,
            )
        self.assertEqual(tracks.kill_sample_count, 3)
        self.assertEqual(tracks.average_attacks_till_death, 2.0)

    def test_pending_attack_credits_killing_blow_sample(self) -> None:
        tracks = HuntTracks(self.config, skill_delay_ms=500)
        track_id = tracks.create_track("horn", 874, 578, 0.65, 0.9, now_tick=self.now).id
        tracks.apply_attack_event(track_id, now_tick=self.now + 1)
        tracks.apply_attack_event(track_id, now_tick=self.now + 2)
        tracks.mark_attack_pending(track_id)
        tracks.apply_death_results(
            [_death_result(track_id)],
            now_tick=self.now + 1,
        )
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
        self.tracks.apply_tracking(
            [_miss(track_id)],
            now_tick=self.now + 1,
        )
        track = self.tracks.get_track_by_id(track_id)
        assert track is not None
        self.assertEqual(track.lost_count, 1)
        self.tracks.apply_tracking([_hit(track_id, 880, 580)], now_tick=self.now + 100)
        track = self.tracks.get_track_by_id(track_id)
        assert track is not None
        self.assertEqual(track.lost_count, 0)
        self.tracks.apply_tracking(
            [_miss(track_id)],
            now_tick=self.now + 200,
        )
        track = self.tracks.get_track_by_id(track_id)
        assert track is not None
        self.assertEqual(track.lost_count, 1)

    def test_attack_event_resets_lost_streak(self) -> None:
        track_id = self._create(874, 578)
        self.tracks.apply_tracking([_miss(track_id)], now_tick=self.now + 1)
        track = self.tracks.get_track_by_id(track_id)
        assert track is not None
        self.assertEqual(track.lost_count, 1)
        self.tracks.apply_attack_event(track_id, now_tick=self.now + 50)
        track = self.tracks.get_track_by_id(track_id)
        assert track is not None
        self.assertEqual(track.lost_count, 0)

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
        self.tracks.apply_death_results(
            [_death_result(track_id)],
            now_tick=self.now + 1,
        )
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
