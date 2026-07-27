"""Tests for thread-safe HuntTracks"""

from __future__ import annotations

import threading
import unittest
from types import SimpleNamespace

from pybot.recognition.detector.detector import load_detector_config
from pybot.recognition.rules import DiscoveryDetection
from pybot.runtime.hunt_policy import HuntPolicy
from pybot.runtime.hunt_tracks import HuntTracks


def _hit(
    track_id: int,
    x: int,
    y: int,
    confidence: float = 0.8,
    *,
    opacity_score: float = 0.55,
) -> SimpleNamespace:
    return SimpleNamespace(
        track_id=track_id,
        found=True,
        x=x,
        y=y,
        confidence=confidence,
        opacity_score=opacity_score,
    )


def _miss(track_id: int) -> SimpleNamespace:
    return SimpleNamespace(
        track_id=track_id,
        found=False,
        x=0,
        y=0,
        confidence=0.0,
        opacity_score=0.0,
    )


def det(
    x: int,
    y: int,
    confidence: float = 0.71,
    scale: float = 0.9,
    *,
    bbox: tuple[int, int, int, int] | None = None,
) -> DiscoveryDetection:
    if bbox is None:
        bbox = (x - 20, y - 20, 40, 40)
    return DiscoveryDetection(
        x=x,
        y=y,
        confidence=confidence,
        candidate_scale=scale,
        living=True,
        bbox=bbox,
    )


class HuntTracksRulesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_detector_config()
        self.tracks = HuntTracks(self.config, skill_delay_ms=5000)
        self.policy = HuntPolicy()
        self.now = 1_000_000

    def _create(self, x: int, y: int) -> int:
        """Create a track directly (tracking owns track creation)."""
        return self.tracks.create_track("horn", x, y, 0.71, 0.9, now_tick=self.now).id

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

    def test_discovery_matches_existing_track_resetting_miss_count(self) -> None:
        # A detection near an existing track is recognised as the same object
        # (no duplicate), resets discovery_miss_count, and does NOT move x/y.
        # Tracking owns all position writes.
        track_id = self._create(874, 578)
        summary = self.tracks.process_discovery_scan(
            [det(900, 610, 0.71, 0.9)],
            mob_name="horn",
            now_tick=self.now + 500,
        )
        self.assertEqual(summary.added_count, 0)  # matched, not a new candidate
        self.assertEqual(summary.matched_count, 1)
        self.assertEqual(summary.removed_count, 0)
        self.assertEqual(self.tracks.get_track_count(), 1)
        track = self.tracks.get_track_by_id(track_id)
        assert track is not None
        # Position unchanged — tracking owns that
        self.assertEqual(track.x, 874)
        self.assertEqual(track.y, 578)
        # Discovery miss count reset by match
        self.assertEqual(track.discovery_miss_count, 0)

    def test_tracking_miss_advances_lost_count_normally(self) -> None:
        """Miss advances lost count — no soft prior to snap to."""
        track_id = self._create(874, 578)
        self.tracks.apply_tracking([_miss(track_id)], now_tick=self.now + 600)
        track = self.tracks.get_track_by_id(track_id)
        assert track is not None
        self.assertEqual(track.lost_count, 1)
        self.assertEqual((track.x, track.y), (874, 578))

    def test_tracking_hit_updates_position(self) -> None:
        track_id = self._create(874, 578)
        self.tracks.apply_tracking(
            [_hit(track_id, 905, 615)],
            now_tick=self.now + 600,
        )
        track = self.tracks.get_track_by_id(track_id)
        assert track is not None
        self.assertEqual((track.x, track.y), (905, 615))

    def test_outside_roi_unmatched_removes_immediately(self) -> None:
        from pybot.runtime.capture.window_roi import HuntRoi

        # Capture-time position is outside ROI — removed immediately.
        track_id = self.tracks.create_track(
            "horn", 50, 50, 0.65, 0.9, now_tick=self.now
        ).id
        track = self.tracks.get_track_by_id(track_id)
        assert track is not None
        track.x = 900
        track.y = 600
        roi = HuntRoi(x=800, y=500, w=200, h=200)
        summary = self.tracks.process_discovery_scan(
            [],
            mob_name="horn",
            now_tick=self.now + 100,
            existing_track_positions=[(track_id, 50, 50)],
            existing_positions=[],
            hunt_roi=roi,
        )
        self.assertEqual(summary.removed_count, 1)
        self.assertEqual(summary.removed_ids, [track_id])
        self.assertIsNone(self.tracks.get_track_by_id(track_id))

    def test_discovery_marks_absent_inside_hunt_roi_without_removing(self) -> None:
        # In-ROI discovery miss marks the track; tracking removes on joint miss.
        from pybot.runtime.capture.window_roi import HuntRoi

        kept = self._create(874, 578)
        also_inside = self.tracks.create_track(
            "horn", 900, 600, 0.65, 0.9, now_tick=self.now
        ).id
        roi = HuntRoi(x=0, y=0, w=2000, h=2000)
        summary = self.tracks.process_discovery_scan(
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
        self.assertEqual(kept_track.discovery_miss_count, 0)
        self.assertEqual(absent_track.discovery_miss_count, 1)

    def test_two_discovery_misses_removes_track(self) -> None:
        track_id = self._create(874, 578)
        # First miss → miss_count = 1, track survives
        summary = self.tracks.process_discovery_scan(
            [], mob_name="horn", now_tick=self.now + 50,
        )
        self.assertEqual(summary.removed_count, 0)
        self.assertIsNotNone(self.tracks.get_track_by_id(track_id))
        track = self.tracks.get_track_by_id(track_id)
        assert track is not None
        self.assertEqual(track.discovery_miss_count, 1)
        # Second miss → miss_count = 2, track removed
        summary = self.tracks.process_discovery_scan(
            [], mob_name="horn", now_tick=self.now + 100,
        )
        self.assertEqual(summary.removed_count, 1)
        self.assertEqual(summary.removed_ids, [track_id])
        self.assertIsNone(self.tracks.get_track_by_id(track_id))

    def test_discovery_miss_preserved_when_tracking_hits(self) -> None:
        """Tracker hit does NOT reset discovery_miss_count.

        Only discovery determines liveness. The tracker is a pure follower —
        if it finds background noise it should not interfere with discovery's
        2-miss removal.
        """
        track_id = self._create(874, 578)
        self.tracks.process_discovery_scan([], mob_name="horn", now_tick=self.now + 50)
        track = self.tracks.get_track_by_id(track_id)
        assert track is not None
        self.assertEqual(track.discovery_miss_count, 1)
        self.tracks.apply_tracking(
            [_hit(track_id, 880, 580)],
            now_tick=self.now + 100,
        )
        track = self.tracks.get_track_by_id(track_id)
        assert track is not None
        # tracker hit does NOT reset discovery_miss_count — stays at 1
        self.assertEqual(track.discovery_miss_count, 1)

    def test_outside_roi_removed_gone_track_inside_roi_marked_absent(self) -> None:
        from pybot.runtime.capture.window_roi import HuntRoi

        kept = self._create(874, 578)
        gone = self.tracks.create_track(
            "horn", 50, 50, 0.65, 0.9, now_tick=self.now
        ).id
        # ROI covers the kept mob but not (50,50).
        roi = HuntRoi(x=800, y=500, w=200, h=200)
        summary = self.tracks.process_discovery_scan(
            [det(874, 578, 0.75, 0.9)],
            mob_name="horn",
            now_tick=self.now + 100,
            hunt_roi=roi,
        )
        self.assertEqual(summary.added_count, 0)
        self.assertEqual(summary.matched_count, 1)
        # gone track was outside ROI — removed immediately
        self.assertEqual(summary.removed_count, 1)
        self.assertEqual(summary.removed_ids, [gone])
        self.assertIsNotNone(self.tracks.get_track_by_id(kept))
        self.assertIsNone(self.tracks.get_track_by_id(gone))

    def test_discovery_without_roi_does_not_remove_absent_tracks(self) -> None:
        first = self._create(874, 578)
        second = self.tracks.create_track(
            "horn", 200, 200, 0.65, 0.9, now_tick=self.now
        ).id
        summary = self.tracks.process_discovery_scan(
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
        missed_ids, _opacity_dead = self.tracks.apply_tracking(
            [_miss(track_id)],
            now_tick=self.now + 5_000,
        )
        self.assertEqual(missed_ids, [track_id])
        track = self.tracks.get_track_by_id(track_id)
        assert track is not None
        self.assertEqual(track.lost_count, 1)

    def test_two_miss_drop_allows_recreate(self) -> None:
        track_id = self._create(874, 578)
        # Two consecutive misses → removed
        self.tracks.process_discovery_scan(
            [], mob_name="horn", now_tick=self.now + 50,
        )
        self.tracks.process_discovery_scan(
            [], mob_name="horn", now_tick=self.now + 100,
        )
        self.assertIsNone(self.tracks.get_track_by_id(track_id))
        # Re-create after removal (tracking creates the new track)
        new_id = self.tracks.create_track(
            "horn", 874, 578, 0.75, 0.9, now_tick=self.now + 200,
        ).id
        self.assertIsNotNone(self.tracks.get_track_by_id(new_id))
        self.assertEqual(self.tracks.get_alive_count(), 1)

    def test_stale_tracking_after_area_reset_is_ignored(self) -> None:
        track_id = self._create(874, 578)
        epoch = self.tracks.area_epoch
        self.tracks.area_reset()
        new_id = self.tracks.create_track(
            "horn", 900, 600, 0.7, 0.9, now_tick=self.now + 1
        ).id
        self.assertEqual(new_id, track_id)  # ids reuse after reset
        missed_ids, _opacity_dead = self.tracks.apply_tracking(
            [_miss(track_id)],
            now_tick=self.now + 2,
            area_epoch=epoch,
        )
        self.assertEqual(missed_ids, [])
        surviving = self.tracks.get_track_by_id(new_id)
        assert surviving is not None
        self.assertEqual((surviving.x, surviving.y), (900, 600))

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

    def test_attack_event_does_not_clear_lost_streak(self) -> None:
        track_id = self._create(874, 578)
        self.tracks.apply_tracking([_miss(track_id)], now_tick=self.now + 1)
        track = self.tracks.get_track_by_id(track_id)
        assert track is not None
        self.assertEqual(track.lost_count, 1)
        self.tracks.apply_attack_event(track_id, now_tick=self.now + 50)
        track = self.tracks.get_track_by_id(track_id)
        assert track is not None
        self.assertEqual(track.lost_count, 1)
        self.assertEqual(track.attack_count, 1)

    def test_area_reset_clears_tracks(self) -> None:
        self._create(874, 578)
        self.tracks.area_reset()
        self.assertEqual(self.tracks.get_track_count(), 0)
        self.assertEqual(self.tracks.area_epoch, 1)

    def test_reconcile_aborts_when_area_epoch_advanced(self) -> None:
        epoch = self.tracks.area_epoch
        self.tracks.area_reset()
        summary = self.tracks.process_discovery_scan(
            [det(100, 200)],
            mob_name="horn",
            now_tick=self.now,
            area_epoch=epoch,
        )
        self.assertEqual(summary.added_count, 0)
        self.assertEqual(self.tracks.get_track_count(), 0)
        self.assertEqual(self.tracks.area_epoch, epoch + 1)

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

    def test_discovery_publishes_candidates(self) -> None:
        """Discovery publishes candidates; tracking ingests and creates."""
        self._create(874, 578)
        # Process a scan with one new detection far from existing track
        summary = self.tracks.process_discovery_scan(
            [det(874, 578, 0.75, 0.9), det(500, 300, 0.8, 0.85)],
            mob_name="horn",
            now_tick=self.now + 500,
        )
        # One matched, one new candidate
        self.assertEqual(summary.matched_count, 1)
        self.assertEqual(summary.added_count, 1)

        # Candidate should be available for tracking to ingest
        candidates = self.tracks.get_and_clear_new_candidates()
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].x, 500)
        self.assertEqual(candidates[0].y, 300)

        # Second call returns empty (already cleared)
        self.assertEqual(len(self.tracks.get_and_clear_new_candidates()), 0)

    def test_opacity_decay_removes_stationary_track(self) -> None:
        track_id = self._create(874, 578)
        # Calibrate living baseline (same pixel → stationary).
        for i in range(4):
            missed, dead = self.tracks.apply_tracking(
                [_hit(track_id, 874, 578, opacity_score=0.60)],
                now_tick=self.now + (i + 1) * 16,
            )
            self.assertEqual(missed, [])
            self.assertEqual(dead, [])
        track = self.tracks.get_track_by_id(track_id)
        assert track is not None
        self.assertEqual(track.opacity_baseline_samples, 4)
        self.assertGreaterEqual(track.opacity_baseline, 0.60)

        # One decay frame — not yet confirmed.
        missed, dead = self.tracks.apply_tracking(
            [_hit(track_id, 874, 578, opacity_score=0.20)],
            now_tick=self.now + 100,
        )
        self.assertEqual(missed, [])
        self.assertEqual(dead, [])
        self.assertIsNotNone(self.tracks.get_track_by_id(track_id))

        # Second consecutive decay frame confirms death.
        missed, dead = self.tracks.apply_tracking(
            [_hit(track_id, 874, 578, opacity_score=0.18)],
            now_tick=self.now + 200,
        )
        self.assertEqual(missed, [])
        self.assertEqual([e.track_id for e in dead], [track_id])
        self.assertIsNone(self.tracks.get_track_by_id(track_id))

    def test_opacity_decay_ignored_while_moving(self) -> None:
        track_id = self._create(874, 578)
        for i in range(4):
            self.tracks.apply_tracking(
                [_hit(track_id, 874, 578, opacity_score=0.60)],
                now_tick=self.now + (i + 1) * 16,
            )
        # Large displacement → moving; low opacity must not remove.
        for i in range(5):
            x = 874 + (i + 1) * 20
            missed, dead = self.tracks.apply_tracking(
                [_hit(track_id, x, 578, opacity_score=0.10)],
                now_tick=self.now + 100 + (i + 1) * 16,
            )
            self.assertEqual(missed, [])
            self.assertEqual(dead, [])
        self.assertIsNotNone(self.tracks.get_track_by_id(track_id))

    def test_five_idle_attacks_remove_unreachable(self) -> None:
        track_id = self._create(500, 500)
        # Far from character — not melee-guarded.
        for i in range(4):
            action, count = self.tracks.evaluate_idle_attack(
                track_id,
                was_idle=True,
                mob_x=500,
                mob_y=500,
                char_x=0,
                char_y=0,
                now_tick=self.now + i,
            )
            self.assertEqual(action, "none")
            self.assertEqual(count, i + 1)
            track = self.tracks.get_track_by_id(track_id)
            assert track is not None
            self.assertEqual(track.state, "alive")

        action, count = self.tracks.evaluate_idle_attack(
            track_id,
            was_idle=True,
            mob_x=500,
            mob_y=500,
            char_x=0,
            char_y=0,
            now_tick=self.now + 10,
        )
        self.assertEqual(action, "unreachable")
        self.assertEqual(count, 5)
        self.assertIsNone(self.tracks.get_track_by_id(track_id))
        # Death site blocks rediscovery at the same coordinates.
        summary = self.tracks.process_discovery_scan(
            [det(500, 500)],
            mob_name="horn",
            now_tick=self.now + 20,
        )
        self.assertEqual(summary.added_count, 0)
        self.assertEqual(len(self.tracks.get_and_clear_new_candidates()), 0)

    def test_idle_streak_resets_on_sp_spend(self) -> None:
        track_id = self._create(500, 500)
        for _ in range(3):
            self.tracks.evaluate_idle_attack(
                track_id, was_idle=True, mob_x=500, mob_y=500, char_x=0, char_y=0,
            )
        action, count = self.tracks.evaluate_idle_attack(
            track_id, was_idle=False, mob_x=500, mob_y=500, char_x=0, char_y=0,
        )
        self.assertEqual(action, "none")
        self.assertEqual(count, 0)
        track = self.tracks.get_track_by_id(track_id)
        assert track is not None
        self.assertTrue(track.was_accessible)
        self.assertEqual(track.idle_attack_count, 0)

    def test_unknown_sp_does_not_reset_idle_or_fake_access(self) -> None:
        track_id = self._create(500, 500)
        self.tracks.evaluate_idle_attack(
            track_id, was_idle=True, mob_x=500, mob_y=500, char_x=0, char_y=0,
        )
        action, count = self.tracks.evaluate_idle_attack(
            track_id, was_idle=None, mob_x=500, mob_y=500, char_x=0, char_y=0,
        )
        self.assertEqual(action, "none")
        self.assertEqual(count, 1)
        track = self.tracks.get_track_by_id(track_id)
        assert track is not None
        self.assertFalse(track.was_accessible)
        self.assertEqual(track.idle_attack_count, 1)

    def test_melee_range_idle_resets_streak(self) -> None:
        track_id = self._create(100, 100)
        for _ in range(3):
            self.tracks.evaluate_idle_attack(
                track_id, was_idle=True, mob_x=500, mob_y=500, char_x=0, char_y=0,
            )
        action, count = self.tracks.evaluate_idle_attack(
            track_id, was_idle=True, mob_x=100, mob_y=100, char_x=100, char_y=100,
        )
        self.assertEqual(action, "none")
        self.assertEqual(count, 0)
        track = self.tracks.get_track_by_id(track_id)
        assert track is not None
        self.assertEqual(track.idle_attack_count, 0)

    def test_accessible_stationary_dies_at_two_idle(self) -> None:
        track_id = self._create(500, 500)
        self.tracks.evaluate_idle_attack(
            track_id, was_idle=False, mob_x=500, mob_y=500, char_x=0, char_y=0,
        )
        track = self.tracks.get_track_by_id(track_id)
        assert track is not None
        track.discovery_stationary = True
        action, count = self.tracks.evaluate_idle_attack(
            track_id, was_idle=True, mob_x=500, mob_y=500, char_x=0, char_y=0,
        )
        self.assertEqual(action, "none")
        self.assertEqual(count, 1)
        action, count = self.tracks.evaluate_idle_attack(
            track_id, was_idle=True, mob_x=500, mob_y=500, char_x=0, char_y=0,
        )
        self.assertEqual(action, "dead")
        self.assertEqual(count, 2)
        self.assertIsNone(self.tracks.get_track_by_id(track_id))

    def test_idle_death_blocks_rediscovery_at_death_site(self) -> None:
        track_id = self._create(500, 500)
        self.tracks.evaluate_idle_attack(
            track_id, was_idle=False, mob_x=500, mob_y=500, char_x=0, char_y=0,
            now_tick=self.now,
        )
        track = self.tracks.get_track_by_id(track_id)
        assert track is not None
        track.discovery_stationary = True
        self.tracks.evaluate_idle_attack(
            track_id, was_idle=True, mob_x=500, mob_y=500, char_x=0, char_y=0,
            now_tick=self.now + 10,
        )
        self.tracks.evaluate_idle_attack(
            track_id, was_idle=True, mob_x=500, mob_y=500, char_x=0, char_y=0,
            now_tick=self.now + 20,
        )
        self.assertIsNone(self.tracks.get_track_by_id(track_id))

        # Corpse heat slightly offset from the death site must still be held.
        summary = self.tracks.process_discovery_scan(
            [det(505, 502)],
            mob_name="horn",
            now_tick=self.now + 50,
        )
        self.assertEqual(summary.added_count, 0)
        self.assertEqual(len(self.tracks.get_and_clear_new_candidates()), 0)
        self.assertEqual(self.tracks.get_alive_count(), 0)

        # Drift within deathSiteRadiusPx (160) — still blocked, site follows.
        summary = self.tracks.process_discovery_scan(
            [det(500 + 120, 500)],
            mob_name="horn",
            now_tick=self.now + 100,
        )
        self.assertEqual(summary.added_count, 0)
        self.assertEqual(len(self.tracks.get_and_clear_new_candidates()), 0)

    def test_opacity_death_blocks_rediscovery_at_death_site(self) -> None:
        track_id = self._create(500, 500)
        for i in range(4):
            self.tracks.apply_tracking(
                [_hit(track_id, 500, 500, opacity_score=0.60)],
                now_tick=self.now + (i + 1) * 16,
            )
        missed, dead = self.tracks.apply_tracking(
            [_hit(track_id, 500, 500, opacity_score=0.20)],
            now_tick=self.now + 100,
        )
        self.assertEqual(dead, [])
        missed, dead = self.tracks.apply_tracking(
            [_hit(track_id, 500, 500, opacity_score=0.18)],
            now_tick=self.now + 200,
        )
        self.assertEqual([e.track_id for e in dead], [track_id])
        self.assertIsNone(self.tracks.get_track_by_id(track_id))

        summary = self.tracks.process_discovery_scan(
            [det(500, 500)],
            mob_name="horn",
            now_tick=self.now + 250,
        )
        self.assertEqual(summary.added_count, 0)
        self.assertEqual(len(self.tracks.get_and_clear_new_candidates()), 0)

    def test_death_site_expires_after_cooldown(self) -> None:
        track_id = self._create(500, 500)
        self.tracks.evaluate_idle_attack(
            track_id, was_idle=False, mob_x=500, mob_y=500, char_x=0, char_y=0,
            now_tick=self.now,
        )
        track = self.tracks.get_track_by_id(track_id)
        assert track is not None
        track.discovery_stationary = True
        self.tracks.evaluate_idle_attack(
            track_id, was_idle=True, mob_x=500, mob_y=500, char_x=0, char_y=0,
            now_tick=self.now + 10,
        )
        self.tracks.evaluate_idle_attack(
            track_id, was_idle=True, mob_x=500, mob_y=500, char_x=0, char_y=0,
            now_tick=self.now + 20,
        )
        cooldown = int(self.config["deathRediscoveryCooldownMs"])
        # Past cooldown — corpse site no longer blocks a new candidate.
        summary = self.tracks.process_discovery_scan(
            [det(500, 500)],
            mob_name="horn",
            now_tick=self.now + 20 + cooldown + 100,
        )
        self.assertEqual(summary.added_count, 1)
        self.assertEqual(len(self.tracks.get_and_clear_new_candidates()), 1)

    def test_accessible_without_blob_stationary_removes_at_five(self) -> None:
        track_id = self._create(500, 500)
        self.tracks.evaluate_idle_attack(
            track_id, was_idle=False, mob_x=500, mob_y=500, char_x=0, char_y=0,
        )
        track = self.tracks.get_track_by_id(track_id)
        assert track is not None
        track.discovery_stationary = False
        for i in range(4):
            action, count = self.tracks.evaluate_idle_attack(
                track_id, was_idle=True, mob_x=500, mob_y=500, char_x=0, char_y=0,
            )
            self.assertEqual(action, "none")
            self.assertEqual(count, i + 1)
        action, count = self.tracks.evaluate_idle_attack(
            track_id, was_idle=True, mob_x=500, mob_y=500, char_x=0, char_y=0,
            now_tick=self.now + 50,
        )
        self.assertEqual(action, "unreachable")
        self.assertEqual(count, 5)
        self.assertIsNone(self.tracks.get_track_by_id(track_id))
    def test_discovery_blob_stability_sets_stationary(self) -> None:
        track_id = self._create(500, 500)
        bbox = (480, 480, 40, 40)
        # First match: seed blob, not yet stationary.
        self.tracks.process_discovery_scan(
            [det(500, 500, bbox=bbox)],
            mob_name="horn",
            now_tick=self.now + 50,
        )
        track = self.tracks.get_track_by_id(track_id)
        assert track is not None
        self.assertTrue(track.discovery_blob_seen)
        self.assertFalse(track.discovery_stationary)

        # Stable center coords → stationary (bbox shape ignored).
        self.tracks.process_discovery_scan(
            [det(501, 500, bbox=(470, 470, 60, 60))],
            mob_name="horn",
            now_tick=self.now + 100,
        )
        track = self.tracks.get_track_by_id(track_id)
        assert track is not None
        self.assertTrue(track.discovery_stationary)

        # Same center again → still stationary.
        self.tracks.process_discovery_scan(
            [det(501, 500, bbox=(480, 480, 40, 40))],
            mob_name="horn",
            now_tick=self.now + 120,
        )
        track = self.tracks.get_track_by_id(track_id)
        assert track is not None
        self.assertTrue(track.discovery_stationary)

        # Moved center → not stationary.
        self.tracks.process_discovery_scan(
            [det(560, 500, bbox=(540, 480, 40, 40))],
            mob_name="horn",
            now_tick=self.now + 150,
        )
        track = self.tracks.get_track_by_id(track_id)
        assert track is not None
        self.assertFalse(track.discovery_stationary)


if __name__ == "__main__":
    unittest.main()
