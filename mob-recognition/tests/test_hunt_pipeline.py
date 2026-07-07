"""Hunt pipeline contract tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent.parent
MOB_REC = Path(__file__).resolve().parent.parent
SIMPLE = MOB_REC / "simple"
for path in (str(MOB_REC), str(SIMPLE)):
    if path not in sys.path:
        sys.path.insert(0, path)

from cli import apply_scale_calibration  # noqa: E402
from detector import SimpleMobDetector, load_simple_config  # noqa: E402
from hunt_track_rules import (  # noqa: E402
    HUNT_ATTACK_RESULT_WINDOW_MS,
    HUNT_UNREACHABLE_CONFIRM_STREAK,
    MobTrack,
    StateObservation,
    apply_attack_event,
    apply_state_observation,
    attack_block_reason,
    collect_state_requests,
    is_attackable,
    is_pending,
    select_target_id,
    state_request_scale,
)
from tracking.state_recognizer import evaluate_track_states  # noqa: E402


def playfield_roi(frame):
    height, width = frame.shape[:2]
    return frame[
        int(height * 0.08) : int(height * 0.92),
        int(width * 0.03) : int(width * 0.97),
    ]


class HuntTrackRulesTests(unittest.TestCase):
    def test_newly_discovered_track_is_attackable_immediately(self) -> None:
        now = 1_000_000
        track = MobTrack.from_discovery(1, 874, 578, 0.65, now_tick=now, discovery_scale=0.9)
        self.assertTrue(is_attackable(track, now))
        self.assertEqual(attack_block_reason(track, now), "")

    def test_ignored_unreachable_does_not_block_first_attack(self) -> None:
        now = 1_000_000
        track = MobTrack.from_discovery(1, 874, 578, 0.65, now_tick=now, discovery_scale=0.9)
        apply_state_observation(track, StateObservation(1, "unreachable"), now + 7_000)
        self.assertEqual(track.state_unreachable_count, 1)
        self.assertEqual(track.last_state_tick, 0)
        self.assertEqual(track.state, "alive")
        self.assertTrue(is_attackable(track, now + 7_000))

    def test_alive_state_refreshes_coords(self) -> None:
        now = 1_000_000
        track = MobTrack.from_discovery(1, 874, 578, 0.65, now_tick=now)
        apply_state_observation(
            track,
            StateObservation(1, "alive", x=870, y=580, confidence=0.7),
            now + 100,
        )
        self.assertEqual(track.x, 870)
        self.assertEqual(track.y, 580)
        self.assertEqual(track.last_state_tick, now + 100)
        self.assertTrue(is_attackable(track, now + 100))

    def test_after_attack_pending_blocks_until_resolved(self) -> None:
        now = 1_000_000
        track = MobTrack.from_discovery(1, 874, 578, 0.65, now_tick=now)
        apply_attack_event(track, now + 200)
        self.assertFalse(is_attackable(track, now + 300))
        self.assertEqual(attack_block_reason(track, now + 300), "pending")

    def test_after_attack_reattack_when_not_pending(self) -> None:
        now = 1_000_000
        track = MobTrack.from_discovery(1, 874, 578, 0.65, now_tick=now)
        apply_attack_event(track, now)
        track.pending_result_resolved = True
        track.state = "alive"
        self.assertTrue(is_attackable(track, now + 20_000))

    def test_select_target_rotates_three_targets(self) -> None:
        now = 1_000_000
        tracks = [
            MobTrack.from_discovery(1, 874, 578, 0.65, now_tick=now),
            MobTrack.from_discovery(2, 900, 610, 0.65, now_tick=now),
            MobTrack.from_discovery(3, 820, 520, 0.65, now_tick=now),
        ]
        self.assertEqual(select_target_id(tracks, now), 1)
        self.assertEqual(select_target_id(tracks, now, last_attack_target_id=1), 2)
        self.assertEqual(select_target_id(tracks, now, last_attack_target_id=2), 3)
        self.assertEqual(select_target_id(tracks, now, last_attack_target_id=3), 1)

    def test_select_target_skips_non_attackable_in_rotation(self) -> None:
        now = 1_000_000
        tracks = [
            MobTrack.from_discovery(1, 874, 578, 0.65, now_tick=now),
            MobTrack.from_discovery(2, 900, 610, 0.65, now_tick=now),
            MobTrack.from_discovery(3, 820, 520, 0.65, now_tick=now),
        ]
        apply_attack_event(tracks[1], now + 100)
        self.assertEqual(select_target_id(tracks, now + 200, last_attack_target_id=1), 3)

    def test_attacked_unreachable_marks_unreachable(self) -> None:
        now = 1_000_000
        track = MobTrack.from_discovery(1, 874, 578, 0.65, now_tick=now)
        apply_attack_event(track, now + 100)
        # First "unreachable" after attack marks unreachable (keeps track)
        kept = apply_state_observation(track, StateObservation(1, "unreachable"), now + 200)
        self.assertTrue(kept, "attacked + unreachable should keep track")
        self.assertEqual(track.state, "unreachable")

    def test_attacked_unreachable_marks_after_alive(self) -> None:
        now = 1_000_000
        track = MobTrack.from_discovery(1, 874, 578, 0.65, now_tick=now)
        apply_attack_event(track, now + 100)
        track.pending_result_resolved = True
        track.state = "alive"
        # First "unreachable" after attack marks unreachable (keeps track)
        kept = apply_state_observation(
            track,
            StateObservation(1, "unreachable"),
            now + 200,
        )
        self.assertTrue(kept, "attacked + unreachable should keep track")
        self.assertEqual(track.state, "unreachable")

    def test_pending_timeout_clears_pending_without_alive_state(self) -> None:
        now = 1_000_000
        track = MobTrack.from_discovery(1, 874, 578, 0.65, now_tick=now)
        apply_attack_event(track, now + 100)
        self.assertTrue(is_pending(track, now + 200))
        self.assertFalse(is_pending(track, now + 100 + HUNT_ATTACK_RESULT_WINDOW_MS + 1))
        self.assertEqual(track.state, "alive")
        self.assertTrue(track.pending_result_resolved)

    def test_all_alive_tracks_included_in_state_requests(self) -> None:
        now = 1_000_000
        fresh = MobTrack.from_discovery(1, 874, 578, 0.65, now_tick=now, discovery_scale=0.9)
        attacked = MobTrack.from_discovery(2, 757, 482, 0.65, now_tick=now, discovery_scale=0.9)
        apply_attack_event(attacked, now + 100)
        reqs = collect_state_requests([fresh, attacked])
        self.assertEqual([r["id"] for r in reqs], [2, 1])
        self.assertTrue(is_attackable(fresh, now + 100))

    def test_discovery_scale_used_for_state_request(self) -> None:
        track = MobTrack.from_discovery(1, 100, 200, 0.6, now_tick=0, discovery_scale=0.9)
        self.assertEqual(state_request_scale(track, session_scale_hint=0.75), 0.9)
        track.discovery_scale = 0
        track.candidate_scale = 0
        self.assertEqual(state_request_scale(track, session_scale_hint=0.85), 0.85)

    def test_state_alive_updates_coords_when_applied_before_first_attack(self) -> None:
        now = 1_000_000
        track = MobTrack.from_discovery(1, 874, 578, 0.65, now_tick=now, discovery_scale=0.9)
        apply_state_observation(
            track,
            StateObservation(1, "alive", x=900, y=610, confidence=0.71),
            now + 100,
        )
        self.assertEqual(track.x, 900)
        self.assertEqual(track.y, 610)
        self.assertEqual(track.attack_count, 0)
        self.assertTrue(is_attackable(track, now + 100))
        self.assertEqual(select_target_id([track], now + 100, roi_center=(960, 540)), 1)


class HuntPipelineIntegrationTests(unittest.TestCase):
    """Discovery + state vision + track rules — reproduces live-session failure modes."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.base_config = load_simple_config()
        cls.fixture_dir = MOB_REC / "test-fixtures" / "game-screenshots"
        cls.frame = cv2.imread(str(cls.fixture_dir / "333.png"), cv2.IMREAD_COLOR)
        if cls.frame is None:
            raise unittest.SkipTest("fixture 333.png missing")
        cls.roi = playfield_roi(cls.frame)

    def _detector_at_discovery_scale(self) -> SimpleMobDetector:
        calibrated = apply_scale_calibration(self.base_config, (0.82, 0.98), True)
        detector = SimpleMobDetector(ROOT, calibrated)
        detector.apply_runtime_config(calibrated)
        return detector

    def test_discovery_to_attackable_without_waiting_for_state(self) -> None:
        detector = self._detector_at_discovery_scale()
        discovery = detector.detect(self.roi, "horn")
        living = [c for c in discovery.accepted if not c.is_dead]
        self.assertGreater(len(living), 0)

        anchor = living[0]
        now = 3_182_750_000
        track = MobTrack.from_discovery(
            1,
            anchor.center_x,
            anchor.center_y,
            anchor.final_score,
            now_tick=now,
            discovery_scale=anchor.candidate_scale,
        )
        self.assertTrue(is_attackable(track, now))
        self.assertEqual(select_target_id([track], now), 1)

    def test_live_session_timeline_unreachable_ignored_still_attacks(self) -> None:
        """Replays log pattern: create @19:54:30, unreachable ignored @19:54:37 — must stay attackable."""
        detector = self._detector_at_discovery_scale()
        discovery = detector.detect(self.roi, "horn")
        living = [c for c in discovery.accepted if not c.is_dead]
        self.assertGreater(len(living), 0)
        anchor = living[0]

        t_create = 0
        t_unreachable = 7_000
        track = MobTrack.from_discovery(
            1,
            anchor.center_x,
            anchor.center_y,
            0.65,
            now_tick=t_create,
            discovery_scale=anchor.candidate_scale,
        )

        state_req = {
            "trackId": 1,
            "x": track.x,
            "y": track.y,
            "scale": state_request_scale(track),
        }
        updates = evaluate_track_states(detector, self.roi, "horn", [state_req])
        self.assertEqual(len(updates), 1)

        obs = updates[0]
        apply_state_observation(
            track,
            StateObservation(1, obs["state"], x=obs.get("x", 0), y=obs.get("y", 0)),
            t_create + 2_000,
        )

        apply_state_observation(track, StateObservation(1, "unreachable"), t_unreachable)
        self.assertEqual(track.state, "alive")
        self.assertTrue(
            is_attackable(track, t_unreachable),
            "ignored unreachable must not prevent first attack",
        )
        self.assertEqual(select_target_id([track], t_unreachable), 1)

    def test_state_alive_then_multi_tick_attackable_after_first_attack(self) -> None:
        detector = self._detector_at_discovery_scale()
        discovery = detector.detect(self.roi, "horn")
        living = [c for c in discovery.accepted if not c.is_dead]
        anchor = living[0]

        now = 100_000
        track = MobTrack.from_discovery(
            1,
            anchor.center_x,
            anchor.center_y,
            anchor.final_score,
            now_tick=now,
            discovery_scale=anchor.candidate_scale,
        )
        state_track = {
            "trackId": 1,
            "x": track.x,
            "y": track.y,
            "scale": anchor.candidate_scale,
        }

        for tick in range(5):
            at = now + tick * 2_000
            updates = evaluate_track_states(detector, self.roi, "horn", [state_track])
            obs = updates[0]
            if obs["state"] == "alive":
                apply_state_observation(
                    track,
                    StateObservation(1, "alive", x=obs["x"], y=obs["y"], confidence=obs["confidence"]),
                    at,
                )
                state_track["x"] = obs["x"] - 0  # roi-local in test (no offset)
                state_track["y"] = obs["y"]
            elif obs["state"] == "unreachable":
                apply_state_observation(track, StateObservation(1, "unreachable"), at)

            if track.attack_count == 0:
                self.assertTrue(
                    is_attackable(track, at),
                    f"tick={tick} state={obs['state']} must stay attackable before first attack",
                )

        apply_attack_event(track, now + 20_000)
        self.assertFalse(is_attackable(track, now + 20_100))

        apply_state_observation(
            track,
            StateObservation(1, "alive", x=track.x, y=track.y, confidence=0.7),
            now + 21_000,
        )
        track.pending_result_resolved = True
        track.state = "alive"
        self.assertTrue(is_attackable(track, now + 21_000))


if __name__ == "__main__":
    unittest.main()
