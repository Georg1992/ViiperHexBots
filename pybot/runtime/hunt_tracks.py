"""Thread-safe MobTrack store"""

from __future__ import annotations

import copy
import threading
import time
from dataclasses import dataclass

from pybot.recognition.rules import (
    DiscoveryDetection,
    MobTrack,
    ReconcileSummary,
    apply_attack_event,
    apply_discovery_reanchor,
    apply_movement_observation,
    apply_track_observation,
    clear_discovery_observation,
    has_discovery_observation,
    is_joint_absent_confirmed,
    joint_absent_confirm_ms,
    is_alive,
    mark_discovery_absent,
    movement_thresholds,
)

from pybot.runtime.track_reconciler import TrackReconciler
from pybot.runtime.capture.window_roi import HuntRoi
from pybot.recognition.detector.detector import load_detector_config


def monotonic_ms() -> int:
    return int(time.monotonic() * 1000)


@dataclass(frozen=True)
class MobTrackSnapshot:
    id: int
    x: int
    y: int
    confidence: float
    attack_count: int
    state: str
    mob_name: str
    updated_tick: int
    discovery_scale: float
    candidate_scale: float


@dataclass(frozen=True)
class AreaClearStatus:
    clear: bool
    reason: str
    alive_count: int


class HuntTracks:
    def __init__(
        self,
        detector_config: dict | None = None,
        *,
        skill_delay_ms: int = 5000,
    ) -> None:
        self._lock = threading.RLock()
        self._tracks: list[MobTrack] = []
        self._detector_config_ref = detector_config
        self._skill_delay_ms = max(skill_delay_ms, 1)
        self._next_id = 1
        self._area_epoch = 0
        self._last_reconcile_summary: ReconcileSummary | None = None

    def reset(self) -> None:
        with self._lock:
            self._tracks = []
            self._next_id = 1
            self._last_reconcile_summary = None

    def area_reset(self) -> None:
        with self._lock:
            self._area_reset_locked()

    def try_claim_clear_for_teleport(self) -> bool:
        """Atomically claim an empty area for teleport.

        Returns False if any alive track exists. On True, advances the area
        epoch and clears tracks immediately so a concurrent discovery scan
        cannot create tracks into the area being left.
        """
        with self._lock:
            if any(is_alive(track) for track in self._tracks):
                return False
            self._area_reset_locked()
            return True

    def _area_reset_locked(self) -> None:
        self._area_epoch += 1
        self._tracks = []
        self._next_id = 1
        self._last_reconcile_summary = None

    # Deprecated no-ops kept for API compatibility.
    def mark_attack_pending(self, track_id: int) -> None:  # noqa: ARG002
        pass

    def clear_attack_pending(self, track_id: int) -> None:  # noqa: ARG002
        pass

    @property
    def average_attacks_till_death(self) -> float:
        return 1.0

    @property
    def kill_sample_count(self) -> int:
        return 0

    @property
    def max_attacks_per_mob_before_unreachable(self) -> int:
        return 999_999

    @property
    def area_epoch(self) -> int:
        with self._lock:
            return self._area_epoch

    def get_track_count(self) -> int:
        with self._lock:
            return len(self._tracks)

    def get_alive_count(self, now_tick: int | None = None) -> int:
        with self._lock:
            return sum(1 for track in self._tracks if is_alive(track))

    def has_alive_tracks(self, now_tick: int | None = None) -> bool:
        with self._lock:
            return any(is_alive(track) for track in self._tracks)

    def get_area_clear_candidate(self, now_tick: int | None = None) -> AreaClearStatus:
        with self._lock:
            alive = sum(1 for track in self._tracks if is_alive(track))
        return AreaClearStatus(
            clear=alive == 0,
            reason="" if alive == 0 else "alive_tracks",
            alive_count=alive,
        )

    def get_track_by_id(self, track_id: int) -> MobTrack | None:
        with self._lock:
            for track in self._tracks:
                if track.id == track_id:
                    return track
            return None

    def snapshot_for_track(self, track_id: int, now_tick: int | None = None) -> MobTrackSnapshot | None:
        with self._lock:
            track = self._get_track_by_id_locked(track_id)
            if track is None:
                return None
            return self._to_snapshot(track)

    def snapshot_tracks(self, now_tick: int | None = None) -> list[MobTrackSnapshot]:
        with self._lock:
            return [self._to_snapshot(track) for track in self._tracks]

    def snapshot_alive(self, now_tick: int | None = None) -> list[MobTrackSnapshot]:
        with self._lock:
            return [self._to_snapshot(track) for track in self._tracks if is_alive(track)]

    def apply_attack_event(self, track_id: int, *, now_tick: int | None = None) -> bool:
        tick = now_tick if now_tick is not None else monotonic_ms()
        with self._lock:
            track = self._get_track_by_id_locked(track_id)
            if track is None:
                return False
            apply_attack_event(track, tick)
            return True

    def positions_snapshot(self, now_tick: int | None = None) -> list[tuple[int, int]]:
        with self._lock:
            return [(t.x, t.y) for t in self._tracks if is_alive(t)]

    def alive_track_positions_snapshot(
        self, now_tick: int | None = None
    ) -> list[tuple[int, int, int]]:
        """Alive (track_id, x, y) at one instant for discovery absence matching."""
        with self._lock:
            return [(t.id, t.x, t.y) for t in self._tracks if is_alive(t)]

    def discovery_frame_snapshot(
        self, now_tick: int | None = None
    ) -> tuple[int, list[tuple[int, int]], list[tuple[int, int, int, float]]]:
        """Atomic sample for one discovery capture: epoch + dedup + alive positions.

        Alive entries are ``(track_id, x, y, scale)`` at capture time.
        """
        tick = now_tick if now_tick is not None else monotonic_ms()
        with self._lock:
            alive = [
                (
                    t.id,
                    t.x,
                    t.y,
                    t.discovery_scale if t.discovery_scale > 0 else 1.0,
                )
                for t in self._tracks
                if is_alive(t)
            ]
            return (
                self._area_epoch,
                [(t.x, t.y) for t in self._tracks if is_alive(t)],
                alive,
            )

    def tracking_frame_snapshot(
        self, now_tick: int | None = None
    ) -> tuple[int, list[MobTrack]]:
        """Atomic sample for one tracking pass: epoch + deep-copied alive tracks."""
        with self._lock:
            alive = [copy.deepcopy(t) for t in self._tracks if is_alive(t)]
            return self._area_epoch, alive

    def reconcile_detections(
        self,
        detections: list[DiscoveryDetection],
        *,
        mob_name: str = "",
        now_tick: int | None = None,
        existing_positions: list[tuple[int, int]] | None = None,
        existing_track_positions: list[tuple[int, int, int]] | list[tuple[int, int, int, float]] | None = None,
        area_epoch: int | None = None,
        hunt_roi: HuntRoi | None = None,
    ) -> ReconcileSummary:
        """Discovery step: create new tracks; notify on unmatched tracks.

        Unmatched tracks are marked ``discovery_absent`` so tracking can drop
        them after a sustained local miss (``trackJointAbsentConfirmMs``).
        Discovery never deletes tracks — tracking owns removal.
        """
        tick = now_tick if now_tick is not None else monotonic_ms()
        del hunt_roi
        with self._lock:
            if area_epoch is not None and area_epoch != self._area_epoch:
                empty = ReconcileSummary(
                    tracks_before=len(self._tracks),
                    tracks_after=len(self._tracks),
                    alive_before=sum(1 for t in self._tracks if is_alive(t)),
                    alive_after=sum(1 for t in self._tracks if is_alive(t)),
                    created_ids=[],
                    removed_ids=[],
                    matched_count=0,
                    added_count=0,
                    removed_count=0,
                )
                self._last_reconcile_summary = empty
                return empty
            positions = (
                existing_positions
                if existing_positions is not None
                else [(t.x, t.y) for t in self._tracks if is_alive(t)]
            )
            track_positions = (
                existing_track_positions
                if existing_track_positions is not None
                else [(t.id, t.x, t.y) for t in self._tracks if is_alive(t)]
            )
            summary = TrackReconciler.reconcile(
                self._tracks,
                detections,
                positions,
                mob_name=mob_name,
                now_tick=tick,
                create_track_fn=self._create_track_locked,
                detector_config=self._detector_config_ref,
                existing_track_positions=track_positions,
            )
            unmatched_ids = set(summary.removed_ids or [])
            for track_id in unmatched_ids:
                track = self._get_track_by_id_locked(track_id)
                if track is None:
                    continue
                mark_discovery_absent(track, now_tick=tick)
                clear_discovery_observation(track)
            summary.removed_ids = []
            summary.removed_count = 0
            summary.tracks_after = len(self._tracks)
            summary.alive_after = sum(1 for t in self._tracks if is_alive(t))
            self._last_reconcile_summary = summary
            return summary

    def apply_tracking(
        self,
        results,
        *,
        now_tick: int | None = None,
        area_epoch: int | None = None,
    ) -> list[int]:
        """Refresh coordinates from LocalTracker results (pure tracking only).

        This method never removes tracks. Cleanup is handled by
        ``apply_tracking_cleanup()``.
        """
        tick = now_tick if now_tick is not None else monotonic_ms()
        missed_ids: list[int] = []
        with self._lock:
            if area_epoch is not None and area_epoch != self._area_epoch:
                return []
            for result in results:
                track = self._get_track_by_id_locked(result.track_id)
                if track is None:
                    continue

                if result.found:
                    move_px, stop_px = movement_thresholds(self._detector_config())
                    apply_movement_observation(
                        track,
                        x=result.x,
                        y=result.y,
                        move_threshold_px=move_px,
                        stop_threshold_px=stop_px,
                    )
                    apply_track_observation(
                        track,
                        found=True,
                        x=result.x,
                        y=result.y,
                        confidence=result.confidence,
                        now_tick=tick,
                    )
                    continue
                if has_discovery_observation(track):
                    if apply_discovery_reanchor(track, now_tick=tick):
                        continue
                apply_track_observation(
                    track,
                    found=False,
                    x=result.x,
                    y=result.y,
                    confidence=result.confidence,
                    now_tick=tick,
                )
                track.moving = False
                missed_ids.append(result.track_id)
            return missed_ids

    def apply_tracking_cleanup(
        self,
        *,
        now_tick: int | None = None,
        area_epoch: int | None = None,
    ) -> tuple[list[int], list[int]]:
        """Remove tracks that are jointly absent.

        Returns ``(lost_ids, unreachable_ids)`` — unreachable_ids always empty.
        """
        tick = now_tick if now_tick is not None else monotonic_ms()
        with self._lock:
            if area_epoch is not None and area_epoch != self._area_epoch:
                return [], []
            joint_absent_ids: list[int] = []
            for track in self._tracks:
                if not is_alive(track):
                    continue
                if not track.discovery_absent:
                    continue
                confirm_ms = joint_absent_confirm_ms(self._detector_config())
                if is_joint_absent_confirmed(
                    track, now_tick=tick, confirm_ms=confirm_ms,
                ):
                    joint_absent_ids.append(track.id)

            if joint_absent_ids:
                self._remove_tracks_locked(set(joint_absent_ids))
            return joint_absent_ids, []

    @property
    def last_reconcile_summary(self) -> ReconcileSummary | None:
        with self._lock:
            return self._last_reconcile_summary

    def create_track(
        self,
        mob_name: str,
        x: int,
        y: int,
        confidence: float,
        candidate_scale: float = 0.0,
        *,
        now_tick: int | None = None,
    ) -> MobTrack:
        tick = now_tick if now_tick is not None else monotonic_ms()
        with self._lock:
            return self._create_track_locked(mob_name, x, y, confidence, candidate_scale, tick)

    def _create_track_locked(
        self,
        mob_name: str,
        x: int,
        y: int,
        confidence: float,
        candidate_scale: float,
        now_tick: int,
    ) -> MobTrack:
        track = MobTrack.from_discovery(
            self._next_id,
            x,
            y,
            confidence,
            now_tick=now_tick,
            discovery_scale=candidate_scale,
            mob_name=mob_name,
            area_epoch=self._area_epoch,
        )
        track.attack_anchor_x, track.attack_anchor_y = x, y
        track.attack_count = 0
        track.attack_count_baseline = 0
        self._next_id += 1
        self._tracks.append(track)
        return track

    def _get_track_by_id_locked(self, track_id: int) -> MobTrack | None:
        for track in self._tracks:
            if track.id == track_id:
                return track
        return None

    def _remove_tracks_locked(self, remove_ids: set[int]) -> None:
        if not remove_ids:
            return
        self._tracks = [track for track in self._tracks if track.id not in remove_ids]

    def _detector_config(self) -> dict:
        return self._detector_config_ref if self._detector_config_ref is not None else load_detector_config()

    @staticmethod
    def _to_snapshot(track: MobTrack) -> MobTrackSnapshot:
        return MobTrackSnapshot(
            id=track.id,
            x=track.x,
            y=track.y,
            confidence=track.confidence,
            attack_count=track.attack_count,
            state=track.state,
            mob_name=track.mob_name,
            updated_tick=track.updated_tick,
            discovery_scale=track.discovery_scale,
            candidate_scale=track.candidate_scale,
        )

    def overlay_track_state(self, now_tick: int | None = None) -> tuple[int, list[MobTrackSnapshot]]:
        with self._lock:
            alive = [self._to_snapshot(track) for track in self._tracks if is_alive(track)]
            return len(self._tracks), alive

    def tracks_for_policy(self, now_tick: int | None = None) -> list[MobTrack]:
        with self._lock:
            return copy.deepcopy(
                [t for t in self._tracks if is_alive(t)]
            )
