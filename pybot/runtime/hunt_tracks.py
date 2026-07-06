"""Thread-safe MobTrack store"""

from __future__ import annotations

import copy
import threading
import time
from dataclasses import dataclass

from pybot.runtime._mob_rec_path import import_hunt_track_rules
_hunt = import_hunt_track_rules()

DiscoveryDetection = _hunt.DiscoveryDetection
LocalTrackObservation = _hunt.LocalTrackObservation
MobTrack = _hunt.MobTrack
ReconcileSummary = _hunt.ReconcileSummary
StateObservation = _hunt.StateObservation
apply_attack_event = _hunt.apply_attack_event
apply_local_track_observation = _hunt.apply_local_track_observation
apply_state_observation = _hunt.apply_state_observation
collect_local_track_requests = _hunt.collect_local_track_requests
collect_state_requests = _hunt.collect_state_requests
is_alive = _hunt.is_alive
is_attackable = _hunt.is_attackable
is_pending = _hunt.is_pending
select_state_confirm_track_id = _hunt.select_state_confirm_track_id
was_attacked = _hunt.was_attacked

from pybot.runtime.track_reconciler import TrackReconciler


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
    pending_result_until_tick: int
    pending_result_resolved: bool


@dataclass(frozen=True)
class AreaClearStatus:
    clear: bool
    reason: str
    alive_or_pending_count: int
    attackable_count: int


class HuntTracks:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._tracks: list[MobTrack] = []
        self._next_id = 1
        self._scan_id = 0
        self._area_epoch = 0
        self._roi_center_x = 0
        self._roi_center_y = 0
        self._last_reconcile_summary: ReconcileSummary | None = None

    def reset(self) -> None:
        with self._lock:
            self._tracks = []
            self._next_id = 1
            self._scan_id = 0
            self._last_reconcile_summary = None

    def area_reset(self) -> None:
        with self._lock:
            self._area_epoch += 1
            self._tracks = []
            self._next_id = 1
            self._scan_id = 0
            self._last_reconcile_summary = None

    @property
    def area_epoch(self) -> int:
        with self._lock:
            return self._area_epoch

    def set_roi_center(self, center_x: int, center_y: int) -> None:
        with self._lock:
            self._roi_center_x = center_x
            self._roi_center_y = center_y

    def get_track_count(self) -> int:
        with self._lock:
            return len(self._tracks)

    def get_alive_or_pending_count(self, now_tick: int | None = None) -> int:
        tick = now_tick if now_tick is not None else monotonic_ms()
        with self._lock:
            return sum(1 for track in self._tracks if is_alive(track) or is_pending(track, tick))

    def get_attackable_count(self, now_tick: int | None = None) -> int:
        tick = now_tick if now_tick is not None else monotonic_ms()
        with self._lock:
            return sum(1 for track in self._tracks if is_attackable(track, tick))

    def has_known_targets(self, now_tick: int | None = None) -> bool:
        tick = now_tick if now_tick is not None else monotonic_ms()
        with self._lock:
            return any(is_alive(track) for track in self._tracks)

    def has_attackable_tracks(self, now_tick: int | None = None) -> bool:
        return self.get_attackable_count(now_tick) > 0

    def get_area_clear_candidate(self, now_tick: int | None = None) -> AreaClearStatus:
        tick = now_tick if now_tick is not None else monotonic_ms()
        alive_or_pending = self.get_alive_or_pending_count(tick)
        attackable = self.get_attackable_count(tick)
        clear = alive_or_pending == 0
        reason = "" if clear else "alive_or_pending_tracks"
        return AreaClearStatus(
            clear=clear,
            reason=reason,
            alive_or_pending_count=alive_or_pending,
            attackable_count=attackable,
        )

    def get_track_by_id(self, track_id: int) -> MobTrack | None:
        with self._lock:
            for track in self._tracks:
                if track.id == track_id:
                    return track
            return None

    def get_coord_age_ms(self, track_id: int, now_tick: int | None = None) -> int:
        tick = now_tick if now_tick is not None else monotonic_ms()
        with self._lock:
            track = self._get_track_by_id_locked(track_id)
            if track is None:
                return 999_999
            return tick - track.updated_tick

    def snapshot_for_track(self, track_id: int, now_tick: int | None = None) -> MobTrackSnapshot | None:
        tick = now_tick if now_tick is not None else monotonic_ms()
        with self._lock:
            track = self._get_track_by_id_locked(track_id)
            if track is None:
                return None
            is_pending(track, tick)
            return self._to_snapshot(track)

    def snapshot_tracks(self, now_tick: int | None = None) -> list[MobTrackSnapshot]:
        tick = now_tick if now_tick is not None else monotonic_ms()
        with self._lock:
            for track in self._tracks:
                is_pending(track, tick)
            return [self._to_snapshot(track) for track in self._tracks]

    def snapshot_attackable(self, now_tick: int | None = None) -> list[MobTrackSnapshot]:
        tick = now_tick if now_tick is not None else monotonic_ms()
        with self._lock:
            snapshots: list[MobTrackSnapshot] = []
            for track in self._tracks:
                if is_attackable(track, tick):
                    snapshots.append(self._to_snapshot(track))
            return snapshots

    def collect_local_track_requests(
        self,
        *,
        session_scale_hint: float = 0.0,
        now_tick: int | None = None,
    ) -> list[dict]:
        tick = now_tick if now_tick is not None else monotonic_ms()
        with self._lock:
            for track in self._tracks:
                is_pending(track, tick)
            return collect_local_track_requests(
                list(self._tracks),
                session_scale_hint=session_scale_hint,
            )

    def select_state_confirm_track_id(self, now_tick: int | None = None) -> int:
        tick = now_tick if now_tick is not None else monotonic_ms()
        with self._lock:
            for track in self._tracks:
                is_pending(track, tick)
            return select_state_confirm_track_id(list(self._tracks), tick)

    def apply_local_track_observations(
        self,
        observations: list[LocalTrackObservation],
        *,
        now_tick: int | None = None,
    ) -> list[int]:
        tick = now_tick if now_tick is not None else monotonic_ms()
        needs_confirm: list[int] = []
        with self._lock:
            for observation in observations:
                track = self._get_track_by_id_locked(observation.id)
                if track is None:
                    continue
                if apply_local_track_observation(track, observation, tick):
                    needs_confirm.append(track.id)
        return needs_confirm

    def clear_local_track_miss(self, track_id: int) -> None:
        with self._lock:
            track = self._get_track_by_id_locked(track_id)
            if track is not None:
                track.local_track_miss_count = 0

    def collect_state_requests(
        self,
        *,
        session_scale_hint: float = 0.0,
        now_tick: int | None = None,
    ) -> list[dict]:
        tick = now_tick if now_tick is not None else monotonic_ms()
        with self._lock:
            for track in self._tracks:
                is_pending(track, tick)
            return collect_state_requests(
                list(self._tracks),
                session_scale_hint=session_scale_hint,
            )

    def apply_state_observations(
        self,
        observations: list[StateObservation],
        *,
        now_tick: int | None = None,
    ) -> None:
        tick = now_tick if now_tick is not None else monotonic_ms()
        with self._lock:
            for observation in observations:
                self._apply_state_observation_locked(observation, tick)

    def apply_attack_event(self, track_id: int, *, now_tick: int | None = None) -> bool:
        tick = now_tick if now_tick is not None else monotonic_ms()
        with self._lock:
            track = self._get_track_by_id_locked(track_id)
            if track is None:
                return False
            apply_attack_event(track, tick)
            return True

    def reconcile_detections(
        self,
        detections: list[DiscoveryDetection],
        *,
        mob_name: str = "",
        now_tick: int | None = None,
    ) -> ReconcileSummary:
        tick = now_tick if now_tick is not None else monotonic_ms()
        with self._lock:
            self._scan_id += 1

            summary = TrackReconciler.reconcile(
                self._tracks,
                detections,
                mob_name=mob_name,
                now_tick=tick,
                create_track_fn=self._create_track_locked,
            )
            self._last_reconcile_summary = summary
            return summary

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
        self._next_id += 1
        self._tracks.append(track)
        return track

    def _apply_state_observation_locked(
        self,
        observation: StateObservation,
        now_tick: int,
    ) -> None:
        track = self._get_track_by_id_locked(observation.id)
        if track is None:
            return
        if observation.state == "dead" and was_attacked(track):
            apply_state_observation(track, observation, now_tick)
            self._remove_tracks_locked({track.id})
            return
        kept = apply_state_observation(track, observation, now_tick)
        if not kept:
            self._remove_tracks_locked({track.id})

    def _get_track_by_id_locked(self, track_id: int) -> MobTrack | None:
        for track in self._tracks:
            if track.id == track_id:
                return track
        return None

    def _remove_tracks_locked(self, remove_ids: set[int]) -> None:
        if not remove_ids:
            return
        self._tracks = [track for track in self._tracks if track.id not in remove_ids]

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
            pending_result_until_tick=track.pending_result_until_tick,
            pending_result_resolved=track.pending_result_resolved,
        )

    def tracks_for_policy(self, now_tick: int | None = None) -> list[MobTrack]:
        tick = now_tick if now_tick is not None else monotonic_ms()
        with self._lock:
            for track in self._tracks:
                is_pending(track, tick)
            return copy.deepcopy(self._tracks)

    def copy_tracks_for_tests(self) -> list[MobTrack]:
        return self.tracks_for_policy()
