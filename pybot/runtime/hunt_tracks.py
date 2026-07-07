"""Thread-safe MobTrack store"""

from __future__ import annotations

import copy
import threading
import time
from dataclasses import dataclass

from pybot.runtime._mob_rec_path import import_hunt_track_rules
_hunt = import_hunt_track_rules()

DiscoveryDetection = _hunt.DiscoveryDetection
MobTrack = _hunt.MobTrack
ReconcileSummary = _hunt.ReconcileSummary
apply_attack_event = _hunt.apply_attack_event
apply_track_observation = _hunt.apply_track_observation
is_alive = _hunt.is_alive
is_track_lost = _hunt.is_track_lost

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


@dataclass(frozen=True)
class AreaClearStatus:
    clear: bool
    reason: str
    alive_count: int


class HuntTracks:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._tracks: list[MobTrack] = []
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
            self._area_epoch += 1
            self._tracks = []
            self._next_id = 1
            self._last_reconcile_summary = None

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
        clear = alive == 0
        return AreaClearStatus(
            clear=clear,
            reason="" if clear else "alive_tracks",
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

    def positions_snapshot(self) -> list[tuple[int, int]]:
        """(x, y) of every alive track — sample this when the discovery frame is
        captured so dedup compares detections against same-instant positions."""
        with self._lock:
            return [(t.x, t.y) for t in self._tracks if is_alive(t)]

    def reconcile_detections(
        self,
        detections: list[DiscoveryDetection],
        *,
        mob_name: str = "",
        now_tick: int | None = None,
        existing_positions: list[tuple[int, int]] | None = None,
    ) -> ReconcileSummary:
        """Discovery step: create tracks for new mobs only (never updates/removes).

        ``existing_positions`` are the known-object positions at frame-capture
        time. When omitted, the current live positions are used (callers that
        don't run tracking concurrently, e.g. tests).
        """
        tick = now_tick if now_tick is not None else monotonic_ms()
        with self._lock:
            positions = (
                existing_positions
                if existing_positions is not None
                else [(t.x, t.y) for t in self._tracks if is_alive(t)]
            )
            summary = TrackReconciler.reconcile(
                self._tracks,
                detections,
                positions,
                mob_name=mob_name,
                now_tick=tick,
                create_track_fn=self._create_track_locked,
            )
            self._last_reconcile_summary = summary
            return summary

    def apply_tracking(
        self,
        results,
        *,
        now_tick: int | None = None,
    ) -> list[int]:
        """Tracking step: refresh coordinates from LocalTracker and drop lost tracks.

        ``results`` is any iterable of objects exposing ``track_id``, ``found``,
        ``x``, ``y`` and ``confidence`` (e.g. ``LocalTrackResult``). Returns the
        IDs of tracks removed because they were missed too many times.
        """
        tick = now_tick if now_tick is not None else monotonic_ms()
        with self._lock:
            for result in results:
                track = self._get_track_by_id_locked(result.track_id)
                if track is None:
                    continue
                apply_track_observation(
                    track,
                    found=result.found,
                    x=result.x,
                    y=result.y,
                    confidence=result.confidence,
                    now_tick=tick,
                )
            lost_ids = [t.id for t in self._tracks if is_track_lost(t)]
            if lost_ids:
                self._remove_tracks_locked(set(lost_ids))
            return lost_ids

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
        )

    def tracks_for_policy(self, now_tick: int | None = None) -> list[MobTrack]:
        tick = now_tick if now_tick is not None else monotonic_ms()
        with self._lock:
            return copy.deepcopy(self._tracks)

    def copy_tracks_for_tests(self) -> list[MobTrack]:
        return self.tracks_for_policy()
