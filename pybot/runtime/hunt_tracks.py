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
    apply_movement_observation,
    apply_opacity_observation,
    apply_track_observation,
    death_movement_thresholds,
    is_alive,
    is_track_lost,
)

from pybot.runtime.track_reconciler import TrackReconciler
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
    def __init__(self, detector_config: dict | None = None) -> None:
        self._lock = threading.RLock()
        self._tracks: list[MobTrack] = []
        self._detector_config_ref = detector_config
        self._next_id = 1
        self._area_epoch = 0
        self._last_reconcile_summary: ReconcileSummary | None = None
        self._death_sites: list[tuple[int, int, int]] = []

    def reset(self) -> None:
        with self._lock:
            self._tracks = []
            self._next_id = 1
            self._last_reconcile_summary = None
            self._death_sites = []

    def area_reset(self) -> None:
        with self._lock:
            self._area_epoch += 1
            self._tracks = []
            self._next_id = 1
            self._last_reconcile_summary = None
            self._death_sites = []

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

    def positions_snapshot(self, now_tick: int | None = None) -> list[tuple[int, int]]:
        """Positions discovery should treat as already known (alive tracks + recent deaths).

        Sample this when the discovery frame is captured so dedup compares detections
        against same-instant positions.
        """
        tick = now_tick if now_tick is not None else monotonic_ms()
        with self._lock:
            return self._dedup_positions_locked(tick)

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
                else self._dedup_positions_locked(tick)
            )
            summary = TrackReconciler.reconcile(
                self._tracks,
                detections,
                positions,
                mob_name=mob_name,
                now_tick=tick,
                create_track_fn=self._create_track_locked,
                detector_config=self._detector_config_ref,
            )
            self._last_reconcile_summary = summary
            return summary

    def apply_tracking(
        self,
        results,
        *,
        now_tick: int | None = None,
    ) -> tuple[list[int], list[int]]:
        """Tracking step: refresh coordinates from LocalTracker and drop lost/dead tracks.

        ``results`` is any iterable of objects exposing ``track_id``, ``found``,
        ``x``, ``y`` and ``confidence`` (e.g. ``LocalTrackResult``). Returns
        ``(dead_ids, lost_ids)`` for tracks removed this tick.
        """
        tick = now_tick if now_tick is not None else monotonic_ms()
        dead_ids: list[int] = []
        with self._lock:
            for result in results:
                track = self._get_track_by_id_locked(result.track_id)
                if track is None:
                    continue
                if getattr(result, "dead", False):
                    self._record_death_site_locked(track.x, track.y, tick)
                    dead_ids.append(result.track_id)
                    continue
                if result.found:
                    move_px, stop_px = death_movement_thresholds(self._detector_config())
                    apply_movement_observation(
                        track,
                        x=result.x,
                        y=result.y,
                        move_threshold_px=move_px,
                        stop_threshold_px=stop_px,
                    )
                apply_track_observation(
                    track,
                    found=result.found,
                    x=result.x,
                    y=result.y,
                    confidence=result.confidence,
                    now_tick=tick,
                )
                # LocalTrackResult carries opacity state; test stubs (_hit) omit it.
                if result.found and hasattr(result, "opacity_baseline"):
                    apply_opacity_observation(
                        track,
                        opacity_baseline=result.opacity_baseline,
                        opacity_baseline_samples=result.opacity_baseline_samples,
                        opacity_decay_streak=result.opacity_decay_streak,
                    )
            remove_ids = set(dead_ids)
            lost_ids = [t.id for t in self._tracks if t.id not in remove_ids and is_track_lost(t)]
            remove_ids.update(lost_ids)
            if remove_ids:
                self._remove_tracks_locked(remove_ids)
            return dead_ids, lost_ids

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

    def _detector_config(self) -> dict:
        return self._detector_config_ref if self._detector_config_ref is not None else load_detector_config()

    def _death_rediscovery_cooldown_ms(self) -> int:
        return int(self._detector_config()["deathRediscoveryCooldownMs"])

    def _prune_death_sites_locked(self, now_tick: int) -> None:
        cooldown = self._death_rediscovery_cooldown_ms()
        self._death_sites = [
            (x, y, removed_tick)
            for x, y, removed_tick in self._death_sites
            if now_tick - removed_tick <= cooldown
        ]

    def _record_death_site_locked(self, x: int, y: int, removed_tick: int) -> None:
        self._prune_death_sites_locked(removed_tick)
        self._death_sites.append((x, y, removed_tick))

    def _dedup_positions_locked(self, now_tick: int) -> list[tuple[int, int]]:
        self._prune_death_sites_locked(now_tick)
        positions = [(t.x, t.y) for t in self._tracks if is_alive(t)]
        positions.extend((x, y) for x, y, _removed_tick in self._death_sites)
        return positions

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
            return copy.deepcopy(self._tracks)
