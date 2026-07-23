"""Thread-safe MobTrack store"""

from __future__ import annotations

import copy
import threading
import time
from collections import deque
from dataclasses import dataclass

from pybot.recognition.rules import (
    DiscoveryDetection,
    MobTrack,
    ReconcileSummary,
    apply_attack_event,
    apply_discovery_reanchor,
    apply_movement_observation,
    apply_opacity_observation,
    apply_track_observation,
    clear_discovery_observation,
    death_movement_thresholds,
    has_discovery_observation,
    joint_absent_confirm_ms,
    is_alive,
    is_track_unreachable_by_attacks,
    mob_attack_anchor_key,
    max_attacks_per_mob_before_unreachable,
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
        self._removed_sites: list[tuple[int, int, int]] = []
        self._attacks_by_anchor: dict[tuple[int, int], int] = {}
        self._pending_attack_track_ids: set[int] = set()
        self._kill_history: deque[int] = deque(
            maxlen=int(self._detector_config()["attacksTillDeathHistoryWindow"])
        )

    def reset(self) -> None:
        with self._lock:
            self._tracks = []
            self._next_id = 1
            self._last_reconcile_summary = None
            self._removed_sites = []
            self._attacks_by_anchor = {}
            self._pending_attack_track_ids = set()
            self._kill_history.clear()

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
        self._removed_sites = []
        self._attacks_by_anchor = {}
        self._pending_attack_track_ids = set()

    def mark_attack_pending(self, track_id: int) -> None:
        """Mark a track as having an in-flight attack click (not yet recorded)."""
        with self._lock:
            self._pending_attack_track_ids.add(track_id)

    def clear_attack_pending(self, track_id: int) -> None:
        """Drop in-flight attack mark (e.g. input failed before the click landed)."""
        with self._lock:
            self._pending_attack_track_ids.discard(track_id)

    @property
    def area_epoch(self) -> int:
        with self._lock:
            return self._area_epoch

    def record_kill(self, attack_count: int) -> None:
        """Record how many attacks a mob needed before death."""
        with self._lock:
            self._record_kill_locked(attack_count)

    @property
    def average_attacks_till_death(self) -> float:
        with self._lock:
            return self._session_average_attacks_till_death_locked()

    @property
    def kill_sample_count(self) -> int:
        with self._lock:
            return len(self._kill_history)

    @property
    def max_attacks_per_mob_before_unreachable(self) -> int:
        with self._lock:
            return self._max_attacks_per_mob_before_unreachable_locked()

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
        """Record an attack on one mob track.

        Returns False when the track exceeded the unreachable attack budget.
        Removal is performed by the tracking layer on its next tick.
        """
        tick = now_tick if now_tick is not None else monotonic_ms()
        with self._lock:
            self._pending_attack_track_ids.discard(track_id)
            track = self._get_track_by_id_locked(track_id)
            if track is None:
                return False
            apply_attack_event(track, tick)
            anchor = (track.attack_anchor_x, track.attack_anchor_y)
            self._attacks_by_anchor[anchor] = track.attack_count
            limit = self._max_attacks_per_mob_before_unreachable_locked()
            return not is_track_unreachable_by_attacks(track, limit)

    def positions_snapshot(self, now_tick: int | None = None) -> list[tuple[int, int]]:
        """Positions discovery should treat as already known (alive + recent removals).

        Includes alive tracks and recently removed sites (opacity death or
        unreachable) so discovery does not immediately recreate a corpse.
        """
        tick = now_tick if now_tick is not None else monotonic_ms()
        with self._lock:
            return self._dedup_positions_locked(tick)

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
                self._dedup_positions_locked(tick),
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
        ``hunt_roi`` is accepted for call-site compatibility; absence marking
        does not depend on it.

        ``existing_positions`` are the known-object positions at frame-capture
        time. When omitted, the current live positions are used (callers that
        don't run tracking concurrently, e.g. tests).

        ``existing_track_positions`` are alive ``(id, x, y[, scale])`` at
        frame-capture time used to decide which tracks were not seen on this scan.

        ``area_epoch`` is the epoch sampled with that frame. If the store's
        epoch has advanced (teleport / area_reset), this is a no-op so
        pre-reset detections cannot spawn tracks into the new area.
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
                else self._dedup_positions_locked(tick)
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
                # Notify tracking only — never delete from discovery.
                track.discovery_absent = True
                clear_discovery_observation(track)
            summary.removed_ids = []
            summary.removed_count = 0
            summary.tracks_after = len(self._tracks)
            summary.alive_after = sum(1 for t in self._tracks if is_alive(t))
            self._last_reconcile_summary = summary
            return summary

    def apply_death_results(
        self,
        death_results: list[tuple[int, float, int, int, bool]],
        *,
        now_tick: int | None = None,
        area_epoch: int | None = None,
    ) -> list[int]:
        """Apply opacity death results. Returns removed dead track ids.

        Each entry is ``(track_id, opacity_baseline, opacity_baseline_samples,
        opacity_decay_streak, dead)`` from ``probe_track_death()``.
        Non-dead tracks get their opacity state updated; dead tracks are
        removed and kill samples recorded.
        """
        tick = now_tick if now_tick is not None else monotonic_ms()
        dead_ids: list[int] = []
        with self._lock:
            if area_epoch is not None and area_epoch != self._area_epoch:
                return []
            for tid, baseline, samples, streak, dead in death_results:
                track = self._get_track_by_id_locked(tid)
                if track is None:
                    continue
                if dead:
                    sample = self._kill_sample_attack_count_locked(track)
                    self._pending_attack_track_ids.discard(tid)
                    self._record_kill_locked(sample)
                    self._record_removed_site_locked(track.x, track.y, tick)
                    dead_ids.append(tid)
                else:
                    apply_opacity_observation(
                        track,
                        opacity_baseline=baseline,
                        opacity_baseline_samples=samples,
                        opacity_decay_streak=streak,
                    )
            if dead_ids:
                self._remove_tracks_locked(set(dead_ids))
            return dead_ids

    def apply_tracking(
        self,
        results,
        *,
        now_tick: int | None = None,
        area_epoch: int | None = None,
    ) -> tuple[list[int], list[int], list[int]]:
        """Tracking step: refresh coordinates from LocalTracker and drop gone tracks.

        ``results`` is any iterable of objects exposing ``track_id``, ``found``,
        ``x``, ``y`` and ``confidence`` (e.g. ``LocalTrackResult``). Returns
        ``(dead_ids, lost_ids, unreachable_ids)`` for tracks removed this tick.
        ``lost_ids`` means joint absence (discovery_absent + sustained local
        miss for ``trackJointAbsentConfirmMs``), not a single-frame miss —
        tracking keeps searching until discovery confirms.

        Death detection is handled separately by ``apply_death_results()`` —
        this method expects results without death flags (from the coords worker).

        ``area_epoch`` is the epoch sampled with the tracking frame. If the store
        advanced (teleport / area_reset) while local follow was running, discard
        the whole batch so stale ids cannot mutate post-reset tracks.
        """
        tick = now_tick if now_tick is not None else monotonic_ms()
        dead_ids: list[int] = []
        with self._lock:
            if area_epoch is not None and area_epoch != self._area_epoch:
                return [], [], []
            joint_absent_ids: list[int] = []
            for result in results:
                track = self._get_track_by_id_locked(result.track_id)
                if track is None:
                    continue
                if getattr(result, "dead", False):
                    # Opacity death (from direct API callers or if coords
                    # worker ever flags death). The death worker uses
                    # apply_death_results() instead.
                    sample = self._kill_sample_attack_count_locked(track)
                    self._pending_attack_track_ids.discard(result.track_id)
                    self._record_kill_locked(sample)
                    self._record_removed_site_locked(result.x, result.y, tick)
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
                        found=True,
                        x=result.x,
                        y=result.y,
                        confidence=result.confidence,
                        now_tick=tick,
                    )
                    if hasattr(result, "opacity_baseline"):
                        apply_opacity_observation(
                            track,
                            opacity_baseline=result.opacity_baseline,
                            opacity_baseline_samples=result.opacity_baseline_samples,
                            opacity_decay_streak=result.opacity_decay_streak,
                        )
                    continue
                if has_discovery_observation(track):
                    # Local miss but discovery still sees the mob — snap once
                    # when drifted from the prior; if already there, fall
                    # through so lost_count advances and discovery is woken.
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
                if hasattr(result, "opacity_baseline"):
                    apply_opacity_observation(
                        track,
                        opacity_baseline=result.opacity_baseline,
                        opacity_baseline_samples=result.opacity_baseline_samples,
                        opacity_decay_streak=result.opacity_decay_streak,
                    )
                if track.discovery_absent:
                    # Discovery unmatched + local miss: keep searching until
                    # the miss has lasted long enough (one wake/scan is not
                    # enough — living movers briefly look absent).
                    last_found = (
                        track.last_found_tick
                        if track.last_found_tick > 0
                        else track.created_tick
                    )
                    confirm_ms = joint_absent_confirm_ms(self._detector_config())
                    if (tick - last_found) >= confirm_ms:
                        joint_absent_ids.append(result.track_id)
            remove_ids = set(dead_ids)
            remove_ids.update(joint_absent_ids)
            # "Lost" for callers = joint absence only. Do not drop on local miss
            # timeout — tracking keeps searching; discovery confirms gone/dead.
            lost_ids = list(joint_absent_ids)
            remove_ids.update(lost_ids)
            unreachable_ids = self._expire_unreachable_locked(
                tick, exclude_ids=remove_ids
            )
            remove_ids.update(unreachable_ids)
            if remove_ids:
                self._remove_tracks_locked(remove_ids)
            return dead_ids, lost_ids, unreachable_ids

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
        anchor = mob_attack_anchor_key(
            x,
            y,
            cell_px=int(self._detector_config()["trackDedupRadiusPx"]),
        )
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
        track.attack_anchor_x, track.attack_anchor_y = anchor
        track.attack_count = self._attacks_by_anchor.get(anchor, 0)
        track.attack_count_baseline = track.attack_count
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

    def _max_attacks_per_mob_before_unreachable_locked(self) -> int:
        return max_attacks_per_mob_before_unreachable(
            average_attacks_till_death=self._session_average_attacks_till_death_locked(),
            skill_delay_ms=self._skill_delay_ms,
        )

    def _session_average_attacks_till_death_locked(self) -> float:
        if not self._kill_history:
            return float(self._detector_config()["defaultAverageAttacksTillDeath"])
        return sum(self._kill_history) / len(self._kill_history)

    def _record_kill_locked(self, attack_count: int) -> None:
        if attack_count <= 0:
            return
        self._kill_history.append(attack_count)

    def _kill_sample_attack_count_locked(self, track: MobTrack) -> int:
        """Attacks this track needed to die, including an in-flight click."""
        attacks_this_life = track.attack_count - track.attack_count_baseline
        if track.id in self._pending_attack_track_ids:
            attacks_this_life = max(attacks_this_life, attacks_this_life + 1)
        return attacks_this_life

    def _removed_site_cooldown_ms(self) -> int:
        return int(self._detector_config()["deathRediscoveryCooldownMs"])

    def _prune_removed_sites_locked(self, now_tick: int) -> None:
        cooldown = self._removed_site_cooldown_ms()
        self._removed_sites = [
            (x, y, removed_tick)
            for x, y, removed_tick in self._removed_sites
            if now_tick - removed_tick <= cooldown
        ]

    def _record_removed_site_locked(self, x: int, y: int, removed_tick: int) -> None:
        self._prune_removed_sites_locked(removed_tick)
        self._removed_sites.append((x, y, removed_tick))

    def _expire_unreachable_locked(
        self,
        now_tick: int,
        *,
        exclude_ids: set[int] | None = None,
    ) -> list[int]:
        """Drop tracks that exceeded the per-mob attack budget without dying.

        Seeds a rediscovery ghost at the last position so discovery does not
        immediately recreate the same site (often a corpse that opacity never
        confirmed). Attack-anchor counts are cleared so a later recreate after
        cooldown starts with a fresh budget.
        """
        skip = exclude_ids or set()
        limit = self._max_attacks_per_mob_before_unreachable_locked()
        unreachable_ids: list[int] = []
        for track in self._tracks:
            if track.id in skip:
                continue
            if not is_track_unreachable_by_attacks(track, limit):
                continue
            anchor = (track.attack_anchor_x, track.attack_anchor_y)
            self._attacks_by_anchor.pop(anchor, None)
            self._pending_attack_track_ids.discard(track.id)
            self._record_removed_site_locked(track.x, track.y, now_tick)
            unreachable_ids.append(track.id)
        return unreachable_ids

    def _dedup_positions_locked(self, now_tick: int) -> list[tuple[int, int]]:
        self._prune_removed_sites_locked(now_tick)
        positions = [(t.x, t.y) for t in self._tracks if is_alive(t)]
        positions.extend((x, y) for x, y, _removed_tick in self._removed_sites)
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
            # Exclude discovery-death notifications — tracking owns the drop,
            # but attack must not keep clicking a corpse while waiting.
            return copy.deepcopy(
                [t for t in self._tracks if is_alive(t)]
            )
