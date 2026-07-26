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
    apply_discovery_match,
    apply_movement_observation,
    apply_opacity_observation,
    apply_track_observation,
    is_alive,
    movement_thresholds,
)

from pybot.runtime.track_reconciler import DiscoveryReconcileResult, TrackReconciler
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
        self._discovery_candidates: list[DiscoveryDetection] = []

    def reset(self) -> None:
        with self._lock:
            self._tracks = []
            self._next_id = 1
            self._last_reconcile_summary = None
            self._discovery_candidates = []

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
        self._discovery_candidates = []

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
            if len(self._tracks) == 0:
                # No tracks at all — reset ID counter to prevent unbounded
                # growth across many create/remove cycles.
                self._next_id = 1
                self._last_reconcile_summary = None
                self._discovery_candidates = []
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
    ) -> tuple[int, list[tuple[int, int]], list[tuple[int, int, int]]]:
        """Atomic sample for one discovery capture: epoch + dedup + positions.

        Dedup positions include unreachable tracks so discovery does not
        re-create tracks for corpses whose sprites still match the detector.
        Track positions (3-tuples) are alive-only — unreachable IDs must
        not participate in discovery absence matching.
        """
        tick = now_tick if now_tick is not None else monotonic_ms()
        with self._lock:
            alive = [t for t in self._tracks if is_alive(t)]
            unreachable = [t for t in self._tracks if t.state == "unreachable"]
            return (
                self._area_epoch,
                [(t.x, t.y) for t in alive] + [(t.x, t.y) for t in unreachable],
                [(t.id, t.x, t.y) for t in alive],
            )

    def tracking_frame_snapshot(
        self, now_tick: int | None = None
    ) -> tuple[int, list[MobTrack]]:
        """Atomic sample for one tracking pass: epoch + deep-copied alive tracks."""
        with self._lock:
            alive = [copy.deepcopy(t) for t in self._tracks if is_alive(t)]
            return self._area_epoch, alive

    # ── Discovery candidates pipeline ────────────────────────────────────

    def get_and_clear_new_candidates(self) -> list[DiscoveryDetection]:
        """Return and clear the new-mob candidate list for tracking to ingest."""
        with self._lock:
            candidates = self._discovery_candidates
            self._discovery_candidates = []
            return candidates

    def process_discovery_scan(
        self,
        detections: list[DiscoveryDetection],
        *,
        mob_name: str = "",
        now_tick: int | None = None,
        existing_positions: list[tuple[int, int]] | None = None,
        existing_track_positions: list[tuple[int, int, int]] | None = None,
        area_epoch: int | None = None,
        hunt_roi: HuntRoi | None = None,
    ) -> ReconcileSummary:
        """Discovery step: match detections, mark absence, evaluate removal factors.

        Does NOT create tracks — publishing new candidates so tracking can
        create them on a fresh frame with exact coordinates.

        After matching detections against known tracks, evaluates all
        removal factors on unmatched tracks:
        1. Outside hunt ROI → removed immediately.
        2. Two missed discovery scans → removed.
        3. First miss → marked discovery_absent (stays alive).

        Add new removal factors as separate ``_evaluate_*`` methods and
        call them here in the ``remove_ids`` collection block.
        """
        tick = now_tick if now_tick is not None else monotonic_ms()
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
                else [
                    (t.x, t.y) for t in self._tracks
                    if is_alive(t) or t.state == "unreachable"
                ]
            )
            track_positions = (
                existing_track_positions
                if existing_track_positions is not None
                else [(t.id, t.x, t.y) for t in self._tracks if is_alive(t)]
            )
            result: DiscoveryReconcileResult = TrackReconciler.match_and_absent(
                detections,
                positions,
                track_positions,
                detector_config=self._detector_config_ref,
            )

            # Reset discovery_miss_count for matched tracks
            for tid in result.matched_ids:
                track = self._get_track_by_id_locked(tid)
                if track is not None:
                    apply_discovery_match(track, now_tick=tick)

            # Publish new candidates for tracking to create on fresh frame
            self._discovery_candidates = list(result.new_candidates)

            unmatched_ids = set(result.removed_ids)

            # ── Collect removal reasons ──────────────────────────────────
            # Each factor is a self-contained evaluation. Add new factors
            # by calling a new _evaluate_* method here.
            remove_ids: set[int] = set()

            # Factor 1: Tracks that left the hunt ROI
            out_of_range = self._evaluate_out_of_range_removal(
                unmatched_ids, hunt_roi, track_positions,
            )
            remove_ids.update(out_of_range)

            # Factor 2: Tracks missed by discovery 2+ scans in a row
            # Only the remaining in-range tracks are evaluated — out-of-range
            # tracks were already handled by Factor 1.
            remaining_ids = unmatched_ids - out_of_range
            miss_remove, _first_miss = self._evaluate_discovery_miss_removal(
                remaining_ids,
            )
            remove_ids.update(miss_remove)

            # Execute removal
            if remove_ids:
                self._remove_tracks_locked(remove_ids)

            alive_after = sum(1 for t in self._tracks if is_alive(t))
            summary = ReconcileSummary(
                tracks_before=alive_after + len(remove_ids),
                tracks_after=alive_after,
                alive_before=alive_after + len(remove_ids),
                alive_after=alive_after,
                created_ids=[],
                removed_ids=sorted(remove_ids),
                matched_count=result.matched_count,
                added_count=len(result.new_candidates),
                removed_count=len(remove_ids),
            )
            self._last_reconcile_summary = summary
            return summary

    # ── Removal-factor evaluators ────────────────────────────────────────
    # Each method evaluates ONE removal factor and returns a set of track
    # IDs to remove (or (set, list) tuple). Add new factors as new methods.

    def _evaluate_out_of_range_removal(
        self,
        unmatched_ids: set[int],
        hunt_roi: HuntRoi | None,
        track_positions: list[tuple[int, int, int]],
    ) -> set[int]:
        """Factor 1: Remove tracks whose capture-time position is outside the hunt ROI."""
        if hunt_roi is None:
            return set()
        out: set[int] = set()
        for track_id in unmatched_ids:
            tx, ty = self._capture_position(track_id, track_positions)
            if tx is None:
                continue
            if not (
                hunt_roi.x <= tx < hunt_roi.x + hunt_roi.w
                and hunt_roi.y <= ty < hunt_roi.y + hunt_roi.h
            ):
                out.add(track_id)
        return out

    def _evaluate_discovery_miss_removal(
        self,
        unmatched_ids: set[int],
    ) -> tuple[set[int], list[int]]:
        """Factor 2: Remove tracks missed by 2+ consecutive discovery scans.

        Only receives in-range tracks (out-of-range handled by Factor 1).

        Side-effect: increments ``discovery_miss_count``.

        Returns ``(remove_ids, first_miss_ids)`` where:
        - ``remove_ids``: tracks with miss_count >= 2 (to be removed).
        - ``first_miss_ids``: tracks on their first miss (to be marked absent).
        """
        remove_ids: set[int] = set()
        first_miss_ids: list[int] = []
        for track_id in unmatched_ids:
            track = self._get_track_by_id_locked(track_id)
            if track is None:
                continue
            track.discovery_miss_count += 1
            if track.discovery_miss_count >= 2:
                remove_ids.add(track_id)
            else:
                first_miss_ids.append(track_id)
        return remove_ids, first_miss_ids

    def apply_tracking(
        self,
        results,
        *,
        now_tick: int | None = None,
        area_epoch: int | None = None,
    ) -> tuple[list[int], list[int]]:
        """Refresh coordinates from LocalTracker results.

        Returns ``(missed_ids, opacity_dead_ids)``.
        - *missed_ids*: tracks not found by the local tracker.
        - *opacity_dead_ids*: tracks removed by opacity-decay death detection.

        Tracking owns all position writes — discovery only publishes
        candidates; tracking creates tracks on fresh frames.
        """
        tick = now_tick if now_tick is not None else monotonic_ms()
        missed_ids: list[int] = []
        with self._lock:
            if area_epoch is not None and area_epoch != self._area_epoch:
                return [], []
            opacity_dead: set[int] = set()
            config = self._detector_config()
            for result in results:
                track = self._get_track_by_id_locked(result.track_id)
                if track is None:
                    continue

                if result.found:
                    move_px, stop_px = movement_thresholds(config)
                    apply_movement_observation(
                        track,
                        x=result.x,
                        y=result.y,
                        move_threshold_px=move_px,
                        stop_threshold_px=stop_px,
                    )
                    if apply_opacity_observation(
                        track,
                        opacity_score=result.opacity_score,
                        config=config,
                    ):
                        opacity_dead.add(result.track_id)
                        continue

                    apply_track_observation(
                        track,
                        found=True,
                        x=result.x,
                        y=result.y,
                        confidence=result.confidence,
                        now_tick=tick,
                    )
                    continue

                # Tracking miss — coast on velocity, advance lost count.
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

            if opacity_dead:
                self._remove_tracks_locked(opacity_dead)

            return missed_ids, sorted(opacity_dead)

    _IDLE_DEAD_THRESHOLD = 2
    _IDLE_UNREACHABLE_THRESHOLD = 5
    _MELEE_RADIUS_PX = 150

    def evaluate_idle_attack(
        self,
        track_id: int,
        *,
        was_idle: bool,
        mob_x: int,
        mob_y: int,
        char_x: int,
        char_y: int,
    ) -> tuple[str, int]:
        """Check idle-attack death / unreachable conditions.

        Called after each attack. *was_idle* is True when SP did not change
        during this specific attack (pre-attack SP == post-attack SP),
        measured per-attack so other tracks' SP consumption cannot interfere.

        Two independent paths:

        **Dead** — mob was hittable (SP consumed at least once), then
        stopped moving and the next 2 attacks were idle → track is removed.

        **Unreachable** — 5 consecutive idle attacks on any track → marked
        unreachable (red dot, blocks rediscovery).  This catches both
        never-hittable mobs and accessible mobs that moved behind walls.

        The melee-range guard (150 px) prevents false positives when the
        character is sitting on the mob and auto-attacks are hitting.

        Returns ``(action, idle_count)`` where *action* is one of
        ``"none"``, ``"dead"``, or ``"unreachable"``.
        """
        with self._lock:
            track = self._get_track_by_id_locked(track_id)
            if track is None:
                return "none", 0

            if was_idle:
                # Mob must NOT be at melee range ("sitting on character")
                dx = mob_x - char_x
                dy = mob_y - char_y
                if (dx * dx + dy * dy) <= (self._MELEE_RADIUS_PX * self._MELEE_RADIUS_PX):
                    track.idle_attack_count = 0
                    return "none", 0

                track.idle_attack_count += 1

                # Path 1: was accessible + stationary + 2 idle → dead
                if track.was_accessible and not track.moving and track.idle_attack_count >= self._IDLE_DEAD_THRESHOLD:
                    self._remove_tracks_locked({track_id})
                    return "dead", track.idle_attack_count

                # Path 2: 5 idle attacks (any accessibility) → unreachable
                if track.idle_attack_count >= self._IDLE_UNREACHABLE_THRESHOLD:
                    track.state = "unreachable"
                    return "unreachable", track.idle_attack_count

                return "none", track.idle_attack_count

            # SP consumed → real attack → mob is hittable
            track.was_accessible = True
            track.idle_attack_count = 0
            return "none", 0

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
        track.attack_count = 0
        track.attack_count_baseline = 0
        track.idle_attack_count = 0
        track.was_accessible = False
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
    def _capture_position(
        track_id: int,
        track_positions: list[tuple[int, int, int]],
    ) -> tuple[int | None, int | None]:
        """Get capture-time (x, y) for a track from the snapshot positions."""
        for entry in track_positions:
            if entry[0] == track_id:
                return int(entry[1]), int(entry[2])
        return None, None

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

    def overlay_track_state(self, now_tick: int | None = None) -> tuple[int, list[MobTrackSnapshot], list[MobTrackSnapshot]]:
        with self._lock:
            alive = [self._to_snapshot(track) for track in self._tracks if is_alive(track)]
            unreachable = [self._to_snapshot(track) for track in self._tracks if track.state == "unreachable"]
            return len(self._tracks), alive, unreachable

    def tracks_for_policy(self, now_tick: int | None = None) -> list[MobTrack]:
        with self._lock:
            return copy.deepcopy(
                [t for t in self._tracks if is_alive(t)]
            )
