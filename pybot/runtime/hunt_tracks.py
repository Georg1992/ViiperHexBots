"""Thread-safe MobTrack store"""

from __future__ import annotations

import copy
import threading
import time
from dataclasses import dataclass, replace

from pybot.recognition.rules import (
    DiscoveryDetection,
    MobTrack,
    ReconcileSummary,
    apply_attack_event,
    apply_discovery_match,
    apply_movement_observation,
    apply_opacity_observation,
    apply_track_observation,
    clear_discovery_blob_observation,
    is_alive,
    movement_thresholds,
)

from pybot.runtime.track_reconciler import DiscoveryReconcileResult, TrackReconciler
from pybot.runtime.capture.window_roi import HuntRoi
from pybot.runtime.constants import (
    DISCOVERY_MISS_REMOVE_COUNT,
    IDLE_DEAD_ATTACK_COUNT,
    IDLE_UNREACHABLE_ATTACK_COUNT,
    MELEE_IDLE_GUARD_RADIUS_PX,
)

from pybot.recognition.detector.detector import load_detector_config


def monotonic_ms() -> int:
    return int(time.monotonic() * 1000)


@dataclass(frozen=True)
class OpacityDeathEvent:
    """One track removed by opacity-decay death detection."""

    track_id: int
    x: int
    y: int
    baseline: float
    opacity_score: float
    streak: int


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
    idle_attack_count: int = 0
    was_accessible: bool = False
    discovery_stationary: bool = False
    moving: bool = False


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
        self._discovery_candidates: list[DiscoveryDetection] = []
        # Recent death positions (x, y, removed_tick) — block rediscovery of
        # fading corpse heat until deathRediscoveryCooldownMs elapses.
        self._death_sites: list[tuple[int, int, int]] = []

    def reset(self) -> None:
        with self._lock:
            self._tracks = []
            self._next_id = 1
            self._discovery_candidates = []
            self._death_sites = []

    def area_reset(self) -> None:
        with self._lock:
            self._area_reset_locked()

    def try_claim_clear_for_teleport(self) -> bool:
        """Atomically claim an empty area for teleport.

        Returns False if any alive track or pending discovery candidate
        exists. On True, advances the area epoch and clears tracks so a
        concurrent discovery scan cannot create tracks into the area being
        left.
        """
        with self._lock:
            if any(is_alive(track) for track in self._tracks):
                return False
            if self._discovery_candidates:
                return False
            self._area_reset_locked()
            return True

    def _area_reset_locked(self) -> None:
        self._area_epoch += 1
        self._tracks = []
        self._next_id = 1
        self._discovery_candidates = []
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
        """Return whether the area has no alive tracks or pending candidates.

        Read-only: must not clear pending discovery candidates. Attack's
        no-target path polls this while discovery may have published
        candidates that tracking has not ingested yet — those must block
        area-clear teleport so mobs are not skipped mid-ingest.
        """
        del now_tick
        with self._lock:
            alive = sum(1 for track in self._tracks if is_alive(track))
            pending = len(self._discovery_candidates)
            if len(self._tracks) == 0:
                # No tracks at all — reset ID counter to prevent unbounded
                # growth across many create/remove cycles.
                self._next_id = 1
            if alive > 0:
                return AreaClearStatus(
                    clear=False, reason="alive_tracks", alive_count=alive,
                )
            if pending > 0:
                return AreaClearStatus(
                    clear=False, reason="pending_candidates", alive_count=0,
                )
            return AreaClearStatus(clear=True, reason="", alive_count=0)

    def has_pending_discovery_candidates(self) -> bool:
        """True when discovery published mobs that tracking has not ingested."""
        with self._lock:
            return bool(self._discovery_candidates)

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

    def discovery_frame_snapshot(
        self, now_tick: int | None = None
    ) -> tuple[int, list[tuple[int, int]], list[tuple[int, int, int]]]:
        """Atomic sample for one discovery capture: epoch + dedup + positions.

        Dedup positions are alive tracks only. Death sites are absorbed later
        with ``deathSiteRadiusPx`` (not mixed into track dedup).
        """
        tick = now_tick if now_tick is not None else monotonic_ms()
        with self._lock:
            alive = [t for t in self._tracks if is_alive(t)]
            return (
                self._area_epoch,
                self._dedup_positions_locked(tick, alive=alive),
                [(t.id, t.x, t.y) for t in alive],
            )

    def tracking_frame_snapshot(
        self, now_tick: int | None = None
    ) -> tuple[int, list[MobTrack]]:
        """Atomic sample for one tracking pass: epoch + copied alive tracks.

        ``replace`` is enough — MobTrack fields are scalars/tuples; the
        snapshot is read-only before ``apply_tracking`` writes under the lock.
        """
        with self._lock:
            alive = [replace(t) for t in self._tracks if is_alive(t)]
            return self._area_epoch, alive

    # ── Discovery candidates pipeline ────────────────────────────────────

    def get_and_clear_new_candidates(self) -> list[DiscoveryDetection]:
        """Return and clear the new-mob candidate list for tracking to ingest."""
        with self._lock:
            candidates = self._discovery_candidates
            self._discovery_candidates = []
            return candidates

    def requeue_discovery_candidates(
        self, candidates: list[DiscoveryDetection]
    ) -> None:
        """Put candidates back when tracking could not process them (e.g. empty frame)."""
        if not candidates:
            return
        with self._lock:
            self._merge_candidates_locked(candidates)

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
        1. Outside hunt ROI → removed immediately (no death site).
        2. Two missed discovery scans → removed. If the track was already
           opacity-fading, records a death site; otherwise bookkeeping only.
        3. First miss → ``discovery_miss_count`` += 1 (stays alive one more scan).

        Confirmed death (opacity / idle-dead) uses ``_remove_dead_tracks_locked``
        elsewhere and records death sites.
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
                    death_sites_active=len(self._death_sites),
                )
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
            result: DiscoveryReconcileResult = TrackReconciler.match_and_absent(
                detections,
                positions,
                track_positions,
                detector_config=self._detector_config_ref,
            )

            config = self._detector_config()
            # Reset discovery_miss_count + update discovery_stationary for matches
            for tid, detection in result.matched:
                track = self._get_track_by_id_locked(tid)
                if track is not None:
                    apply_discovery_match(
                        track,
                        now_tick=tick,
                        detection=detection,
                        config=config,
                    )

            # Absorb corpse heat into death sites (larger radius than track
            # dedup). Refresh site position + cooldown while heat remains.
            kept_candidates: list = []
            death_absorbed = 0
            for detection in result.new_candidates:
                if self._absorb_into_death_site_locked(
                    detection.x, detection.y, tick
                ):
                    death_absorbed += 1
                    continue
                kept_candidates.append(detection)

            # Merge with any unconsumed candidates so a back-to-back discovery
            # scan cannot drop detections tracking has not ingested yet.
            self._merge_candidates_locked(kept_candidates)

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

            # Execute removal. Miss removals that were already fading get a
            # death site so corpse heat cannot be rediscovered as a new track.
            if remove_ids:
                fading_ids = {
                    tid
                    for tid in remove_ids
                    if self._track_opacity_fading_locked(tid)
                }
                plain_ids = remove_ids - fading_ids
                if fading_ids:
                    self._remove_dead_tracks_locked(fading_ids, tick)
                if plain_ids:
                    self._remove_tracks_locked(plain_ids)

            self._prune_death_sites_locked(tick)
            alive_after = sum(1 for t in self._tracks if is_alive(t))
            summary = ReconcileSummary(
                tracks_before=alive_after + len(remove_ids),
                tracks_after=alive_after,
                alive_before=alive_after + len(remove_ids),
                alive_after=alive_after,
                created_ids=[],
                removed_ids=sorted(remove_ids),
                removed_out_of_range_ids=sorted(out_of_range),
                removed_discovery_miss_ids=sorted(miss_remove),
                matched_count=result.matched_count + death_absorbed,
                added_count=len(kept_candidates),
                removed_count=len(remove_ids),
                death_sites_active=len(self._death_sites),
            )
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
        - ``remove_ids``: tracks with miss_count >= 2.
        - ``first_miss_ids``: tracks on their first miss (still alive).

        Caller records a death site when the removed track was opacity-fading.
        """
        remove_ids: set[int] = set()
        first_miss_ids: list[int] = []
        for track_id in unmatched_ids:
            track = self._get_track_by_id_locked(track_id)
            if track is None:
                continue
            track.discovery_miss_count += 1
            clear_discovery_blob_observation(track)
            if track.discovery_miss_count >= DISCOVERY_MISS_REMOVE_COUNT:
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
    ) -> tuple[list[int], list[OpacityDeathEvent]]:
        """Refresh coordinates from LocalTracker results.

        Returns ``(missed_ids, opacity_deaths)``.
        - *missed_ids*: tracks not found by the local tracker.
        - *opacity_deaths*: tracks removed by opacity-decay death detection.

        Tracking owns all position writes — discovery only publishes
        candidates; tracking creates tracks on fresh frames.
        """
        tick = now_tick if now_tick is not None else monotonic_ms()
        missed_ids: list[int] = []
        with self._lock:
            if area_epoch is not None and area_epoch != self._area_epoch:
                return [], []
            opacity_deaths: list[OpacityDeathEvent] = []
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
                    baseline = track.opacity_baseline
                    streak = track.opacity_decay_streak
                    if apply_opacity_observation(
                        track,
                        opacity_score=result.opacity_score,
                        config=config,
                    ):
                        # Corpse is at the found coords this frame — record
                        # death site there, not the previous track position.
                        track.x = result.x
                        track.y = result.y
                        opacity_deaths.append(
                            OpacityDeathEvent(
                                track_id=result.track_id,
                                x=result.x,
                                y=result.y,
                                baseline=baseline,
                                opacity_score=float(result.opacity_score),
                                streak=streak + 1,
                            )
                        )
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

            if opacity_deaths:
                self._remove_dead_tracks_locked(
                    {event.track_id for event in opacity_deaths},
                    tick,
                )

            return missed_ids, opacity_deaths

    def evaluate_idle_attack(
        self,
        track_id: int,
        *,
        was_idle: bool | None,
        mob_x: int,
        mob_y: int,
        char_x: int,
        char_y: int,
        now_tick: int | None = None,
    ) -> tuple[str, int]:
        """Check idle-attack death / unreachable conditions.

        Called after each attack. *was_idle*:
        - ``True`` — SP did not change (pre == post, both readable)
        - ``False`` — SP dropped (skill consumed)
        - ``None`` — SP unread / unknown; idle and accessibility state
          are left untouched (must not fake a hit or an idle)

        Two independent paths:

        **Dead** — mob was hittable (SP consumed at least once), discovery
        heat blob is stationary (unchanged across scans), and the next 2
        attacks were idle → track removed + death site.

        **Unreachable** — 5 consecutive idle attacks on any track → track
        removed + death site (same removal as dead). Catches never-hittable
        mobs and accessible mobs that moved behind walls.

        The melee-range guard (150 px) prevents false positives when the
        character is sitting on the mob and auto-attacks are hitting.

        Returns ``(action, idle_count)`` where *action* is one of
        ``"none"``, ``"dead"``, or ``"unreachable"``.
        """
        with self._lock:
            track = self._get_track_by_id_locked(track_id)
            if track is None:
                return "none", 0

            # SP unknown — do not invent idle or accessibility.
            if was_idle is None:
                return "none", track.idle_attack_count

            if was_idle:
                # Melee auto-attacks often consume no SP; do not treat idle
                # skill presses as death/unreachable while sitting on the mob.
                dx = mob_x - char_x
                dy = mob_y - char_y
                if (dx * dx + dy * dy) <= (
                    MELEE_IDLE_GUARD_RADIUS_PX * MELEE_IDLE_GUARD_RADIUS_PX
                ):
                    return "none", track.idle_attack_count

                if track.was_accessible:
                    # Path 1: hittable + discovery-stationary + not moving
                    # + N idle attacks = dead.
                    if track.moving or not track.discovery_stationary:
                        track.idle_attack_count = 0
                        return "none", 0

                    track.idle_attack_count += 1
                    if track.idle_attack_count >= IDLE_DEAD_ATTACK_COUNT:
                        tick = now_tick if now_tick is not None else monotonic_ms()
                        self._remove_dead_tracks_locked({track_id}, tick)
                        return "dead", track.idle_attack_count

                    return "none", track.idle_attack_count
                else:
                    # Path 2: never confirmed hit + N idle attacks = unreachable.
                    track.idle_attack_count += 1
                    if track.idle_attack_count >= IDLE_UNREACHABLE_ATTACK_COUNT:
                        tick = now_tick if now_tick is not None else monotonic_ms()
                        self._remove_dead_tracks_locked({track_id}, tick)
                        return "unreachable", track.idle_attack_count

                    return "none", track.idle_attack_count

            # SP consumed → real attack → mob is hittable
            track.was_accessible = True
            track.idle_attack_count = 0
            return "none", 0

    def create_track(
        self,
        mob_name: str,
        x: int,
        y: int,
        confidence: float,
        candidate_scale: float = 0.0,
        *,
        now_tick: int | None = None,
        area_epoch: int | None = None,
    ) -> MobTrack | None:
        tick = now_tick if now_tick is not None else monotonic_ms()
        with self._lock:
            if area_epoch is not None and area_epoch != self._area_epoch:
                return None
            return self._create_track_locked(
                mob_name, x, y, confidence, candidate_scale, tick
            )

    def _track_opacity_fading_locked(self, track_id: int) -> bool:
        track = self._get_track_by_id_locked(track_id)
        return track is not None and track.opacity_decay_streak > 0

    def _merge_candidates_locked(
        self, new_candidates: list[DiscoveryDetection]
    ) -> None:
        """Append *new_candidates* onto the pending queue, deduped by radius."""
        if not new_candidates and not self._discovery_candidates:
            return
        config = self._detector_config()
        dedup_radius = int(config["trackDedupRadiusPx"])
        radius_sq = dedup_radius * dedup_radius
        merged: list[DiscoveryDetection] = list(new_candidates)
        known = [(c.x, c.y) for c in merged]
        for prior in self._discovery_candidates:
            duplicate = False
            for kx, ky in known:
                dx = prior.x - kx
                dy = prior.y - ky
                if (dx * dx + dy * dy) <= radius_sq:
                    duplicate = True
                    break
            if not duplicate:
                merged.append(prior)
                known.append((prior.x, prior.y))
        self._discovery_candidates = merged

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

    def _remove_dead_tracks_locked(self, remove_ids: set[int], now_tick: int) -> None:
        """Remove confirmed-dead tracks and record death sites for rediscovery block."""
        if not remove_ids:
            return
        for track in self._tracks:
            if track.id in remove_ids:
                self._record_death_site_locked(track.x, track.y, now_tick)
        self._remove_tracks_locked(remove_ids)

    def _death_rediscovery_cooldown_ms(self) -> int:
        return int(self._detector_config()["deathRediscoveryCooldownMs"])

    def _death_site_radius_px(self) -> int:
        return int(self._detector_config()["deathSiteRadiusPx"])

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

    def _absorb_into_death_site_locked(
        self, x: int, y: int, now_tick: int
    ) -> bool:
        """If *(x, y)* is near a death site, refresh that site and return True.

        Follows corpse-heat drift and extends the rediscovery cooldown while
        the corpse remains visible.
        """
        self._prune_death_sites_locked(now_tick)
        radius = self._death_site_radius_px()
        radius_sq = radius * radius
        best_i: int | None = None
        best_d = 0
        for i, (sx, sy, _removed) in enumerate(self._death_sites):
            dx = x - sx
            dy = y - sy
            dist = dx * dx + dy * dy
            if dist <= radius_sq and (best_i is None or dist < best_d):
                best_i = i
                best_d = dist
        if best_i is None:
            return False
        self._death_sites[best_i] = (x, y, now_tick)
        return True

    def _dedup_positions_locked(
        self,
        now_tick: int,
        *,
        alive: list[MobTrack] | None = None,
    ) -> list[tuple[int, int]]:
        """Alive-track positions discovery must treat as already known.

        Death sites are absorbed separately via ``_absorb_into_death_site_locked``
        (larger radius + cooldown refresh).
        """
        del now_tick
        if alive is None:
            alive = [t for t in self._tracks if is_alive(t)]
        return [(t.x, t.y) for t in alive]

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
            idle_attack_count=track.idle_attack_count,
            was_accessible=track.was_accessible,
            discovery_stationary=track.discovery_stationary,
            moving=track.moving,
        )

    def overlay_track_state(
        self, now_tick: int | None = None
    ) -> tuple[int, list[MobTrackSnapshot]]:
        with self._lock:
            alive = [self._to_snapshot(track) for track in self._tracks if is_alive(track)]
            return len(self._tracks), alive

    def tracks_for_policy(self, now_tick: int | None = None) -> list[MobTrack]:
        with self._lock:
            return copy.deepcopy(
                [t for t in self._tracks if is_alive(t)]
            )
