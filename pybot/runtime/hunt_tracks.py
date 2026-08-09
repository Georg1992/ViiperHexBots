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
from pybot.runtime.death_sites import DeathSiteStore
from pybot.runtime.constants import (
    DISCOVERY_MISS_REMOVE_COUNT,
    IDLE_DEAD_ATTACK_COUNT,
    IDLE_UNREACHABLE_ATTACK_COUNT,
    MELEE_IDLE_GUARD_RADIUS_PX,
)

from pybot.recognition.detector.detector import load_detector_config


def monotonic_ms() -> int:
    return int(time.monotonic() * 1000)


def _within_melee_guard(mx: int, my: int, char_x: int, char_y: int) -> bool:
    """True when a mob is within the character melee occlusion disk."""
    dx = mx - char_x
    dy = my - char_y
    return (dx * dx + dy * dy) <= (
        MELEE_IDLE_GUARD_RADIUS_PX * MELEE_IDLE_GUARD_RADIUS_PX
    )


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
    debuff_applied: bool = False
    area_epoch: int = 0


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
        detector_config = self._detector_config()
        # Corpse-heat suppression is a separate policy store; this aggregate
        # only coordinates it with track mutations under its own lock.
        self._death_site_store = DeathSiteStore(
            radius_px=int(detector_config["deathSiteRadiusPx"]),
            cooldown_ms=int(detector_config["deathRediscoveryCooldownMs"]),
        )

    def reset(self) -> None:
        """Clear all tracks and invalidate any in-flight frame snapshots."""
        with self._lock:
            self._area_epoch += 1
            self._tracks = []
            self._discovery_candidates = []
            self._death_site_store.clear()

    def area_reset(self) -> None:
        with self._lock:
            self._area_reset_locked()

    def can_claim_clear_for_teleport(self) -> bool:
        """Return whether a clear-area teleport is currently admissible.

        This is a read-only check. The teleport controller performs the actual
        area reset only after the input key is accepted, so a rejected key
        cannot discard the current area state.
        """
        with self._lock:
            return not any(is_alive(track) for track in self._tracks) and not self._discovery_candidates

    def try_claim_clear_for_teleport(self) -> bool:
        """Atomically claim an empty area for teleport.

        Kept for callers/tests that need the old claim-and-reset operation.
        Danger/mode teleport paths should use :meth:`can_claim_clear_for_teleport`
        followed by the accepted-input reset owned by ``teleport_once``.
        """
        with self._lock:
            if not self.can_claim_clear_for_teleport():
                return False
            self._area_reset_locked()
            return True

    def _area_reset_locked(self) -> None:
        self._area_epoch += 1
        self._tracks = []
        # Track IDs remain unique across area resets. Attack decisions may be
        # in flight while a teleport clears the old area; reusing an ID could
        # make that stale decision target a new mob in the same lifecycle.
        self._discovery_candidates = []
        self._death_site_store.clear()

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

    def mark_debuff_applied(self, track_id: int) -> bool:
        """Record that the configured per-mob debuff was cast successfully."""
        with self._lock:
            track = self._get_track_by_id_locked(track_id)
            if track is None or not is_alive(track):
                return False
            track.debuff_applied = True
            return True

    def apply_attack_event(self, track_id: int, *, now_tick: int | None = None) -> bool:
        tick = now_tick if now_tick is not None else monotonic_ms()
        with self._lock:
            track = self._get_track_by_id_locked(track_id)
            if track is None:
                return False
            apply_attack_event(track, tick)
            return True

    def perform_if_current(
        self,
        track_id: int,
        expected_epoch: int | None,
        action,
    ) -> bool:
        """Run one short action only while this exact track is still current.

        The track lock covers the final identity check and the input callback,
        closing the check-then-act gap where discovery could remove a target
        between validation and the skill key/click. Callers must keep *action*
        short and must not perform capture or waits inside it.
        """
        with self._lock:
            track = self._get_track_by_id_locked(track_id)
            if track is None:
                return False
            if (
                expected_epoch is not None
                and track.area_epoch != expected_epoch
            ):
                return False
            result = action()
            return result is not False

    def positions_snapshot(self, now_tick: int | None = None) -> list[tuple[int, int]]:
        with self._lock:
            return [(t.x, t.y) for t in self._tracks if is_alive(t)]

    def discovery_frame_snapshot(
        self, now_tick: int | None = None
    ) -> tuple[int, list[tuple[int, int]], list[tuple[int, int, int, float, float, int]]]:
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
                [
                    (
                        t.id,
                        t.x,
                        t.y,
                        float(t.vel_x),
                        float(t.vel_y),
                        int(t.lost_count),
                    )
                    for t in alive
                ],
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

    def consume_reanchor(self, track_id: int, *, expected_epoch: int | None = None) -> tuple[int, int] | None:
        """Consume one Discovery reanchor hint for the current area.

        Discovery publishes only a search proposal. Tracking consumes it before
        its next fresh-frame follow and remains the sole writer of authoritative
        coordinates. Epoch validation prevents an old hint from crossing a
        teleport boundary.
        """
        with self._lock:
            if expected_epoch is not None and expected_epoch != self._area_epoch:
                return None
            track = self._get_track_by_id_locked(track_id)
            if track is None or not is_alive(track):
                return None
            anchor = track.pending_reanchor
            track.pending_reanchor = None
            return anchor

    # ── Discovery candidates pipeline ────────────────────────────────────

    def get_and_clear_new_candidates(self) -> list[DiscoveryDetection]:
        """Return and clear the new-mob candidate list for tracking to ingest."""
        with self._lock:
            candidates = self._discovery_candidates
            self._discovery_candidates = []
            return candidates

    def requeue_discovery_candidates(
        self,
        candidates: list[DiscoveryDetection],
        *,
        expected_epoch: int | None = None,
    ) -> None:
        """Put candidates back only if they still belong to the current area.

        A tracker may pop candidates, then lose the capture race to a danger
        teleport before it can process them. In that case requeueing would
        move old-screen detections into the new screen. An epoch supplied by
        the pop operation makes this handoff fail closed; legacy callers that
        omit it retain the normal current-area behavior.
        """
        if not candidates:
            return
        with self._lock:
            if expected_epoch is not None and expected_epoch != self._area_epoch:
                return
            self._merge_candidates_locked(candidates)

    def process_discovery_scan(
        self,
        detections: list[DiscoveryDetection],
        *,
        mob_name: str = "",
        now_tick: int | None = None,
        existing_positions: list[tuple[int, int]] | None = None,
        existing_track_positions: list[tuple] | None = None,
        area_epoch: int | None = None,
        hunt_roi: HuntRoi | None = None,
    ) -> ReconcileSummary:
        """Discovery step: match detections, mark absence, evaluate removal factors.

        Does NOT create tracks — publishing new candidates so tracking can
        create them on a fresh frame with exact coordinates.

        After matching detections against known tracks, evaluates all
        removal factors on unmatched tracks:
        1. Outside hunt ROI → removed immediately (no death site).
        2. Three missed discovery scans → removed. If the track was already
           opacity-fading, records a death site; otherwise bookkeeping only.
           Misses only accumulate while local tracking does NOT confirm the
           mob on fresh frames: a tracking hit resets ``discovery_miss_count``
           (see ``apply_track_observation``), so a large kiting sprite that
           discovery's silhouette repeatedly fails to extract is never removed
           while tracking still follows it. Misses inside the melee occlusion
           disk around the character (ROI center) additionally do not count
           while ``lost_count == 0``. Once tracking has also lost it, misses
           count normally so corpses under the character are removed.
        3. Earlier misses → ``discovery_miss_count`` += 1 (stays alive).

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
                    death_sites_active=self._death_site_store.active_count(tick),
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
                else [
                    (
                        t.id,
                        t.x,
                        t.y,
                        float(t.vel_x),
                        float(t.vel_y),
                        int(t.lost_count),
                    )
                    for t in self._tracks
                    if is_alive(t)
                ]
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
                        tracking_lost=self._capture_tracking_lost(
                            tid,
                            track_positions,
                        ),
                    )

            # Absorb corpse heat into death sites (larger radius than track
            # dedup). Refresh site position + cooldown while heat remains.
            kept_candidates: list = []
            death_absorbed = 0
            for detection in result.new_candidates:
                if self._death_site_store.absorb_heat(
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

            # Factor 2: Tracks missed by discovery 3+ scans in a row
            # Only the remaining in-range tracks are evaluated — out-of-range
            # tracks were already handled by Factor 1.
            remaining_ids = unmatched_ids - out_of_range
            miss_remove, _first_miss = self._evaluate_discovery_miss_removal(
                remaining_ids,
                hunt_roi=hunt_roi,
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
                death_sites_active=self._death_site_store.active_count(tick),
            )
            return summary

    # ── Removal-factor evaluators ────────────────────────────────────────
    # Each method evaluates ONE removal factor and returns a set of track
    # IDs to remove (or (set, list) tuple). Add new factors as new methods.

    def _evaluate_out_of_range_removal(
        self,
        unmatched_ids: set[int],
        hunt_roi: HuntRoi | None,
        track_positions: list[tuple],
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
        *,
        hunt_roi: HuntRoi | None = None,
    ) -> tuple[set[int], list[int]]:
        """Factor 2: Remove tracks missed by 3+ consecutive discovery scans.

        Only receives in-range tracks (out-of-range handled by Factor 1).

        Side-effect: increments ``discovery_miss_count`` (except occlusion
        holds described below).

        Returns ``(remove_ids, first_miss_ids)`` where:
        - ``remove_ids``: tracks with miss_count >= 3.
        - ``first_miss_ids``: tracks on their first miss (still alive).

        Caller records a death site when the removed track was opacity-fading.

        Near the character (ROI center), discovery silhouette often fails
        because the player sprite merges into the extract. When local
        tracking still has the mob (``lost_count == 0``), that is occlusion
        — do not count the miss. Once tracking has also lost it, count
        normally so corpses under the character are still removed.

        Tracking confirmation additionally holds the counter everywhere: a
        fresh-frame hit resets ``discovery_miss_count`` (see
        ``apply_track_observation``), so discovery misses alone can never
        remove a mob local tracking still follows.
        """
        remove_ids: set[int] = set()
        first_miss_ids: list[int] = []
        char_x = hunt_roi.center_x if hunt_roi is not None else None
        char_y = hunt_roi.center_y if hunt_roi is not None else None
        for track_id in unmatched_ids:
            track = self._get_track_by_id_locked(track_id)
            if track is None:
                continue
            clear_discovery_blob_observation(track)
            if (
                char_x is not None
                and char_y is not None
                and track.lost_count == 0
                and _within_melee_guard(track.x, track.y, char_x, char_y)
            ):
                continue
            track.discovery_miss_count += 1
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

                # Tracking miss — keep last known position, advance lost count.
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

    def remove_track(self, track_id: int) -> bool:
        """Remove one just-created track when its observation cannot be committed."""
        with self._lock:
            if self._get_track_by_id_locked(track_id) is None:
                return False
            self._remove_tracks_locked({track_id})
            return True

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
                self._death_site_store.record(track.x, track.y, now_tick)
        self._remove_tracks_locked(remove_ids)

    def _dedup_positions_locked(
        self,
        now_tick: int,
        *,
        alive: list[MobTrack] | None = None,
    ) -> list[tuple[int, int]]:
        """Alive-track positions discovery must treat as already known.

        Death sites are absorbed separately via ``DeathSiteStore``
        (larger radius + cooldown refresh).
        """
        del now_tick
        if alive is None:
            alive = [t for t in self._tracks if is_alive(t)]
        return [(t.x, t.y) for t in alive]

    @staticmethod
    def _capture_position(
        track_id: int,
        track_positions: list[tuple],
    ) -> tuple[int | None, int | None]:
        """Get capture-time (x, y) for a track from the snapshot positions."""
        for entry in track_positions:
            if entry[0] == track_id:
                return int(entry[1]), int(entry[2])
        return None, None

    @staticmethod
    def _capture_tracking_lost(track_id: int, track_positions: list[tuple]) -> bool:
        """Return the capture-time local-tracking loss state for a track."""
        for entry in track_positions:
            if entry[0] == track_id:
                return len(entry) > 5 and int(entry[5]) > 0
        return False

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
            debuff_applied=track.debuff_applied,
            area_epoch=track.area_epoch,
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
