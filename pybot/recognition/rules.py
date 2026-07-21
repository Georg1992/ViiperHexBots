"""Reference model of the hunt track pipeline.

Used by tests to lock the pipeline contract.

Ownership:
- **Discovery** creates tracks for new mobs, marks in-ROI unmatched tracks as
  ``discovery_absent``, may drop unmatched tracks that are already outside the
  hunt ROI (left the search area), on match publishes a soft position prior
  (``discovery_obs_*``), and may set ``discovery_death`` (+ death-site coords)
  when a death silhouette wins over living at a known track. It never
  overwrites authoritative ``x``/``y`` and never removes for death — tracking
  owns in-ROI death/lost/unreachable removal.
- **Tracking** is the sole writer of authoritative position, movement, opacity
  death, lost_count, unreachable removal, and in-ROI death/lost removal. It
  consumes discovery priors on miss, drops on joint absence, and removes
  tracks when ``discovery_death`` is set (ghost at the recorded death site)
  or opacity confirms (ghost at the opacity hit).
- **Attack** records attack_count / last_attack_tick only; it reads position
  snapshots for clicks but must not mutate tracking fields or remove tracks.
- Death-flagged tracks are excluded from living discovery match/create so a
  sticky ``discovery_death`` cannot consume a nearby living detection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# Same-object dedup radius for discovery vs existing tracks. Clustering of raw
# detections before track creation uses discoveryClusterRadiusPx from config
# (typically smaller) so nearby distinct mobs are not merged.
HUNT_OBJECT_RADIUS = 90
HUNT_DISCOVERY_CLUSTER_RADIUS = 48

# Consecutive local-follow misses before a track is dropped as lost. Tracking
# ticks every WORKER_POLL_INTERVAL_S; default 40 misses ≈ 2s at 50ms.
HUNT_TRACK_LOST_LIMIT = 40

TrackState = Literal["alive"]


@dataclass
class DiscoveryDetection:
    x: int
    y: int
    confidence: float
    candidate_scale: float = 0.0
    living: bool = True


@dataclass
class ReconcileSummary:
    tracks_before: int = 0
    tracks_after: int = 0
    alive_before: int = 0
    alive_after: int = 0
    created_ids: list[int] | None = None
    removed_ids: list[int] | None = None
    matched_count: int = 0
    added_count: int = 0
    removed_count: int = 0


@dataclass
class MobTrack:
    id: int
    x: int
    y: int
    confidence: float = 0.0
    attack_count: int = 0
    attack_count_baseline: int = 0
    state: TrackState = "alive"
    mob_name: str = ""
    created_tick: int = 0
    updated_tick: int = 0
    last_attack_tick: int = 0
    last_discovery_tick: int = 0
    discovery_scale: float = 0.0
    candidate_scale: float = 0.0
    lost_count: int = 0
    area_epoch: int = 0
    opacity_baseline: float = 0.0
    opacity_baseline_samples: int = 0
    opacity_decay_streak: int = 0
    moving: bool = False
    vel_x: float = 0.0
    vel_y: float = 0.0
    attack_anchor_x: int = 0
    attack_anchor_y: int = 0
    # Set by discovery when this track's coords were unmatched on a scan.
    # Tracking removes the track when it also misses (joint absence).
    discovery_absent: bool = False
    # Soft position prior from the latest discovery match (0 tick = none).
    # Tracking searches/snaps from these coords; authoritative x/y stay tracking-owned.
    discovery_obs_x: int = 0
    discovery_obs_y: int = 0
    discovery_obs_tick: int = 0
    # Discovery helper: death silhouette won over living. Tracking removes.
    discovery_death: bool = False
    # Screen coords of the death site at confirmation (ghost / anti-rediscovery).
    discovery_death_x: int = 0
    discovery_death_y: int = 0

    @classmethod
    def from_discovery(
        cls,
        track_id: int,
        x: int,
        y: int,
        confidence: float,
        *,
        now_tick: int,
        discovery_scale: float = 0.0,
        mob_name: str = "",
        area_epoch: int = 0,
    ) -> MobTrack:
        return cls(
            id=track_id,
            x=x,
            y=y,
            confidence=confidence,
            mob_name=mob_name,
            created_tick=now_tick,
            updated_tick=now_tick,
            last_discovery_tick=now_tick,
            discovery_scale=discovery_scale,
            candidate_scale=discovery_scale,
            area_epoch=area_epoch,
            attack_anchor_x=x,
            attack_anchor_y=y,
        )


def is_alive(track: MobTrack) -> bool:
    return track.state == "alive"


def mob_attack_anchor_key(x: int, y: int, *, cell_px: int) -> tuple[int, int]:
    """Snap a mob position to a stable cell for per-mob attack accounting."""
    return (x // cell_px * cell_px, y // cell_px * cell_px)


def apply_attack_event(track: MobTrack, now_tick: int) -> None:
    """Record one attack directed at this mob track (attack-owned fields only)."""
    track.attack_count += 1
    track.last_attack_tick = now_tick
    track.lost_count = 0


def is_track_unreachable_by_attacks(track: MobTrack, max_attacks: int) -> bool:
    """True when this track exceeded the attack budget without being killed."""
    return track.attack_count >= max_attacks


def max_attacks_per_mob_before_unreachable(
    *,
    average_attacks_till_death: float,
    skill_delay_ms: int,
    attack_window_ms: int = 3000,
) -> int:
    """Unreachable budget: attacks that fit in 3s at current delay, plus session average."""
    delay_ms = max(skill_delay_ms, 1)
    attacks_in_window = attack_window_ms / delay_ms
    return max(1, round(attacks_in_window + average_attacks_till_death))


def select_target_id(
    tracks: list[MobTrack],
    now_tick: int,
    last_attack_target_id: int = 0,
    *,
    max_attacks: int | None = None,
) -> int:
    """Round-robin through alive tracks that are still within the attack budget."""
    alive_ids = sorted(
        track.id
        for track in tracks
        if is_alive(track)
        and (
            max_attacks is None
            or not is_track_unreachable_by_attacks(track, max_attacks)
        )
    )
    if not alive_ids:
        return 0
    if last_attack_target_id not in alive_ids:
        return alive_ids[0]
    last_index = alive_ids.index(last_attack_target_id)
    next_index = (last_index + 1) % len(alive_ids)
    return alive_ids[next_index]


def cluster_living_detections(
    detections: list[DiscoveryDetection],
    cluster_radius: int = HUNT_DISCOVERY_CLUSTER_RADIUS,
) -> list[DiscoveryDetection]:
    living = [d for d in detections if d.living]
    if not living:
        return []
    living.sort(key=lambda d: d.confidence, reverse=True)
    cluster_radius_sq = cluster_radius * cluster_radius
    clusters: list[DiscoveryDetection] = []
    for detection in living:
        merged = False
        for cluster in clusters:
            dx = detection.x - cluster.x
            dy = detection.y - cluster.y
            if (dx * dx + dy * dy) <= cluster_radius_sq:
                merged = True
                break
        if not merged:
            clusters.append(
                DiscoveryDetection(
                    x=detection.x,
                    y=detection.y,
                    confidence=detection.confidence,
                    candidate_scale=detection.candidate_scale,
                    living=True,
                )
            )
    return clusters


def detection_matches_existing(
    x: int,
    y: int,
    positions: list[tuple[int, int]],
    *,
    dedup_radius: int = HUNT_OBJECT_RADIUS,
) -> bool:
    """True if a detection belongs to an object we already know about.

    ``positions`` are the (x, y) of known objects sampled at the same instant
    the discovery frame was captured, so the detection and the positions it is
    compared against share one time reference. Within *dedup_radius* means
    same object.
    """
    radius_sq = dedup_radius * dedup_radius
    for px, py in positions:
        dx = x - px
        dy = y - py
        if (dx * dx) + (dy * dy) <= radius_sq:
            return True
    return False


def has_discovery_observation(track: MobTrack) -> bool:
    """True when discovery published a soft position prior not yet cleared."""
    return track.discovery_obs_tick > 0


def apply_discovery_observation(
    track: MobTrack,
    *,
    x: int,
    y: int,
    now_tick: int,
) -> None:
    """Discovery match prior — does not overwrite authoritative track x/y."""
    track.discovery_obs_x = x
    track.discovery_obs_y = y
    track.discovery_obs_tick = now_tick
    track.last_discovery_tick = now_tick
    track.discovery_absent = False


def note_discovery_death(track: MobTrack, *, x: int, y: int) -> None:
    """Discovery helper: death won at ``(x, y)``; tracking will remove + ghost."""
    track.discovery_death = True
    track.discovery_death_x = int(x)
    track.discovery_death_y = int(y)
    track.discovery_absent = False
    clear_discovery_observation(track)


def clear_discovery_observation(track: MobTrack) -> None:
    track.discovery_obs_x = 0
    track.discovery_obs_y = 0
    track.discovery_obs_tick = 0


def apply_discovery_reanchor(
    track: MobTrack,
    *,
    now_tick: int,
) -> bool:
    """Tracking writer path: snap x/y to discovery prior when drifted from it.

    Returns True when a snap was applied. When already at the prior, returns
    False so the caller can advance normal miss/lost accounting. Keeps
    ``discovery_obs_*`` until a real local hit confirms the mob.
    """
    if track.x == track.discovery_obs_x and track.y == track.discovery_obs_y:
        return False
    track.x = track.discovery_obs_x
    track.y = track.discovery_obs_y
    track.updated_tick = now_tick
    track.lost_count = 0
    track.moving = False
    track.vel_x = 0.0
    track.vel_y = 0.0
    track.discovery_absent = False
    return True


def apply_track_observation(
    track: MobTrack,
    *,
    found: bool,
    x: int,
    y: int,
    confidence: float,
    now_tick: int,
) -> None:
    """Tracking owns position + liveness. Fresh coords on hit; coast only while moving."""
    if found:
        dx = float(x - track.x)
        dy = float(y - track.y)
        track.vel_x = (0.65 * track.vel_x) + (0.35 * dx)
        track.vel_y = (0.65 * track.vel_y) + (0.35 * dy)
        track.x = x
        track.y = y
        track.updated_tick = now_tick
        track.lost_count = 0
        clear_discovery_observation(track)
        track.discovery_absent = False
        if confidence > 0:
            track.confidence = confidence
        return

    # Coast only when movement is established — residual EMA on a stationary
    # miss otherwise jumps the search window and looks like lag/racing.
    if track.moving:
        track.x += int(round(track.vel_x))
        track.y += int(round(track.vel_y))
        track.vel_x *= 0.9
        track.vel_y *= 0.9
    else:
        track.vel_x *= 0.5
        track.vel_y *= 0.5
    track.lost_count += 1


def apply_opacity_observation(
    track: MobTrack,
    *,
    opacity_baseline: float,
    opacity_baseline_samples: int,
    opacity_decay_streak: int,
) -> None:
    track.opacity_baseline = opacity_baseline
    track.opacity_baseline_samples = opacity_baseline_samples
    track.opacity_decay_streak = opacity_decay_streak


def evaluate_track_moving(
    *,
    was_moving: bool,
    displacement_sq: int,
    move_threshold_px: int,
    stop_threshold_px: int,
) -> bool:
    """Hysteresis movement state from frame-to-frame displacement."""
    enter_sq = move_threshold_px * move_threshold_px
    stop_sq = stop_threshold_px * stop_threshold_px
    if was_moving:
        return displacement_sq > stop_sq
    return displacement_sq > enter_sq


def death_movement_thresholds(config: dict) -> tuple[int, int]:
    """Pixel thresholds for entering and leaving the track ``moving`` state."""
    return (
        int(config["deathOpacityMoveThresholdPx"]),
        int(config["deathOpacityStopThresholdPx"]),
    )


def apply_movement_observation(
    track: MobTrack,
    *,
    x: int,
    y: int,
    move_threshold_px: int,
    stop_threshold_px: int,
) -> None:
    dx = x - track.x
    dy = y - track.y
    track.moving = evaluate_track_moving(
        was_moving=track.moving,
        displacement_sq=(dx * dx) + (dy * dy),
        move_threshold_px=move_threshold_px,
        stop_threshold_px=stop_threshold_px,
    )


def is_track_lost(track: MobTrack, *, miss_limit: int = HUNT_TRACK_LOST_LIMIT) -> bool:
    return track.lost_count >= miss_limit


def track_lost_miss_limit(config: dict) -> int:
    return int(config.get("trackLostMissLimit", HUNT_TRACK_LOST_LIMIT))
