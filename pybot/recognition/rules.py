"""Reference model of the hunt track pipeline.

Used by tests to lock the pipeline contract.

Ownership:
- **Discovery** scans the hunt ROI for living mobs each cycle, publishes
  new-mob candidates, matches detections to existing tracks (resetting
  discovery_miss_count), and removes tracks that are out-of-range or missed
  for 2 consecutive scans.  Discovery never creates tracks directly.
- **Tracking** owns track creation and position.  On each tick it ingests
  discovery candidates, runs a local-follow search on the *current fresh
  frame* to get exact coordinates, creates tracks at those coordinates,
  then follows every alive track via pure heatmap local follow (no
  silhouette gate).  On found=True it updates position, velocity, and
  clears discovery priors.  On stationary timeout it flags the track as a
  corpse.  On miss it coasts along velocity and wakes discovery for
  confirmation.
- **Attack** records attack_count / last_attack_tick only; it reads position
  snapshots for clicks but must not mutate tracking fields or remove tracks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# Same-object dedup radius for discovery vs existing tracks. Clustering of raw
# detections before track creation uses discoveryClusterRadiusPx from config
# (typically smaller) so nearby distinct mobs are not merged.
HUNT_OBJECT_RADIUS = 90
HUNT_DISCOVERY_CLUSTER_RADIUS = 48

TrackState = Literal["alive", "unreachable"]


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
    idle_attack_count: int = 0
    was_accessible: bool = False  # True once SP consumption proves the mob is hittable
    state: TrackState = "alive"
    mob_name: str = ""
    created_tick: int = 0
    updated_tick: int = 0
    last_found_tick: int = 0
    last_attack_tick: int = 0
    last_discovery_tick: int = 0
    discovery_scale: float = 0.0
    candidate_scale: float = 0.0
    lost_count: int = 0
    area_epoch: int = 0
    opacity_baseline: float = 0.0
    opacity_baseline_samples: int = 0
    # Monotonic tick when a meaningful opacity fade began; 0 = not fading.
    opacity_decay_streak: int = 0
    moving: bool = False
    vel_x: float = 0.0
    vel_y: float = 0.0
    # Consecutive discovery scans that failed to see this track (unmatched).
    # At >= 2 the track is removed immediately — it failed discovery gates
    # twice in a row, meaning it's dead or gone.
    discovery_miss_count: int = 0

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
            last_found_tick=now_tick,
            last_discovery_tick=now_tick,
            discovery_scale=discovery_scale,
            candidate_scale=discovery_scale,
            area_epoch=area_epoch,

        )


def is_alive(track: MobTrack) -> bool:
    return track.state == "alive"


def apply_attack_event(track: MobTrack, now_tick: int) -> None:
    """Record one attack directed at this mob track (attack-owned fields only)."""
    track.attack_count += 1
    track.last_attack_tick = now_tick
    track.lost_count = 0


def select_target_id(
    tracks: list[MobTrack],
    now_tick: int,
    last_attack_target_id: int = 0,
    *,
    max_attacks: int | None = None,
) -> int:
    """Round-robin through alive tracks."""
    alive_ids = sorted(track.id for track in tracks if is_alive(track))
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


def apply_discovery_match(
    track: MobTrack,
    *,
    now_tick: int,
) -> None:
    """Record that discovery saw this track in its latest scan.

    Resets the discovery-miss streak so the track is not removed by the
    2-miss absence rule.  Does NOT write position — tracking owns that.
    """
    track.last_discovery_tick = now_tick
    track.discovery_miss_count = 0


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
        # Responsive velocity EMA: 50/50 weight so direction changes correct
        # within 1 frame. 0.65/0.35 was too slow for fast mobs and sudden
        # turns — the tracked position lagged behind actual movement.
        track.vel_x = (0.5 * track.vel_x) + (0.5 * dx)
        track.vel_y = (0.5 * track.vel_y) + (0.5 * dy)
        track.x = x
        track.y = y
        track.updated_tick = now_tick
        track.last_found_tick = now_tick
        track.lost_count = 0
        # NOTE: discovery_miss_count is NOT reset here — only discovery
        # (apply_discovery_match) determines liveness. The tracker is a pure
        # follower; if it reports found=True on background noise it should
        # NOT block discovery's 2-miss removal.
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
    track.updated_tick = now_tick


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


def movement_thresholds(config: dict) -> tuple[int, int]:
    """Pixel thresholds for entering and leaving the track ``moving`` state."""
    return (
        int(config["movementMoveThresholdPx"]),
        int(config["movementStopThresholdPx"]),
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

