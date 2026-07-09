"""Reference model of the hunt track pipeline.

Used by tests to lock the pipeline contract. Tracks stay ``alive`` until
tracking removes them (lost after consecutive misses, or dead via opacity
decay when death detection is enabled). Discovery creates tracks; tracking
refreshes coordinates, movement/death state, and expires tracks; attack
rotates through alive targets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# Same-object dedup radius for discovery vs existing tracks. Clustering of raw
# detections before track creation uses discoveryClusterRadiusPx from config
# (typically smaller) so nearby distinct mobs are not merged.
HUNT_OBJECT_RADIUS = 70
HUNT_DISCOVERY_CLUSTER_RADIUS = 48

# Consecutive tracking misses before a track is considered gone. Tracking runs
# every vision tick, so this is a count of local-follow failures, not wall time.
HUNT_TRACK_LOST_LIMIT = 8

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
    matched_count: int = 0
    added_count: int = 0


@dataclass
class MobTrack:
    id: int
    x: int
    y: int
    confidence: float = 0.0
    attack_count: int = 0
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
        )


def is_alive(track: MobTrack) -> bool:
    return track.state == "alive"


def apply_attack_event(track: MobTrack, now_tick: int) -> None:
    """Record an attack on this track."""
    track.attack_count += 1
    track.last_attack_tick = now_tick
    track.updated_tick = now_tick


def select_target_id(
    tracks: list[MobTrack],
    now_tick: int,
    last_attack_target_id: int = 0,
) -> int:
    """Round-robin through all alive tracks."""
    alive_ids = sorted(track.id for track in tracks if track.state == "alive")
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


def apply_track_observation(
    track: MobTrack,
    *,
    found: bool,
    x: int,
    y: int,
    confidence: float,
    now_tick: int,
) -> None:
    """Tracking owns position + liveness. Fresh coords on hit, miss count on loss."""
    if found:
        track.x = x
        track.y = y
        track.updated_tick = now_tick
        track.lost_count = 0
        if confidence > 0:
            track.confidence = confidence
    else:
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


def is_track_lost(track: MobTrack) -> bool:
    return track.lost_count >= HUNT_TRACK_LOST_LIMIT
