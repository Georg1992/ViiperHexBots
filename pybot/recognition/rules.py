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
  opacity baseline / decay.  Sustained opacity drop while stationary
  removes the track (in-place death fade).  Discovery matches also update
  ``discovery_stationary`` from heat-blob stability (center + bbox IoU).
  On miss it coasts along velocity and wakes discovery for confirmation.
- **Attack** records attack_count / last_attack_tick only; it reads position
  snapshots for clicks but must not mutate tracking fields or remove tracks.
  Idle-attack SP checks are a separate death / unreachable path
  (death uses discovery blob stationary, not tracking displacement).
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
    # Heat-CC bbox in the same coordinate space as x/y (screen for runtime).
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)


@dataclass
class ReconcileSummary:
    tracks_before: int = 0
    tracks_after: int = 0
    alive_before: int = 0
    alive_after: int = 0
    created_ids: list[int] | None = None
    removed_ids: list[int] | None = None
    removed_out_of_range_ids: list[int] | None = None
    removed_discovery_miss_ids: list[int] | None = None
    matched_count: int = 0
    added_count: int = 0
    removed_count: int = 0
    death_sites_active: int = 0


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
    # Last matched discovery heat blob (for stationary = blob unchanged).
    last_discovery_x: int = 0
    last_discovery_y: int = 0
    last_discovery_bbox: tuple[int, int, int, int] = (0, 0, 0, 0)
    discovery_blob_seen: bool = False
    discovery_stationary: bool = False  # True when consecutive discovery blobs match
    lost_count: int = 0
    area_epoch: int = 0
    opacity_baseline: float = 0.0
    opacity_baseline_samples: int = 0
    # Consecutive stationary frames below the opacity decay ratio; 0 = not fading.
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
                    bbox=detection.bbox,
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
    detection: DiscoveryDetection,
    config: dict,
) -> None:
    """Record that discovery saw this track in its latest scan.

    Resets the discovery-miss streak so the track is not removed by the
    2-miss absence rule.  Does NOT write track position — tracking owns that.

    Updates ``discovery_stationary`` from heat-blob stability (center shift
    + bbox IoU vs the previous matched discovery blob).
    """
    track.last_discovery_tick = now_tick
    track.discovery_miss_count = 0

    bx, by, bw, bh = detection.bbox
    if bw <= 0 or bh <= 0:
        track.discovery_stationary = False
        return

    if not track.discovery_blob_seen:
        track.last_discovery_x = detection.x
        track.last_discovery_y = detection.y
        track.last_discovery_bbox = detection.bbox
        track.discovery_blob_seen = True
        track.discovery_stationary = False
        return

    stop_px = int(config["movementStopThresholdPx"])
    min_iou = float(config["discoveryBlobMinIoU"])
    dx = detection.x - track.last_discovery_x
    dy = detection.y - track.last_discovery_y
    center_stable = (dx * dx + dy * dy) <= (stop_px * stop_px)
    iou = bbox_iou(track.last_discovery_bbox, detection.bbox)
    track.discovery_stationary = center_stable and iou >= min_iou
    track.last_discovery_x = detection.x
    track.last_discovery_y = detection.y
    track.last_discovery_bbox = detection.bbox


def bbox_iou(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> float:
    """IoU of two ``(x, y, w, h)`` boxes; 0 when either has non-positive area."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return 0.0
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix0, iy0 = max(ax, bx), max(ay, by)
    ix1, iy1 = min(ax2, bx2), min(ay2, by2)
    iw, ih = ix1 - ix0, iy1 - iy0
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = float(iw * ih)
    union = float(aw * ah + bw * bh) - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def clear_discovery_blob_observation(track: MobTrack) -> None:
    """Clear blob-stability state when discovery misses this track."""
    track.discovery_stationary = False
    track.discovery_blob_seen = False
    track.last_discovery_bbox = (0, 0, 0, 0)


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


def apply_opacity_observation(
    track: MobTrack,
    *,
    opacity_score: float,
    config: dict,
) -> bool:
    """Update opacity baseline / decay streak; return True when death is confirmed.

    Call after ``apply_movement_observation`` so ``track.moving`` gates the
    fade clock for this frame. Measurement comes from the local tracker
    (``opacity_score``); this mutates only MobTrack opacity fields.
    """
    baseline, samples, streak, dead = evaluate_opacity_death(
        opacity_score=opacity_score,
        baseline=track.opacity_baseline,
        baseline_samples=track.opacity_baseline_samples,
        decay_streak=track.opacity_decay_streak,
        config=config,
        moving=track.moving,
    )
    track.opacity_baseline = baseline
    track.opacity_baseline_samples = samples
    track.opacity_decay_streak = streak
    return dead


def evaluate_opacity_death(
    *,
    opacity_score: float,
    baseline: float,
    baseline_samples: int,
    decay_streak: int,
    config: dict,
    moving: bool,
) -> tuple[float, int, int, bool]:
    """Update opacity baseline state and return whether death is confirmed.

    The first ``deathOpacityBaselineSamples`` found frames establish the
    living baseline (running max). Once calibrated above
    ``deathOpacityMinBaseline``, a score below ``baseline * deathOpacityDecayRatio``
    for ``deathOpacityConfirmTicks`` consecutive stationary frames confirms
    death. Motion holds the streak (blur is not fade); recovery clears it.
    """
    min_samples = int(config["deathOpacityBaselineSamples"])
    min_baseline = float(config["deathOpacityMinBaseline"])
    decay_ratio = float(config["deathOpacityDecayRatio"])
    confirm_ticks = int(config["deathOpacityConfirmTicks"])

    if baseline_samples < min_samples:
        baseline = max(baseline, opacity_score)
        baseline_samples += 1
        return baseline, baseline_samples, 0, False

    if baseline < min_baseline:
        baseline = max(baseline, opacity_score)
        return baseline, baseline_samples, decay_streak, False

    dropped = opacity_score < (baseline * decay_ratio)
    if dropped and moving:
        # Walk / attack blur can look like a drop — hold the fade streak.
        return baseline, baseline_samples, decay_streak, False
    if dropped:
        decay_streak += 1
        if decay_streak >= confirm_ticks:
            return baseline, baseline_samples, 0, True
        return baseline, baseline_samples, decay_streak, False

    return baseline, baseline_samples, 0, False

