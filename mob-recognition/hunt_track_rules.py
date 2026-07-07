"""Reference model of HuntTracks attack/state rules.

Used by tests to lock the hunt pipeline contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

HUNT_ATTACK_RESULT_WINDOW_MS = 1800
HUNT_STATE_INTERVAL_MS = 100
HUNT_UNREACHABLE_CONFIRM_STREAK = 3
HUNT_LOCAL_TRACK_MISS_LIMIT = 3
HUNT_TRACK_MATCH_RADIUS = 90
HUNT_TRACK_MISS_LIMIT = 3
HUNT_DETECTION_CLUSTER_RADIUS = 55
HUNT_MOVEMENT_SLACK_PX_PER_STATE_TICK = 30
HUNT_DISCOVERY_MATCH_SLACK_CAP_PX = 150
HUNT_MAX_CONFIRM_AGE_MS = 5000
HUNT_MAX_CONSECUTIVE_FOUND_WITHOUT_CONFIRM = 10

TrackState = Literal["alive", "pending", "dead", "unreachable"]


@dataclass
class DiscoveryDetection:
    x: int
    y: int
    confidence: float
    candidate_scale: float = 0.0
    living: bool = True
    dead: bool = False


@dataclass
class ReconcileSummary:
    tracks_before: int = 0
    tracks_after: int = 0
    alive_before: int = 0
    alive_or_pending_after: int = 0
    matched_ids: list[int] | None = None
    created_ids: list[int] | None = None
    removed_ids: list[int] | None = None
    matched_count: int = 0
    removed_count: int = 0
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
    last_state_tick: int = 0
    last_attack_tick: int = 0
    last_discovery_tick: int = 0
    pending_result_until_tick: int = 0
    pending_result_resolved: bool = False
    pending_timeout_logged: bool = False
    discovery_scale: float = 0.0
    candidate_scale: float = 0.0
    discovery_miss_count: int = 0
    local_track_miss_count: int = 0
    state_unreachable_count: int = 0
    suspicious_dead_count: int = 0
    area_epoch: int = 0
    last_confirm_tick: int = 0
    consecutive_found_count: int = 0

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
            last_confirm_tick=now_tick,
        )


@dataclass
class LocalTrackObservation:
    id: int
    found: bool
    x: int = 0
    y: int = 0
    confidence: float = 0.0
    miss_reason: str = ""


@dataclass
class StateObservation:
    id: int
    state: TrackState | Literal["unknown"]
    x: int = 0
    y: int = 0
    confidence: float = 0.0


def coord_age_ms(track: MobTrack, now_tick: int) -> int:
    return now_tick - track.updated_tick


def was_attacked(track: MobTrack) -> bool:
    return track.attack_count > 0 or track.state == "pending"


def is_alive(track: MobTrack) -> bool:
    return track.state in ("alive", "pending")


def is_pending(track: MobTrack, now_tick: int) -> bool:
    if track.state != "pending" or track.pending_result_resolved:
        return False
    if track.pending_result_until_tick > now_tick:
        return True
    track.state = "alive"
    track.pending_result_resolved = True
    track.state_unreachable_count = 0
    return False


def is_attackable(
    track: MobTrack,
    now_tick: int,
) -> bool:
    if not is_alive(track):
        return False
    if is_pending(track, now_tick):
        return False
    return True


def attack_block_reason(
    track: MobTrack,
    now_tick: int,
) -> str:
    if not is_alive(track):
        return "not_alive"
    if is_pending(track, now_tick):
        return "pending"
    return ""


def apply_local_track_observation(
    track: MobTrack,
    observation: LocalTrackObservation,
    now_tick: int,
) -> bool:
    """Update coordinates from local follower. Returns True when canonical confirm is needed."""
    if observation.found:
        track.local_track_miss_count = 0
        track.consecutive_found_count += 1
        if observation.x > 0 and observation.y > 0:
            track.x = observation.x
            track.y = observation.y
        if observation.confidence > 0:
            track.confidence = observation.confidence
        track.updated_tick = now_tick
        if track.consecutive_found_count >= HUNT_MAX_CONSECUTIVE_FOUND_WITHOUT_CONFIRM:
            # Don't reset here — keep eligible until confirm worker processes it.
            # apply_state_observation("alive") resets it, or track gets removed if dead.
            return True
        return False

    track.local_track_miss_count += 1
    track.consecutive_found_count = 0
    return track.local_track_miss_count >= HUNT_LOCAL_TRACK_MISS_LIMIT


def apply_state_observation(track: MobTrack, observation: StateObservation, now_tick: int) -> bool:
    """Apply state observation. Returns False if track was removed."""
    if observation.state == "dead":
        if not was_attacked(track):
            track.suspicious_dead_count += 1
            return True
        track.last_state_tick = now_tick
        return False

    if observation.state == "unreachable":
        if was_attacked(track):
            # Mob was attacked and is now unreachable — can't kill it or left range.
            # Mark as unreachable so the attack loop skips it.
            track.state = "unreachable"
            track.last_state_tick = now_tick
            return True
        # Not attacked — could be a temporary tracking miss;
        # wait for streak before marking.
        track.state_unreachable_count += 1
        if track.state_unreachable_count >= HUNT_UNREACHABLE_CONFIRM_STREAK:
            track.state = "unreachable"
            track.last_state_tick = now_tick
        return True

    if observation.state == "unknown":
        return True

    if observation.state == "alive":
        track.last_state_tick = now_tick
        track.last_confirm_tick = now_tick
        track.consecutive_found_count = 0
        track.suspicious_dead_count = 0
        track.state_unreachable_count = 0
        if observation.x > 0 and observation.y > 0:
            track.x = observation.x
            track.y = observation.y
        if observation.confidence > 0:
            track.confidence = observation.confidence
        track.state = "alive"
        track.pending_result_until_tick = 0
        track.pending_result_resolved = True
        track.pending_timeout_logged = False
        track.updated_tick = now_tick
        return True

    return True


def apply_attack_event(track: MobTrack, now_tick: int) -> None:
    track.attack_count += 1
    track.last_attack_tick = now_tick
    track.pending_result_until_tick = now_tick + HUNT_ATTACK_RESULT_WINDOW_MS
    track.pending_result_resolved = False
    track.pending_timeout_logged = False
    track.state_unreachable_count = 0
    track.local_track_miss_count = 0
    track.suspicious_dead_count = 0
    track.state = "pending"
    track.updated_tick = now_tick


def state_request_scale(track: MobTrack, session_scale_hint: float = 0.0) -> float:
    if track.discovery_scale > 0:
        return track.discovery_scale
    if track.candidate_scale > 0:
        return track.candidate_scale
    return session_scale_hint


def collect_local_track_requests(
    tracks: list[MobTrack],
    *,
    session_scale_hint: float = 0.0,
) -> list[dict]:
    """Alive tracks for local coordinate follow (not canonical state)."""
    requests: list[dict] = []
    for track in tracks:
        if not is_alive(track):
            continue
        scale = state_request_scale(track, session_scale_hint)
        req = {"id": track.id, "x": track.x, "y": track.y}
        if scale > 0:
            req["scale"] = scale
        requests.append(req)
    return requests


def select_state_confirm_track_id(
    tracks: list[MobTrack],
    now_tick: int,
) -> int:
    """Pick one track needing canonical state confirm.

    Priority:
    1. Tracks in "pending" state (post-attack result window).
    2. Tracks with too many local-track misses (suspected unreachable).
    3. Tracks that haven't had a state confirm recently (death detection gap).
    """
    pending = [track.id for track in tracks if is_pending(track, now_tick)]
    if pending:
        return pending[0]
    for track in tracks:
        if is_alive(track) and track.local_track_miss_count >= HUNT_LOCAL_TRACK_MISS_LIMIT:
            return track.id
    for track in tracks:
        if (
            is_alive(track)
            and track.consecutive_found_count >= HUNT_MAX_CONSECUTIVE_FOUND_WITHOUT_CONFIRM
        ):
            return track.id
    for track in tracks:
        if (
            is_alive(track)
            and track.last_confirm_tick > 0
            and (now_tick - track.last_confirm_tick) >= HUNT_MAX_CONFIRM_AGE_MS
        ):
            return track.id
    return 0


def collect_state_requests(
    tracks: list[MobTrack],
    *,
    session_scale_hint: float = 0.0,
) -> list[dict]:
    """Mirror HuntTracks_CollectStateRequests — all alive tracks request state."""
    pending: list[dict] = []
    rest: list[dict] = []
    for track in tracks:
        if not is_alive(track):
            continue
        scale = state_request_scale(track, session_scale_hint)
        req = {"id": track.id, "x": track.x, "y": track.y}
        if scale > 0:
            req["scale"] = scale
        if track.state == "pending":
            pending.append(req)
        else:
            rest.append(req)
    return pending + rest


def select_target_id(
    tracks: list[MobTrack],
    now_tick: int,
    roi_center: tuple[int, int] = (0, 0),
    last_attack_target_id: int = 0,
) -> int:
    attackable_ids = sorted(
        track.id for track in tracks if is_attackable(track, now_tick)
    )
    if not attackable_ids:
        return 0
    if last_attack_target_id not in attackable_ids:
        return attackable_ids[0]
    last_index = attackable_ids.index(last_attack_target_id)
    next_index = (last_index + 1) % len(attackable_ids)
    return attackable_ids[next_index]


def discovery_match_radius_px(track: MobTrack, now_tick: int) -> float:
    age_ms = coord_age_ms(track, now_tick)
    movement_slack = (age_ms / HUNT_STATE_INTERVAL_MS) * HUNT_MOVEMENT_SLACK_PX_PER_STATE_TICK
    if movement_slack > HUNT_DISCOVERY_MATCH_SLACK_CAP_PX:
        movement_slack = HUNT_DISCOVERY_MATCH_SLACK_CAP_PX
    radius = HUNT_TRACK_MATCH_RADIUS + movement_slack
    if track.state == "pending":
        radius *= 1.5
    return radius


def discovery_match_radius_sq(track: MobTrack, now_tick: int) -> float:
    radius = discovery_match_radius_px(track, now_tick)
    return radius * radius


def cluster_living_detections(
    detections: list[DiscoveryDetection],
    cluster_radius: int = HUNT_DETECTION_CLUSTER_RADIUS,
) -> list[DiscoveryDetection]:
    living = [d for d in detections if d.living and not d.dead]
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
                    dead=False,
                )
            )
    return clusters


def find_nearest_track_for_detection(
    tracks: list[MobTrack],
    x: int,
    y: int,
    now_tick: int,
    matched_track_ids: set[int],
) -> MobTrack | None:
    best_track: MobTrack | None = None
    best_dist_sq = 999_999_999.0
    for track in tracks:
        if not is_alive(track):
            continue
        if track.id in matched_track_ids:
            continue
        dx = x - track.x
        dy = y - track.y
        dist_sq = (dx * dx) + (dy * dy)
        track_radius_sq = discovery_match_radius_sq(track, now_tick)
        if dist_sq <= track_radius_sq and dist_sq < best_dist_sq:
            best_dist_sq = dist_sq
            best_track = track
    return best_track


def reconcile_detection(
    track: MobTrack,
    detection: DiscoveryDetection,
    *,
    now_tick: int,
) -> None:
    track.last_discovery_tick = now_tick
    track.discovery_miss_count = 0
    if track.attack_count == 0:
        track.x = detection.x
        track.y = detection.y
        track.updated_tick = now_tick
        if detection.confidence > 0:
            track.confidence = detection.confidence
    if track.candidate_scale > 0:
        track.candidate_scale = detection.candidate_scale
    if detection.candidate_scale > 0:
        track.discovery_scale = detection.candidate_scale


def reconcile_unmatched_tracks(
    tracks: list[MobTrack],
    matched_track_ids: set[int],
) -> list[int]:
    removed_ids: list[int] = []
    for track in tracks:
        if track.id in matched_track_ids:
            continue
        if track.state == "dead":
            continue
        # Unreachable tracks: still count discovery misses so they eventually get cleaned up.
        # Dead tracks (confirmed kills) are removed by _apply_state_observation_locked directly.
        if not is_alive(track) and track.state != "unreachable":
            continue
        track.discovery_miss_count += 1
        if track.discovery_miss_count >= HUNT_TRACK_MISS_LIMIT:
            removed_ids.append(track.id)
    return removed_ids
