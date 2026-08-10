"""Sticky local tracking for already-discovered mobs.

The healthy path follows how the known sprite moved since the previous frame:
multiple LK feature points are tracked inside a small ROI and their movement is
accepted only when a robust median consensus agrees.  The original acquisition
patch is retained as a stable identity anchor and is never continuously
rewritten.

When the drift signal is weak, tracking performs a bounded local recovery
(first small, then expanded).  Recovery reuses the descriptor heatmap/silhouette
matcher that Discovery uses, but it never performs a screen-wide scan and it
does not wait for Discovery.
"""

from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import TYPE_CHECKING

import cv2
import numpy as np

from pybot.recognition.detector.descriptors.descriptor import MobDescriptor
from pybot.recognition.detector.scoring.heatmap_detector import (
    _COVERAGE_SIZE_FRAC,
    palette_heatmap,
    sprite_palette_heatmap,
)
from pybot.recognition.detector.tracking.opacity_probe import measure_opacity_score

if TYPE_CHECKING:
    from pybot.recognition.detector.detector import MobDetector


_LOCAL_SUPPRESS_RADIUS_FLOOR_PX = 8
_FAST_BBOX_MIN_PX = 8
_REFINE_WINDOW_SCALE = 1.5
_LOCAL_FOLLOW_MIN_HEAT_FRAC = 0.5
_LOCAL_FOLLOW_BODY_W = 0.55
_LOCAL_FOLLOW_ACCENT_W = 0.45
_LOCAL_FOLLOW_SPRITE_W = 0.75
_LOCAL_FOLLOW_COLOR_W = 0.55
_LOCAL_HEATMAP_DOWNSCALE = 2
_FLOW_MIN_POINTS = 3
_FLOW_MAX_ERROR = 45.0
_FLOW_INLIER_FLOOR_PX = 3.0
_FLOW_MAX_DISPLACEMENT_MULTIPLIER = 1.35
_ANCHOR_MIN_SCORE = 0.18
_IDENTITY_MIN_SPRITE_FRACTION = 0.08
_IDENTITY_MIN_BODY_FRACTION = 0.02
_MAX_RECOVERY_FAILURES = 3
_PREDICTION_MIN_SPEED_PX = 2.0
_FEATURE_SPRITE_MIN_HEAT = 0.18


@dataclass(frozen=True)
class LocalTrackResult:
    track_id: int
    found: bool
    x: int
    y: int
    confidence: float
    miss_reason: str
    opacity_score: float = 0.0
    tracking_lost: bool = False


@dataclass
class _TrackVisualState:
    """Small mutable state owned by one DetectorSession and one Track."""

    center_x: int
    center_y: int
    scale: float
    anchor_gray: np.ndarray
    anchor_width: int
    anchor_height: int
    previous_gray: np.ndarray
    points: np.ndarray
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    recovery_failures: int = 0



def _state_store(detector: MobDetector) -> dict[int, _TrackVisualState]:
    store = getattr(detector, "_local_track_states", None)
    if store is None:
        store = {}
        setattr(detector, "_local_track_states", store)
    return store


def _state_lock(detector: MobDetector) -> threading.RLock:
    lock = getattr(detector, "_local_track_states_lock", None)
    if lock is None:
        lock = threading.RLock()
        setattr(detector, "_local_track_states_lock", lock)
    return lock


def transfer_track_state(detector: MobDetector, source_track_id: int, target_track_id: int) -> bool:
    """Move an acquisition state to the committed Track ID exactly once."""
    with _state_lock(detector):
        store = _state_store(detector)
        state = store.pop(source_track_id, None)
        if state is None:
            return False
        store[target_track_id] = state
        return True


def discard_track_state(detector: MobDetector, track_id: int) -> None:
    with _state_lock(detector):
        _state_store(detector).pop(track_id, None)


def prune_track_states(detector: MobDetector, active_track_ids: set[int]) -> None:
    """Drop visual state for Tracks removed by policy or Discovery."""
    with _state_lock(detector):
        store = _state_store(detector)
        for track_id in tuple(store):
            if track_id not in active_track_ids:
                store.pop(track_id, None)


def clear_track_states(detector: MobDetector) -> None:
    """Invalidate all screen-local visual state at an area transition."""
    with _state_lock(detector):
        _state_store(detector).clear()


def track_local(
    detector: MobDetector,
    frame_bgr: np.ndarray,
    mob_name: str,
    track: dict,
    *,
    offset_x: int = 0,
    offset_y: int = 0,
    search_radius_px: int | None = None,
    suppress_positions: list[tuple[int, int]] | None = None,
) -> LocalTrackResult:
    """Update one known Track on one immutable frame.

    Negative IDs are one-shot acquisition IDs. Positive IDs use LK drift first;
    local heatmap recovery is entered only after the cheap drift signal fails.
    """
    track_id = int(track["trackId"])
    cx = int(track["x"])
    cy = int(track["y"])
    if track_id == 0:
        return _miss_result(track_id, cx + offset_x, cy + offset_y, "invalid_track_id")
    scale = float(track.get("scale", 0.0))
    if scale <= 0.0:
        return _miss_result(track_id, cx + offset_x, cy + offset_y, "invalid_scale")
    descriptor = detector.ensure_descriptor(mob_name)

    if track_id < 0:
        hit = _acquire(
            detector, frame_bgr, descriptor, track_id, cx, cy, scale,
            search_radius_px=search_radius_px,
            suppress_positions=suppress_positions,
        )
        if hit is None:
            return _miss_result(track_id, cx + offset_x, cy + offset_y, "no_peak")
        hit_x, hit_y, similarity, bbox = hit
        return _finalize_hit(
            detector, frame_bgr, descriptor, track_id, hit_x, hit_y, bbox, scale,
            offset_x, offset_y, reset_anchor=True, confidence=similarity,
        )

    with _state_lock(detector):
        state = _state_store(detector).get(track_id)

    # A healthy Track's velocity is one-cycle displacement, not pixels/second.
    # Center this frame's crop at the bounded predicted position so LK only
    # solves the residual error. The previous reference crop remains centered
    # on the last accepted hit; translating the current crop by the prediction
    # keeps a mob moving at the predicted speed in the same local coordinates.
    # Do not predict after a miss: recovery must start from the last confirmed
    # anchor plus its own bounded ladder.
    predicted_x, predicted_y = _predicted_center(track, cx, cy, detector)

    if state is not None:
        # Track coordinates are authoritative and may move with the capture
        # ROI/window. Keep the visual state in the current frame coordinate
        # system before attempting LK.
        state.center_x = predicted_x
        state.center_y = predicted_y
        flow = _follow_flow(
            detector, frame_bgr, descriptor, state,
            prediction_dx=predicted_x - cx,
            prediction_dy=predicted_y - cy,
            search_radius_px=search_radius_px,
            suppress_positions=suppress_positions,
        )
        if flow is not None:
            x, y, confidence = flow
            return _finalize_existing_hit(
                detector, frame_bgr, descriptor, track_id, state, x, y,
                confidence, offset_x, offset_y,
            )

        state.recovery_failures += 1
        recovery_radius = _recovery_radius(detector, descriptor, scale, state.recovery_failures)
        # ``state.center_*`` is already the single predicted center for this
        # frame. Do not add velocity again after flow fails; that double-predicts
        # and makes recovery search ahead of the mob.
        origin_x, origin_y = predicted_x, predicted_y
    else:
        recovery_radius = _recovery_radius(detector, descriptor, scale, 1)
        origin_x, origin_y = predicted_x, predicted_y

    recovered = _recover(
        detector, frame_bgr, descriptor, track_id, origin_x, origin_y, scale,
        recovery_radius, suppress_positions,
    )
    if recovered is not None:
        hit_x, hit_y, similarity, bbox = recovered
        return _finalize_hit(
            detector, frame_bgr, descriptor, track_id, hit_x, hit_y, bbox, scale,
            offset_x, offset_y, reset_anchor=state is None,
            preserved_state=state,
            confidence=similarity,
        )

    exhausted = state is not None and state.recovery_failures >= _MAX_RECOVERY_FAILURES
    if exhausted:
        # Tracking owns the terminal decision after the bounded local ladder;
        # do not leave a stale anchor around to repeat an exhausted search.
        discard_track_state(detector, track_id)
    return _miss_result(
        track_id, cx + offset_x, cy + offset_y,
        "tracking_lost" if exhausted else (
            "local_recovery_miss" if state is not None else "acquire_miss"
        ),
        tracking_lost=exhausted,
    )


def _predicted_center(
    track: dict,
    cx: int,
    cy: int,
    detector: MobDetector,
) -> tuple[int, int]:
    """Predict one healthy tracking step from the last confirmed hit.

    ``velX``/``velY`` are smoothed displacement per tracking cycle. Applying
    more than one step would turn a temporary estimate into runaway drift, so
    prediction is bounded to the configured local moving radius and disabled
    after a miss. The appearance tracker then corrects this prediction on the
    same fresh frame.
    """
    if not bool(track.get("prediction_valid", True)):
        return cx, cy
    if int(track.get("lostCount", 0) or 0) > 0:
        return cx, cy
    try:
        vel_x = float(track.get("velX", 0.0) or 0.0)
        vel_y = float(track.get("velY", 0.0) or 0.0)
    except (TypeError, ValueError):
        return cx, cy
    speed = (vel_x * vel_x + vel_y * vel_y) ** 0.5
    if speed < _PREDICTION_MIN_SPEED_PX:
        return cx, cy
    try:
        raw_now_tick = track.get("nowTick")
        raw_updated_tick = track.get("updatedTick")
        now_tick = int(raw_now_tick or 0)
        updated_tick = int(raw_updated_tick or 0)
    except (TypeError, ValueError):
        raw_now_tick = raw_updated_tick = None
        now_tick, updated_tick = 0, 0
    elapsed_ms = now_tick - updated_tick
    # Missing timestamps mean the caller supplied the legacy one-cycle velocity;
    # preserve that nominal step. When both timestamps are valid, scale the
    # displacement to the actual frame age so a slow multi-mob cycle does not
    # trail while the extrapolation remains bounded.
    if raw_now_tick is None or raw_updated_tick is None or now_tick <= 0 or updated_tick <= 0:
        cadence_ratio = 1.0
    else:
        cadence_ratio = max(0.5, min(3.0, max(1, elapsed_ms) / 20.0))
    limit = max(1, int(detector.local_track_moving_search_radius_px))
    vel_x = max(-limit, min(limit, vel_x * cadence_ratio))
    vel_y = max(-limit, min(limit, vel_y * cadence_ratio))
    return int(round(cx + vel_x)), int(round(cy + vel_y))


def _acquire(
    detector: MobDetector,
    frame: np.ndarray,
    descriptor: MobDescriptor,
    track_id: int,
    cx: int,
    cy: int,
    scale: float,
    *,
    search_radius_px: int | None,
    suppress_positions: list[tuple[int, int]] | None,
) -> tuple[int, int, float, tuple[int, int, int, int]] | None:
    static_fast = bool(
        getattr(detector, "use_sprite_grf", False)
        and detector.descriptor_is_static(descriptor)
        and getattr(detector, "grf_local_track_skip_native_gate", True)
    )
    if not static_fast:
        accepted, bbox, similarity = detector.score_at(frame, descriptor, cx, cy, scale)
        if accepted and bbox is not None:
            bx, by, bw, bh = bbox
            x, y = _refine_hit_to_sprite_center(
                detector, frame, descriptor, bx + bw // 2, by + bh // 2, scale,
            )
            return x, y, similarity, bbox
    radius = search_radius_px or _effective_search_radius(detector, descriptor, scale)
    peak = _find_local_peak(
        detector, frame, descriptor, cx, cy, scale,
        search_radius_px=radius, suppress_positions=suppress_positions,
    )
    if peak is None:
        return None
    hit_x, hit_y, _heat, similarity, peak_bbox = peak
    return hit_x, hit_y, similarity, peak_bbox


def _follow_flow(
    detector: MobDetector,
    frame: np.ndarray,
    descriptor: MobDescriptor,
    state: _TrackVisualState,
    *,
    prediction_dx: int = 0,
    prediction_dy: int = 0,
    search_radius_px: int | None,
    suppress_positions: list[tuple[int, int]] | None,
) -> tuple[int, int, float] | None:
    previous = state.previous_gray
    if previous.size == 0 or state.points.shape[0] < _FLOW_MIN_POINTS:
        return None
    half_w = max(_FAST_BBOX_MIN_PX, state.anchor_width // 2)
    half_h = max(_FAST_BBOX_MIN_PX, state.anchor_height // 2)
    current, x0, y0 = _crop_gray(frame, state.center_x, state.center_y, half_w, half_h)
    current_bgr, _bx0, _by0 = _crop_bgr(
        frame, state.center_x, state.center_y, half_w, half_h,
    )
    if current is None or current_bgr is None or current.shape != previous.shape:
        return None
    # ``previous_gray`` is centered on the last accepted hit, while ``current``
    # is centered on the predicted hit. Translate the old feature coordinates
    # into the current crop before LK; otherwise LK compares different screen
    # locations and can manufacture a near-zero/background displacement, leaving
    # the published Track behind a moving mob.
    prediction = np.asarray(
        (float(prediction_dx), float(prediction_dy)), dtype=np.float32,
    )
    previous_points = state.points.astype(np.float32)
    initial_next_points = previous_points - prediction.reshape(1, 1, 2)
    next_points, status, errors = cv2.calcOpticalFlowPyrLK(
        previous, current, previous_points, initial_next_points,
        winSize=(15, 15), maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 12, 0.03),
        flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
    )
    if next_points is None or status is None:
        return None
    valid = status.reshape(-1).astype(bool)
    if errors is not None:
        valid &= errors.reshape(-1) <= _FLOW_MAX_ERROR
    # Features are selected from the sprite-colored region, not arbitrary
    # background corners inside the crop. Recheck the current endpoints too:
    # animated sprites can expose new colors, but stationary background must
    # never dominate the displacement consensus and drag the Track behind.
    valid &= _points_match_sprite(
        current_bgr,
        next_points.reshape(-1, 2),
        descriptor,
        float(detector.config["minSpritePaletteMatch"]),
    )
    if int(valid.sum()) < _FLOW_MIN_POINTS:
        return None
    displacement = (
        next_points.reshape(-1, 2) - initial_next_points.reshape(-1, 2)
    )[valid]
    median = np.median(displacement, axis=0)
    distances = np.linalg.norm(displacement - median, axis=1)
    mad = float(np.median(distances))
    inliers = distances <= max(_FLOW_INLIER_FLOOR_PX, 2.5 * mad + 1.0)
    if int(inliers.sum()) < _FLOW_MIN_POINTS:
        return None
    delta = np.median(displacement[inliers], axis=0)
    # Carry only the robust consensus points into the next reference frame.
    # A point may pass LK's local error check yet belong to stationary scenery;
    # retaining it would reintroduce that outlier on the following cycle.
    inlier_valid = np.zeros_like(valid, dtype=bool)
    valid_indices = np.flatnonzero(valid)
    inlier_valid[valid_indices[inliers]] = True
    max_delta = float(search_radius_px or detector.local_track_moving_search_radius_px)
    if float(np.linalg.norm(delta)) > max_delta * _FLOW_MAX_DISPLACEMENT_MULTIPLIER:
        return None
    candidate_x = int(round(state.center_x + float(delta[0])))
    candidate_y = int(round(state.center_y + float(delta[1])))
    if not _local_identity_ok(detector, frame, descriptor, candidate_x, candidate_y, state.scale):
        return None
    if suppress_positions and _would_swap_ownership(
        state.center_x, state.center_y, candidate_x, candidate_y,
        suppress_positions,
        max(_LOCAL_SUPPRESS_RADIUS_FLOOR_PX, int(max_delta * 0.35)),
    ):
        return None

    # The LK displacement is residual because the crop was centered at the
    # predicted position. Feed the full observed displacement back into the
    # predictor, otherwise a correctly predicted mob would appear stationary
    # and the Track would fall behind on the next frame.
    observed_dx = float(prediction_dx) + float(delta[0])
    observed_dy = float(prediction_dy) + float(delta[1])
    state.velocity_x = 0.65 * state.velocity_x + 0.35 * observed_dx
    state.velocity_y = 0.65 * state.velocity_y + 0.35 * observed_dy
    _advance_flow_reference(
        frame, state, delta, next_points.reshape(-1, 2), inlier_valid,
    )
    confidence = min(1.0, float(inliers.sum()) / float(max(1, len(displacement))))
    return candidate_x, candidate_y, confidence


def _recover(
    detector: MobDetector,
    frame: np.ndarray,
    descriptor: MobDescriptor,
    track_id: int,
    cx: int,
    cy: int,
    scale: float,
    radius: int,
    suppress_positions: list[tuple[int, int]] | None,
) -> tuple[int, int, float, tuple[int, int, int, int]] | None:
    peak = _find_local_peak(
        detector, frame, descriptor, cx, cy, scale,
        search_radius_px=radius, suppress_positions=suppress_positions,
    )
    if peak is None:
        return None
    hit_x, hit_y, _heat, similarity, bbox = peak
    # Recovery is identity-sensitive. A Track with an existing stable anchor
    # must not jump to an equally colored neighbor merely because it wins heat.
    with _state_lock(detector):
        state = _state_store(detector).get(track_id)
    if state is not None and not _anchor_agrees(frame, state, hit_x, hit_y):
        return None
    return hit_x, hit_y, similarity, bbox


def _finalize_hit(
    detector: MobDetector,
    frame: np.ndarray,
    descriptor: MobDescriptor,
    track_id: int,
    hit_x: int,
    hit_y: int,
    bbox: tuple[int, int, int, int],
    scale: float,
    offset_x: int,
    offset_y: int,
    *,
    reset_anchor: bool,
    preserved_state: _TrackVisualState | None = None,
    confidence: float | None = None,
) -> LocalTrackResult:
    center_x, center_y = _refine_hit_to_sprite_center(
        detector, frame, descriptor, hit_x, hit_y, scale,
    )
    if reset_anchor or preserved_state is None:
        state = _make_state(frame, descriptor, center_x, center_y, scale)
    else:
        state = preserved_state
        state.recovery_failures = 0
        state.center_x = center_x
        state.center_y = center_y
        _reanchor_flow_points(frame, descriptor, state, center_x, center_y)
    with _state_lock(detector):
        _state_store(detector)[track_id] = state
    opacity_score = measure_opacity_score(
        frame, descriptor, bbox,
        float(descriptor.max_sprite_palette_distance),
        float(detector.config["minSpritePaletteMatch"]),
    )
    return LocalTrackResult(
        track_id=track_id, found=True,
        x=center_x + offset_x, y=center_y + offset_y,
        confidence=float(confidence if confidence is not None else 0.0),
        miss_reason="", opacity_score=opacity_score,
    )


def _finalize_existing_hit(
    detector: MobDetector,
    frame: np.ndarray,
    descriptor: MobDescriptor,
    track_id: int,
    state: _TrackVisualState,
    x: int,
    y: int,
    confidence: float,
    offset_x: int,
    offset_y: int,
) -> LocalTrackResult:
    state.recovery_failures = 0
    state.center_x, state.center_y = x, y
    with _state_lock(detector):
        _state_store(detector)[track_id] = state
    bbox = _descriptor_sized_bbox(descriptor, x, y, state.scale)
    opacity_score = measure_opacity_score(
        frame, descriptor, bbox,
        float(descriptor.max_sprite_palette_distance),
        float(detector.config["minSpritePaletteMatch"]),
    )
    return LocalTrackResult(
        track_id=track_id, found=True,
        x=x + offset_x, y=y + offset_y,
        confidence=confidence, miss_reason="", opacity_score=opacity_score,
    )


def _make_state(
    frame: np.ndarray,
    descriptor: MobDescriptor,
    cx: int,
    cy: int,
    scale: float,
) -> _TrackVisualState:
    width = max(_FAST_BBOX_MIN_PX, int(round(descriptor.avg_width * scale * _REFINE_WINDOW_SCALE)))
    height = max(_FAST_BBOX_MIN_PX, int(round(descriptor.avg_height * scale * _REFINE_WINDOW_SCALE)))
    patch_bgr, _x0, _y0 = _crop_bgr(frame, cx, cy, width // 2, height // 2)
    if patch_bgr is None:
        patch_bgr = np.zeros((height, width, 3), dtype=np.uint8)
    patch = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)
    points = _feature_points(patch, descriptor, frame_bgr=patch_bgr)
    anchor = patch.copy()
    return _TrackVisualState(
        center_x=cx, center_y=cy, scale=scale,
        anchor_gray=anchor, anchor_width=width, anchor_height=height,
        previous_gray=patch, points=points,
    )


def _reanchor_flow_points(
    frame: np.ndarray,
    descriptor: MobDescriptor,
    state: _TrackVisualState,
    cx: int,
    cy: int,
) -> None:
    patch_bgr, _x0, _y0 = _crop_bgr(
        frame, cx, cy, state.anchor_width // 2, state.anchor_height // 2,
    )
    if patch_bgr is None:
        return
    patch = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)
    state.previous_gray = patch
    state.points = _feature_points(patch, descriptor, frame_bgr=patch_bgr)


def _advance_flow_reference(
    frame: np.ndarray,
    state: _TrackVisualState,
    delta: np.ndarray,
    next_points: np.ndarray,
    valid: np.ndarray,
) -> None:
    """Keep LK points persistent while moving the reference crop."""
    half_w = state.anchor_width // 2
    half_h = state.anchor_height // 2
    current, _x0, _y0 = _crop_gray(
        frame,
        state.center_x + int(round(delta[0])),
        state.center_y + int(round(delta[1])),
        half_w,
        half_h,
    )
    if current is None:
        return
    # Both ``previous_gray`` and ``current`` are centered crops. LK points are
    # therefore already in the correct local coordinate system for the next
    # cycle; preserve them instead of rediscovering background corners.
    state.previous_gray = current
    integer_delta = np.asarray(
        (int(round(float(delta[0]))), int(round(float(delta[1])))),
        dtype=np.float32,
    )
    adjusted = next_points[valid] - integer_delta
    state.points = adjusted.reshape(-1, 1, 2).astype(np.float32)
    if state.points.shape[0] < _FLOW_MIN_POINTS:
        state.points = _feature_points(state.previous_gray, None, frame_bgr=None)


def _would_swap_ownership(
    old_x: int,
    old_y: int,
    new_x: int,
    new_y: int,
    other_positions: list[tuple[int, int]],
    exclusion_radius: int,
) -> bool:
    """Reject a flow result that is materially closer to another Track."""
    own_distance = (new_x - old_x) ** 2 + (new_y - old_y) ** 2
    for other_x, other_y in other_positions:
        other_distance = (new_x - other_x) ** 2 + (new_y - other_y) ** 2
        if other_distance <= exclusion_radius * exclusion_radius and other_distance < own_distance:
            return True
    return False


def _feature_points(
    patch_gray: np.ndarray,
    descriptor: MobDescriptor,
    *,
    frame_bgr: np.ndarray | None,
) -> np.ndarray:
    if patch_gray.size == 0:
        return np.empty((0, 1, 2), dtype=np.float32)
    mask = np.zeros_like(patch_gray, dtype=np.uint8)
    margin_x = max(1, patch_gray.shape[1] // 6)
    margin_y = max(1, patch_gray.shape[0] // 6)
    mask[margin_y : patch_gray.shape[0] - margin_y,
         margin_x : patch_gray.shape[1] - margin_x] = 255
    if descriptor is not None and frame_bgr is not None:
        sprite_heat = sprite_palette_heatmap(
            frame_bgr,
            descriptor.match_palette_bgr,
            float(descriptor.max_sprite_palette_distance),
        )
        sprite_mask = (
            sprite_heat >= _FEATURE_SPRITE_MIN_HEAT
        ).astype(np.uint8) * 255
        sprite_mask = cv2.dilate(sprite_mask, np.ones((3, 3), np.uint8))
        restricted = cv2.bitwise_and(mask, sprite_mask)
        # Keep the center fallback for unusual animated frames where the
        # palette gate has no pixels; losing all features is worse than using
        # the bounded central crop for one cycle.
        if int(np.count_nonzero(restricted)) >= _FLOW_MIN_POINTS:
            mask = restricted
    points = cv2.goodFeaturesToTrack(
        patch_gray, maxCorners=32, qualityLevel=0.01,
        minDistance=3, blockSize=5, mask=mask,
    )
    if points is None:
        return np.empty((0, 1, 2), dtype=np.float32)
    return points.astype(np.float32)


def _points_match_sprite(
    frame_bgr: np.ndarray,
    points: np.ndarray,
    descriptor: MobDescriptor,
    threshold: float,
) -> np.ndarray:
    """Return an endpoint mask for points that remain on sprite-colored pixels."""
    heat = sprite_palette_heatmap(
        frame_bgr,
        descriptor.match_palette_bgr,
        float(descriptor.max_sprite_palette_distance),
    )
    result = np.zeros((points.shape[0],), dtype=bool)
    height, width = heat.shape[:2]
    for index, point in enumerate(points):
        px, py = int(round(float(point[0]))), int(round(float(point[1])))
        if 0 <= px < width and 0 <= py < height:
            result[index] = bool(heat[py, px] >= threshold)
    return result


def _crop_bgr(
    frame: np.ndarray,
    cx: int,
    cy: int,
    half_w: int,
    half_h: int,
) -> tuple[np.ndarray | None, int, int]:
    height, width = frame.shape[:2]
    x0 = int(cx - half_w)
    y0 = int(cy - half_h)
    x1 = int(cx + half_w)
    y1 = int(cy + half_h)
    if x1 <= x0 or y1 <= y0:
        return None, x0, y0

    # Keep a fixed-size crop at the frame boundary. LK points remain in the
    # same ROI-relative coordinate system, while edge mobs do not lose the
    # cheap path merely because part of their search window is off-screen.
    clipped_x0, clipped_y0 = max(0, x0), max(0, y0)
    clipped_x1, clipped_y1 = min(width, x1), min(height, y1)
    if clipped_x1 <= clipped_x0 or clipped_y1 <= clipped_y0:
        return None, x0, y0
    crop = frame[clipped_y0:clipped_y1, clipped_x0:clipped_x1]
    if clipped_x0 == x0 and clipped_y0 == y0 and clipped_x1 == x1 and clipped_y1 == y1:
        return crop, x0, y0
    padded = np.zeros((y1 - y0, x1 - x0, 3), dtype=crop.dtype)
    padded[clipped_y0 - y0 : clipped_y1 - y0, clipped_x0 - x0 : clipped_x1 - x0] = crop
    return padded, x0, y0


def _crop_gray(
    frame: np.ndarray,
    cx: int,
    cy: int,
    half_w: int,
    half_h: int,
) -> tuple[np.ndarray | None, int, int]:
    bgr, x0, y0 = _crop_bgr(frame, cx, cy, half_w, half_h)
    if bgr is None:
        return None, x0, y0
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), x0, y0


def _local_identity_ok(
    detector: MobDetector,
    frame: np.ndarray,
    descriptor: MobDescriptor,
    cx: int,
    cy: int,
    scale: float,
) -> bool:
    bbox = _descriptor_sized_bbox(descriptor, cx, cy, scale)
    x, y, width, height = bbox
    fh, fw = frame.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(fw, x + width), min(fh, y + height)
    if x1 <= x0 or y1 <= y0:
        return False
    region = frame[y0:y1, x0:x1]
    sprite = sprite_palette_heatmap(
        region, descriptor.match_palette_bgr,
        float(descriptor.max_sprite_palette_distance),
    )
    body = palette_heatmap(region, descriptor.body_palette)
    sprite_fraction = float((sprite >= float(detector.config["minSpritePaletteMatch"])).mean())
    body_fraction = float((body >= 0.5).mean())
    return (
        sprite_fraction >= _IDENTITY_MIN_SPRITE_FRACTION
        and body_fraction >= _IDENTITY_MIN_BODY_FRACTION
    )


def _anchor_agrees(
    frame: np.ndarray,
    state: _TrackVisualState,
    cx: int,
    cy: int,
) -> bool:
    patch, _x0, _y0 = _crop_gray(
        frame, cx, cy, state.anchor_width // 2, state.anchor_height // 2,
    )
    if patch is None or state.anchor_gray.size == 0:
        return False
    if patch.shape != state.anchor_gray.shape:
        return False
    target = patch
    if float(target.std()) < 2.0 or float(state.anchor_gray.std()) < 2.0:
        return True
    score = float(cv2.matchTemplate(target, state.anchor_gray, cv2.TM_CCOEFF_NORMED)[0, 0])
    return bool(np.isfinite(score) and score >= _ANCHOR_MIN_SCORE)


def _recovery_radius(
    detector: MobDetector,
    descriptor: MobDescriptor,
    scale: float,
    failure_count: int,
) -> int:
    base = int(detector.local_track_search_radius_px)
    moving = int(detector.local_track_moving_search_radius_px)
    cap = max(moving, int(detector.local_track_max_search_radius_px))
    scaled = int(round(max(descriptor.avg_width, descriptor.avg_height) * scale * 1.5))
    if failure_count <= 1:
        return min(cap, max(base, scaled // 2))
    return min(cap, max(moving, scaled))


def _effective_search_radius(detector: MobDetector, descriptor: MobDescriptor, scale: float) -> int:
    base = int(detector.local_track_moving_search_radius_px)
    multiplier = float(getattr(detector, "local_track_sprite_radius_multiplier", 1.5))
    cap = max(base, int(detector.local_track_max_search_radius_px))
    extent = max(float(descriptor.avg_width), float(descriptor.avg_height)) * max(float(scale), 0.0)
    return min(cap, max(base, int(round(extent * multiplier))))


def _miss_result(
    track_id: int,
    x: int,
    y: int,
    reason: str,
    *,
    tracking_lost: bool = False,
) -> LocalTrackResult:
    return LocalTrackResult(
        track_id=track_id, found=False, x=x, y=y,
        confidence=0.0, miss_reason=reason,
        tracking_lost=tracking_lost,
    )


def _find_local_peak(
    detector: MobDetector,
    frame_bgr: np.ndarray,
    descriptor: MobDescriptor,
    cx: int,
    cy: int,
    scale: float,
    *,
    search_radius_px: int,
    suppress_positions: list[tuple[int, int]] | None = None,
) -> tuple[int, int, float, float, tuple[int, int, int, int]] | None:
    frame_h, frame_w = frame_bgr.shape[:2]
    margin_x = int(round(descriptor.avg_width * scale * _COVERAGE_SIZE_FRAC))
    margin_y = int(round(descriptor.avg_height * scale * _COVERAGE_SIZE_FRAC))
    pad = search_radius_px + max(margin_x, margin_y)
    x0, y0 = max(0, cx - pad), max(0, cy - pad)
    x1, y1 = min(frame_w, cx + pad + 1), min(frame_h, cy + pad + 1)
    if x1 <= x0 or y1 <= y0:
        return None
    crop = frame_bgr[y0:y1, x0:x1]
    pyramid = _LOCAL_HEATMAP_DOWNSCALE if min(crop.shape[:2]) >= 16 else 1
    work_bgr = (
        cv2.resize(crop, (max(1, crop.shape[1] // pyramid), max(1, crop.shape[0] // pyramid)), interpolation=cv2.INTER_AREA)
        if pyramid > 1 else crop
    )
    heat = _build_local_follow_heatmap(detector.heatmap_detector, work_bgr, descriptor, scale / pyramid)
    if heat.size == 0:
        return None
    if suppress_positions:
        suppress_radius = max(_LOCAL_SUPPRESS_RADIUS_FLOOR_PX, search_radius_px // 2)
        for sx, sy in suppress_positions:
            lx, ly = (sx - x0) / pyramid, (sy - y0) / pyramid
            if 0 <= lx < heat.shape[1] and 0 <= ly < heat.shape[0]:
                cv2.circle(heat, (int(round(lx)), int(round(ly))), max(1, suppress_radius // pyramid), 0.0, -1)
    yy, xx = np.ogrid[:heat.shape[0], :heat.shape[1]]
    anchor_x, anchor_y = (cx - x0) / pyramid, (cy - y0) / pyramid
    radius = max(1, int(round(search_radius_px / pyramid)))
    heat = np.where((xx - anchor_x) ** 2 + (yy - anchor_y) ** 2 <= radius * radius, heat, 0.0)
    min_heat = detector.heatmap_detector.min_center_heat * _LOCAL_FOLLOW_MIN_HEAT_FRAC
    for _ in range(2):
        peak_value = float(heat.max())
        if peak_value < min_heat:
            return None
        py, px = np.unravel_index(int(heat.argmax()), heat.shape)
        hit_x = int(round(px * pyramid + x0 + (pyramid - 1) / 2))
        hit_y = int(round(py * pyramid + y0 + (pyramid - 1) / 2))
        static_fast = bool(
            getattr(detector, "use_sprite_grf", False)
            and detector.descriptor_is_static(descriptor)
            and getattr(detector, "grf_local_track_skip_native_gate", True)
        )
        if static_fast:
            bbox = _descriptor_sized_bbox(descriptor, hit_x, hit_y, scale)
            if _local_identity_ok(detector, frame_bgr, descriptor, hit_x, hit_y, scale):
                return hit_x, hit_y, peak_value, float(peak_value), bbox
        else:
            accepted, bbox, similarity = detector.score_at(
                frame_bgr, descriptor, hit_x, hit_y, scale,
            )
            if accepted and bbox is not None:
                return hit_x, hit_y, peak_value, float(similarity), bbox
        cv2.circle(heat, (int(px), int(py)), max(1, radius // 3), 0.0, -1)
    return None


def _build_local_follow_heatmap(heatmap_detector, crop_bgr: np.ndarray, descriptor: MobDescriptor, scale: float) -> np.ndarray:
    sprite = sprite_palette_heatmap(crop_bgr, descriptor.match_palette_bgr, descriptor.max_sprite_palette_distance)
    body = palette_heatmap(crop_bgr, descriptor.body_palette)
    accent = palette_heatmap(crop_bgr, descriptor.accent_colors)
    color = np.maximum(body * _LOCAL_FOLLOW_BODY_W, accent * _LOCAL_FOLLOW_ACCENT_W)
    final = np.zeros(crop_bgr.shape[:2], dtype=np.float32)
    for track_scale in _local_follow_scales(heatmap_detector._center_scales(crop_bgr.shape[1]), scale):
        window = (max(3, int(round(descriptor.avg_width * track_scale)) | 1), max(3, int(round(descriptor.avg_height * track_scale)) | 1))
        final = np.maximum(final, np.maximum(
            cv2.blur(sprite, window) * _LOCAL_FOLLOW_SPRITE_W,
            cv2.blur(color, window) * _LOCAL_FOLLOW_COLOR_W,
        ).astype(np.float32))
    return final


def _local_follow_scales(center_scales: list[float], track_scale: float) -> list[float]:
    scales = [float(track_scale)]
    distinct = [float(value) for value in center_scales if abs(float(value) - float(track_scale)) > 1e-9]
    if distinct:
        scales.append(min(distinct, key=lambda value: abs(value - float(track_scale))))
    return scales


def _descriptor_sized_bbox(descriptor: MobDescriptor, cx: int, cy: int, scale: float) -> tuple[int, int, int, int]:
    width = max(_FAST_BBOX_MIN_PX, int(round(descriptor.avg_width * scale)))
    height = max(_FAST_BBOX_MIN_PX, int(round(descriptor.avg_height * scale)))
    return int(round(cx - width / 2)), int(round(cy - height / 2)), width, height


def _refine_hit_to_sprite_center(
    detector: MobDetector,
    frame_bgr: np.ndarray,
    descriptor: MobDescriptor,
    cx: int,
    cy: int,
    scale: float,
) -> tuple[int, int]:
    width = max(_FAST_BBOX_MIN_PX, int(round(descriptor.avg_width * scale * _REFINE_WINDOW_SCALE)))
    height = max(_FAST_BBOX_MIN_PX, int(round(descriptor.avg_height * scale * _REFINE_WINDOW_SCALE)))
    fh, fw = frame_bgr.shape[:2]
    x0, y0 = max(0, int(cx - width / 2)), max(0, int(cy - height / 2))
    x1, y1 = min(fw, x0 + width), min(fh, y0 + height)
    if x1 <= x0 or y1 <= y0:
        return int(cx), int(cy)
    region = frame_bgr[y0:y1, x0:x1]
    heat = sprite_palette_heatmap(region, descriptor.match_palette_bgr, float(descriptor.max_sprite_palette_distance))
    mask = heat >= float(detector.config["minSpritePaletteMatch"])
    if not np.any(mask):
        return int(cx), int(cy)
    local_x, local_y = int(cx) - x0, int(cy) - y0
    nlab, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    chosen = int(labels[local_y, local_x]) if nlab > 1 and 0 <= local_x < mask.shape[1] and 0 <= local_y < mask.shape[0] else 0
    if chosen <= 0 and nlab > 1:
        chosen = max(range(1, nlab), key=lambda label: int(stats[label, cv2.CC_STAT_AREA]))
    if chosen <= 0:
        return int(cx), int(cy)
    ys, xs = np.where(labels == chosen)
    return x0 + int(round((xs.min() + xs.max()) / 2)), y0 + int(round((ys.min() + ys.max()) / 2))
