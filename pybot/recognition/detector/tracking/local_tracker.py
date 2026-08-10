"""Local coordinate follower for already-discovered tracks.

Deterministic follow around the predicted position:
1. Build one lightweight local color/sprite heatmap on a small image pyramid.
2. Search the strongest local peak(s) around a bounded recovery prediction.
3. Verify only the winning peak(s) with the native-resolution silhouette gate.

The expensive gate is deliberately not run at the old center first: that center
is stale for moving mobs and doing so duplicated the largest part of every
tracking tick.    Tracking is pure follow for position and reports terminal local loss;
    Discovery remains an independent validation/removal observer.
Warm tracking only publishes fresh coordinates; discovery owns absence/death validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np

from pybot.recognition.detector.descriptors.descriptor import MobDescriptor
from pybot.recognition.detector.scoring.heatmap_detector import (
    _COVERAGE_SIZE_FRAC,
    palette_heatmap,
    sprite_palette_heatmap,
)

if TYPE_CHECKING:
    from pybot.recognition.detector.detector import MobDetector

_LOCAL_SUPPRESS_RADIUS_FLOOR_PX = 8
# Same floor as detector._MIN_DESCRIPTOR_PX (kept local to avoid a circular
# import from detector.py which lazily imports this module).
_FAST_BBOX_MIN_PX = 8
# Fast-path peaks must clear a multiple of the local-follow heat floor before
# they are accepted WITHOUT the native silhouette gate. A real hit is a strong
# palette peak; this rejects weak marginal blobs that a red-tinted terrain
# fragment could produce.
_LOCAL_FAST_MIN_HEAT_MULT = 2.0
# Keep re-anchoring to one small descriptor-sized local window. The center
# calculation is independent of the temporal anchors and remains fail-closed.
_REFINE_WINDOW_SCALE = 1.25
_LOCAL_CROSS_TRACK_SUPPRESS_DIV = 2
_LOCAL_FOLLOW_MIN_HEAT_FRAC = 0.5
_LOCAL_FOLLOW_BODY_W = 0.55
_LOCAL_FOLLOW_ACCENT_W = 0.45
_LOCAL_FOLLOW_SPRITE_W = 0.75
_LOCAL_FOLLOW_COLOR_W = 0.55
# Tracking has one deterministic peak decision. Large sprites (especially
# Anubis) use a 2x image pyramid to keep local palette work bounded; normal
# animated sprites use the native gate only during initial acquisition.
_LOCAL_HEATMAP_DOWNSCALE = 2
# Temporal-follow correlation runs on a half-resolution grayscale patch. It is
# deliberately independent of the full detector so a confirmed track does not
# pay the discovery/silhouette cost on every frame.
_TEMPLATE_DOWNSCALE = 2
_TEMPLATE_MIN_SCORE = 0.42
_TEMPLATE_MIN_STD = 3.0
# The grayscale anchor is kept stable; local palette heat is used only for
# the current-frame center correction, never to rebuild the anchor.
_LOCAL_RECOVERY_RADIUS_MULTIPLIER = 2
_LOCAL_TRACK_LOST_MISSES = 8


@dataclass(frozen=True)
class _TrackAnchor:
    """One small visual anchor inside a confirmed sprite patch."""

    image_gray: np.ndarray
    # Native-pixel offset from the full sprite patch center.
    offset_x: int
    offset_y: int


@dataclass(frozen=True)
class _TrackTemplate:
    # Several independent anchors keep tracking alive when the game cursor
    # occludes one part of the sprite. The full patch dimensions are retained
    # only to rebuild the candidate sprite window after anchor consensus.
    anchors: tuple[_TrackAnchor, ...]
    # Native-frame dimensions and the sprite center inside the cached patch.
    width: int
    height: int
    center_x: int
    center_y: int
    scale: float


# Keep anchors away from the sprite center: the large game cursor most often
# covers that area. Four independent corners are enough to survive one blocked
# region while keeping the hot path bounded.
_ANCHOR_GRID = ((0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75))
_ANCHOR_MIN_NATIVE_PX = 6
_ANCHOR_MAX_COUNT = 4
_ANCHOR_CONSENSUS_RADIUS_PX = 8
_ANCHOR_MIN_COUNT = 2
_CENTER_HOLE_MAX_GAP_PX = 8


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
    """Follow one known track near its last known center.

    Negative IDs are provisional and perform one-shot local acquisition.
    Positive IDs require a transferred cached template and perform only the
    temporal follow. Zero is invalid and returns a miss.

    ``suppress_positions``: ROI-relative (x, y) of other tracks whose heat
    signature should be suppressed in the peak search. Prevents track A from
    locking onto mob B when two mobs are close together.
    """
    track_id = int(track["trackId"])
    if track_id == 0:
        return _miss_result(
            track_id=track_id,
            x=int(track["x"]) + offset_x,
            y=int(track["y"]) + offset_y,
            reason="invalid_track_id",
        )
    cx = int(track["x"])
    cy = int(track["y"])
    scale = float(track.get("scale", 0.0))
    if scale <= 0.0:
        return _miss_result(
            track_id=track_id,
            x=cx + offset_x,
            y=cy + offset_y,
            reason="invalid_scale",
        )

    # Use the configured runner radius as a floor, then give unusually large
    # sprites a proportionally larger bounded search disk. A large mob can
    # move farther between tracker frames while kiting, and its animation can
    # also shift the strongest heat peak away from the last center. Keep an
    # explicit caller radius authoritative for tests/specialized callers.
    descriptor = detector.ensure_descriptor(mob_name)
    if search_radius_px is not None:
        radius = int(search_radius_px)
    else:
        radius = _effective_search_radius(detector, descriptor, scale)

    screen_cx = cx + offset_x
    screen_cy = cy + offset_y

    # ``velX``/``velY`` are smoothed frame displacements from the shared track
    # state. Lead the search during a short miss streak so a kiting mob stays
    # near the center of the crop instead of reaching its outer edge. Stale
    # callers can explicitly disable this with ``prediction_valid=False``;
    # ``lost_count`` carries the bounded recovery horizon.
    prediction_valid = track.get("prediction_valid", True) is not False
    anchor_required = bool(track.get("anchor_required", False))
    lost_count = max(0, int(track.get("lost_count", 0)))
    prediction_dx = float(track.get("velX", 0.0)) if prediction_valid else 0.0
    prediction_dy = float(track.get("velY", 0.0)) if prediction_valid else 0.0
    # During the bounded recovery ladder, keep leading in the same direction.
    # The predicted center is only a search hint; the last confirmed center
    # remains the identity origin below, so prediction cannot accumulate a swap.
    prediction_horizon = min(3, lost_count + 1) if prediction_valid else 0
    prediction_dx *= prediction_horizon
    prediction_dy *= prediction_horizon
    prediction_len = (prediction_dx * prediction_dx + prediction_dy * prediction_dy) ** 0.5
    if prediction_len > float(radius) and prediction_len > 0.0:
        factor = float(radius) / prediction_len
        prediction_dx *= factor
        prediction_dy *= factor
    search_cx = int(round(cx + prediction_dx))
    search_cy = int(round(cy + prediction_dy))
    identity_cx = cx
    identity_cy = cy

    # Provisional IDs use the local heatmap once; the resulting hit becomes
    # the stable positive-track anchor after commit. A positive Track that has
    # lost its anchor must not silently reacquire a nearby identical mob through
    # the generic detector path.
    template_store = _template_store(detector)
    if track_id < 0 or (track_id not in template_store and not anchor_required):
        peak = _find_local_peak(
            detector,
            frame_bgr,
            descriptor,
            search_cx,
            search_cy,
            scale,
            search_radius_px=radius,
            suppress_positions=suppress_positions,
        )
        if peak is None:
            return _miss_result(
                track_id=track_id,
                x=screen_cx,
                y=screen_cy,
                reason="no_peak",
                tracking_lost=False,
            )
        _peak_x, _peak_y, _heat_score, peak_sim, peak_bbox = peak
        result = _finalize_track_hit(
            detector=detector,
            frame_bgr=frame_bgr,
            descriptor=descriptor,
            track_id=track_id,
            bbox=peak_bbox,
            similarity=peak_sim,
            scale=scale,
            offset_x=offset_x,
            offset_y=offset_y,
        )
        if result is None:
            return _miss_result(
                track_id=track_id,
                x=screen_cx,
                y=screen_cy,
                reason="center_projection_failed",
                tracking_lost=False,
            )
        _track_miss_store(detector).pop(track_id, None)
        return result

    template_hit = _follow_cached_template(
        detector,
        frame_bgr,
        descriptor,
        track_id=track_id,
        cx=search_cx,
        cy=search_cy,
        scale=scale,
        search_radius_px=radius,
        suppress_positions=suppress_positions,
        offset_x=offset_x,
        offset_y=offset_y,
        identity_cx=identity_cx,
        identity_cy=identity_cy,
        identity_radius_px=radius,
    )
    if template_hit is None and track_id in template_store:
        # The warm anchor missed. Search farther for the SAME cached template;
        # do not fall back to a generic peak, because that can swap onto a
        # nearby identical mob.
        recovery_radius = min(
            int(getattr(detector, "local_track_max_search_radius_px", radius)),
            max(radius + 1, radius * _LOCAL_RECOVERY_RADIUS_MULTIPLIER),
        )
        recovered = _follow_cached_template(
            detector,
            frame_bgr,
            descriptor,
            track_id=track_id,
            cx=search_cx,
            cy=search_cy,
            scale=scale,
            search_radius_px=recovery_radius,
            suppress_positions=suppress_positions,
            offset_x=offset_x,
            offset_y=offset_y,
            # Re-center the identity corridor on the bounded motion
            # prediction, not on the stale pre-miss coordinate. The search may
            # expand to recovery_radius, but the accepted template hit must
            # remain near the predicted position; this retains fast mobs while
            # preventing an identical neighbor at the far edge of the expanded
            # crop from winning merely because its grayscale match is stronger.
            identity_cx=search_cx,
            identity_cy=search_cy,
            identity_radius_px=radius,
        )
        if recovered is not None:
            _track_miss_store(detector).pop(track_id, None)
            return recovered
        misses = _track_miss_store(detector)
        count = int(misses.get(track_id, 0)) + 1
        misses[track_id] = count
        return _miss_result(
            track_id=track_id,
            x=screen_cx,
            y=screen_cy,
            reason="template_miss",
            tracking_lost=count >= _LOCAL_TRACK_LOST_MISSES,
        )
    if template_hit is not None:
        _track_miss_store(detector).pop(track_id, None)
        return template_hit
    misses = _track_miss_store(detector)
    count = int(misses.get(track_id, 0)) + 1
    misses[track_id] = count
    return _miss_result(
        track_id=track_id,
        x=screen_cx,
        y=screen_cy,
        reason="anchor_missing",
        tracking_lost=count >= _LOCAL_TRACK_LOST_MISSES,
    )


def _effective_search_radius(
    detector: MobDetector,
    descriptor: MobDescriptor,
    scale: float,
) -> int:
    """Return a bounded local-follow radius appropriate for sprite size.

    The normal moving radius remains the floor for ordinary mobs. For larger
    sprites, search at least ``multiplier * rendered_extent`` pixels so the
    follow window reflects the object's visual footprint rather than treating
    every mob as the same size. The hard cap prevents a malformed descriptor
    from turning one local pass into an unbounded frame scan.
    """
    base = int(detector.local_track_moving_search_radius_px)
    multiplier = float(detector.local_track_sprite_radius_multiplier)
    cap = max(base, int(detector.local_track_max_search_radius_px))
    rendered_extent = max(
        float(descriptor.avg_width), float(descriptor.avg_height),
    ) * max(float(scale), 0.0)
    scaled_radius = int(round(rendered_extent * multiplier))
    return min(cap, max(base, scaled_radius))


def _miss_result(
    *, track_id: int, x: int, y: int, reason: str,
    confidence: float = 0.0,
    tracking_lost: bool = False,
) -> LocalTrackResult:
    return LocalTrackResult(
        track_id=track_id, found=False, x=x, y=y,
        confidence=confidence, miss_reason=reason,
        tracking_lost=tracking_lost,
    )


def _finalize_track_hit(
    *,
    detector: MobDetector,
    frame_bgr: np.ndarray,
    descriptor: MobDescriptor,
    track_id: int,
    bbox: tuple[int, int, int, int],
    similarity: float,
    scale: float,
    offset_x: int, offset_y: int,
) -> LocalTrackResult | None:
    bx, by, bw, bh = bbox
    if bw <= 0 or bh <= 0:
        return None
    # ``score_at`` already paid for the native silhouette gate. Re-anchor its
    # current-frame extract center once more through the same cheap local mask
    # used by warm tracking; no unverified acquisition coordinate is publishable.
    raw_x = bx + bw // 2
    raw_y = by + bh // 2
    projected = _refine_hit_to_sprite_center(
        detector, frame_bgr, descriptor, raw_x, raw_y, scale,
    )
    if projected is None:
        return None
    x = projected[0] + offset_x
    y = projected[1] + offset_y

    # Provisional acquisition creates the first stable identity anchor. Warm
    # tracking re-anchors the published coordinate on every fresh frame without
    # replacing this template. Death/absence validation is discovery-owned.
    if track_id not in _template_store(detector):
        _remember_track_template(
            detector,
            track_id=track_id,
            frame_bgr=frame_bgr,
            bbox=bbox,
            scale=scale,
        )

    return LocalTrackResult(
        track_id=track_id, found=True, x=x, y=y,
        confidence=similarity, miss_reason="",
        opacity_score=0.0,
    )


def _template_store(detector: MobDetector) -> dict[int, _TrackTemplate]:
    """Get the per-detector temporal cache without global cross-session state."""
    store = getattr(detector, "_local_track_templates", None)
    if store is None:
        store = {}
        setattr(detector, "_local_track_templates", store)
    return store


def _track_miss_store(detector: MobDetector) -> dict[int, int]:
    store = getattr(detector, "_local_track_misses", None)
    if store is None:
        store = {}
        setattr(detector, "_local_track_misses", store)
    return store


def transfer_track_template(
    detector: MobDetector,
    source_track_id: int,
    target_track_id: int,
) -> bool:
    """Move a provisional template to the real track ID exactly once."""
    if source_track_id == target_track_id:
        return source_track_id in _template_store(detector)
    store = _template_store(detector)
    template = store.pop(source_track_id, None)
    if template is None:
        return False
    store[target_track_id] = template
    return True


def discard_track_template(detector: MobDetector, track_id: int) -> None:
    """Discard a provisional template that was not committed to a track."""
    _template_store(detector).pop(track_id, None)


def transfer_track_state(
    detector: MobDetector,
    source_track_id: int,
    target_track_id: int,
) -> bool:
    """Current session API: transfer the provisional visual anchor."""
    return transfer_track_template(detector, source_track_id, target_track_id)


def discard_track_state(detector: MobDetector, track_id: int) -> None:
    """Current session API: discard an uncommitted visual anchor."""
    discard_track_template(detector, track_id)
    _track_miss_store(detector).pop(track_id, None)


def prune_track_states(detector: MobDetector, active_track_ids: set[int]) -> None:
    """Drop anchors for Tracks removed by the runtime store."""
    store = _template_store(detector)
    for track_id in tuple(store):
        if track_id not in active_track_ids:
            store.pop(track_id, None)
    misses = _track_miss_store(detector)
    for track_id in tuple(misses):
        if track_id not in active_track_ids:
            misses.pop(track_id, None)


def clear_track_states(detector: MobDetector) -> None:
    """Clear all screen-local anchors at an area/session reset."""
    _template_store(detector).clear()
    _track_miss_store(detector).clear()


def _remember_track_template(
    detector: MobDetector,
    *,
    track_id: int,
    frame_bgr: np.ndarray,
    bbox: tuple[int, int, int, int],
    scale: float,
) -> None:
    """Cache a native patch from a confirmed hit for the next fresh frame."""
    x, y, width, height = (int(v) for v in bbox)
    frame_h, frame_w = frame_bgr.shape[:2]
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(frame_w, x + max(0, width))
    y1 = min(frame_h, y + max(0, height))
    if x1 <= x0 or y1 <= y0:
        return
    patch = frame_bgr[y0:y1, x0:x1]
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    if gray.size == 0 or float(gray.std()) < _TEMPLATE_MIN_STD:
        return

    patch_h, patch_w = gray.shape[:2]
    anchor_w = min(
        patch_w,
        max(_ANCHOR_MIN_NATIVE_PX, int(round(patch_w / 3.0))),
    )
    anchor_h = min(
        patch_h,
        max(_ANCHOR_MIN_NATIVE_PX, int(round(patch_h / 3.0))),
    )
    anchors_with_quality: list[tuple[float, _TrackAnchor]] = []
    for rel_x, rel_y in _ANCHOR_GRID:
        center_x = int(round(rel_x * max(0, patch_w - 1)))
        center_y = int(round(rel_y * max(0, patch_h - 1)))
        ax0 = max(0, min(patch_w - anchor_w, center_x - anchor_w // 2))
        ay0 = max(0, min(patch_h - anchor_h, center_y - anchor_h // 2))
        anchor_gray = gray[ay0:ay0 + anchor_h, ax0:ax0 + anchor_w]
        quality = float(anchor_gray.std())
        if anchor_gray.size == 0 or quality < _TEMPLATE_MIN_STD:
            continue
        reduced = cv2.resize(
            anchor_gray,
            (
                max(1, anchor_gray.shape[1] // _TEMPLATE_DOWNSCALE),
                max(1, anchor_gray.shape[0] // _TEMPLATE_DOWNSCALE),
            ),
            interpolation=cv2.INTER_AREA,
        )
        anchors_with_quality.append((
            quality,
            _TrackAnchor(
                image_gray=reduced,
                offset_x=int(round(ax0 + anchor_gray.shape[1] / 2.0 - patch_w / 2.0)),
                offset_y=int(round(ay0 + anchor_gray.shape[0] / 2.0 - patch_h / 2.0)),
            ),
        ))

    if not anchors_with_quality:
        return
    # Prefer the most textured independent regions. Keeping the set bounded is
    # important because this runs once per acquisition, not on every tick.
    anchors_with_quality.sort(key=lambda item: item[0], reverse=True)
    anchors = tuple(anchor for _quality, anchor in anchors_with_quality[:_ANCHOR_MAX_COUNT])
    store = _template_store(detector)
    store[track_id] = _TrackTemplate(
        anchors=anchors,
        width=x1 - x0,
        height=y1 - y0,
        center_x=(x0 + x1) // 2,
        center_y=(y0 + y1) // 2,
        scale=scale,
    )
    # The anchor set is stable for the Track. Runtime lifecycle pruning removes
    # it when the Track disappears; never replace it with cursor-corrupted
    # current-frame pixels.


def _follow_cached_template(
    detector: MobDetector,
    frame_bgr: np.ndarray,
    descriptor: MobDescriptor,
    *,
    track_id: int,
    cx: int,
    cy: int,
    scale: float,
    search_radius_px: int,
    suppress_positions: list[tuple[int, int]] | None,
    offset_x: int,
    offset_y: int,
    identity_cx: int,
    identity_cy: int,
    identity_radius_px: int,
) -> LocalTrackResult | None:
    """Follow a previously confirmed patch; return None when reacquisition is needed."""
    template = _template_store(detector).get(track_id)
    if template is None:
        return None
    anchors = tuple(template.anchors)
    if not anchors:
        return None

    frame_h, frame_w = frame_bgr.shape[:2]
    margin_x = int(round(descriptor.avg_width * scale * _COVERAGE_SIZE_FRAC))
    margin_y = int(round(descriptor.avg_height * scale * _COVERAGE_SIZE_FRAC))
    pad = search_radius_px + max(margin_x, margin_y)
    x0 = max(0, int(cx - pad))
    y0 = max(0, int(cy - pad))
    x1 = min(frame_w, int(cx + pad + 1))
    y1 = min(frame_h, int(cy + pad + 1))
    if x1 <= x0 or y1 <= y0:
        return None
    crop = frame_bgr[y0:y1, x0:x1]
    crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    work = cv2.resize(
        crop_gray,
        (max(1, crop_gray.shape[1] // _TEMPLATE_DOWNSCALE),
         max(1, crop_gray.shape[0] // _TEMPLATE_DOWNSCALE)),
        interpolation=cv2.INTER_AREA,
    )
    if float(work.std()) < _TEMPLATE_MIN_STD:
        return None

    identity_radius = max(1, int(identity_radius_px))
    candidate_hits: list[tuple[int, int, float]] = []
    score_x = np.arange(work.shape[1], dtype=np.float32)[None, :]
    score_y = np.arange(work.shape[0], dtype=np.float32)[:, None]
    for anchor in anchors:
        anchor_gray = anchor.image_gray
        ah, aw = anchor_gray.shape[:2]
        if anchor_gray.size == 0 or aw > work.shape[1] or ah > work.shape[0]:
            continue
        if float(anchor_gray.std()) < _TEMPLATE_MIN_STD:
            continue
        scores = cv2.matchTemplate(work, anchor_gray, cv2.TM_CCOEFF_NORMED)
        score_h, score_w = scores.shape[:2]
        # Each anchor proposes the full sprite center from its own matched
        # center minus its fixed offset. This is the identity-constrained
        # equivalent of tracking several small pixels/boxes independently.
        candidate_x = np.broadcast_to(
            x0 + (score_x[:, :score_w] + aw / 2.0) * _TEMPLATE_DOWNSCALE
            - float(anchor.offset_x),
            (score_h, score_w),
        )
        candidate_y = np.broadcast_to(
            y0 + (score_y[:score_h, :] + ah / 2.0) * _TEMPLATE_DOWNSCALE
            - float(anchor.offset_y),
            (score_h, score_w),
        )
        valid = (
            (candidate_x - identity_cx) ** 2
            + (candidate_y - identity_cy) ** 2
            <= float(identity_radius * identity_radius)
        )
        valid_scores = np.array(scores, copy=True)
        valid_scores[~valid] = -np.inf
        _min_val, max_val, _min_loc, max_loc = cv2.minMaxLoc(valid_scores)
        if np.isfinite(max_val) and max_val >= _TEMPLATE_MIN_SCORE:
            hit_x = int(round(candidate_x[max_loc[1], max_loc[0]]))
            hit_y = int(round(candidate_y[max_loc[1], max_loc[0]]))
            candidate_hits.append((hit_x, hit_y, float(max_val)))

    if not candidate_hits:
        return None

    # A cursor-corrupted anchor is simply absent. Select the largest tight
    # consensus among the surviving anchors; a single valid anchor is still
    # useful when the cursor covers the rest, but it remains identity-bounded.
    best_cluster: list[tuple[int, int, float]] = []
    best_score = float("-inf")
    for seed in candidate_hits:
        cluster = [
            candidate for candidate in candidate_hits
            if (candidate[0] - seed[0]) ** 2 + (candidate[1] - seed[1]) ** 2
            <= _ANCHOR_CONSENSUS_RADIUS_PX ** 2
        ]
        cluster_score = float(sum(candidate[2] for candidate in cluster))
        if (len(cluster), cluster_score) > (len(best_cluster), best_score):
            best_cluster = cluster
            best_score = cluster_score
    if len(best_cluster) < _ANCHOR_MIN_COUNT:
        return None
    hit_x = int(round(float(np.median([candidate[0] for candidate in best_cluster]))))
    hit_y = int(round(float(np.median([candidate[1] for candidate in best_cluster]))))
    max_val = max(candidate[2] for candidate in best_cluster)
    # Every successful warm frame must be anchored by the current sprite mask.
    # No raw template coordinate is publishable.
    projected = _refine_hit_to_sprite_center(
        detector, frame_bgr, descriptor, hit_x, hit_y, scale,
    )
    if projected is None:
        return None
    hit_x, hit_y = projected
    if (
        (hit_x - identity_cx) ** 2 + (hit_y - identity_cy) ** 2
        > float(identity_radius * identity_radius)
    ):
        return None
    native_w = max(1, int(template.width))
    native_h = max(1, int(template.height))
    if suppress_positions:
        suppress_radius = max(_LOCAL_SUPPRESS_RADIUS_FLOOR_PX, search_radius_px // 3)
        if any((hit_x - sx) ** 2 + (hit_y - sy) ** 2 <= suppress_radius ** 2
               for sx, sy in suppress_positions):
            return None

    bbox = (
        hit_x - native_w // 2,
        hit_y - native_h // 2,
        native_w,
        native_h,
    )
    # Keep the original template stable. Replacing it on every frame causes
    # template drift; only the current-frame palette bbox may publish a center.
    return LocalTrackResult(
        track_id=track_id,
        found=True,
        x=hit_x + offset_x,
        y=hit_y + offset_y,
        confidence=float(max_val),
        miss_reason="",
        opacity_score=0.0,
    )


def _find_local_peak(
    detector: MobDetector,
    frame_bgr: np.ndarray,
    descriptor: MobDescriptor,
    cx: int, cy: int,
    scale: float,
    *,
    search_radius_px: int,
    suppress_positions: list[tuple[int, int]] | None = None,
) -> tuple[int, int, float, float, tuple[int, int, int, int]] | None:
    frame_h, frame_w = frame_bgr.shape[:2]
    margin_x = int(round(descriptor.avg_width * scale * _COVERAGE_SIZE_FRAC))
    margin_y = int(round(descriptor.avg_height * scale * _COVERAGE_SIZE_FRAC))
    pad = search_radius_px + max(margin_x, margin_y)
    x0 = max(0, cx - pad)
    y0 = max(0, cy - pad)
    x1 = min(frame_w, cx + pad + 1)
    y1 = min(frame_h, cy + pad + 1)
    if x1 <= x0 or y1 <= y0:
        return None

    crop_bgr = frame_bgr[y0:y1, x0:x1]
    pyramid = _LOCAL_HEATMAP_DOWNSCALE
    if min(crop_bgr.shape[:2]) < pyramid * 8:
        pyramid = 1
    if pyramid > 1:
        work_bgr = cv2.resize(
            crop_bgr,
            (
                max(1, crop_bgr.shape[1] // pyramid),
                max(1, crop_bgr.shape[0] // pyramid),
            ),
            interpolation=cv2.INTER_AREA,
        )
    else:
        work_bgr = crop_bgr
    local_final = _build_local_follow_heatmap(
        detector.heatmap_detector,
        work_bgr,
        descriptor,
        scale / pyramid,
    )
    if local_final.size == 0:
        return None

    # Suppress heat near other tracks so each track grabs its own mob.
    if suppress_positions:
        suppress_radius = max(
            _LOCAL_SUPPRESS_RADIUS_FLOOR_PX,
            search_radius_px // _LOCAL_CROSS_TRACK_SUPPRESS_DIV,
        )
        for sx, sy in suppress_positions:
            lsx = (sx - x0) / pyramid
            lsy = (sy - y0) / pyramid
            if 0 <= lsx < local_final.shape[1] and 0 <= lsy < local_final.shape[0]:
                cv2.circle(
                    local_final,
                    (int(round(lsx)), int(round(lsy))),
                    max(1, int(round(suppress_radius / pyramid))),
                    0.0,
                    thickness=-1,
                )

    anchor_x = (cx - x0) / pyramid
    anchor_y = (cy - y0) / pyramid
    yy, xx = np.ogrid[: local_final.shape[0], : local_final.shape[1]]
    dist_sq = (xx - anchor_x) ** 2 + (yy - anchor_y) ** 2
    radius_work = max(1, int(round(search_radius_px / pyramid)))
    mask = dist_sq <= (radius_work * radius_work)
    work = np.where(mask, local_final, 0.0).copy()
    min_heat = detector.heatmap_detector.min_center_heat * _LOCAL_FOLLOW_MIN_HEAT_FRAC
    peak_val = float(work.max())
    if peak_val < min_heat:
        return None
    peak_y_local, peak_x_local = np.unravel_index(int(work.argmax()), work.shape)
    peak_x = int(round(peak_x_local * pyramid + x0 + (pyramid - 1) / 2))
    peak_y = int(round(peak_y_local * pyramid + y0 + (pyramid - 1) / 2))
    if _fast_track_accept(detector, descriptor):
            # Static modified sprites are already palette-driven, so accept the
            # local peak without the expensive native silhouette gate. The peak
            # must clear a strong heat multiple to avoid weak terrain fragments.
            # Every successful local peak must be anchored by the current
            # sprite mask. An unanchored heat peak is a miss, never a fallback
            # coordinate.
            projected = _refine_hit_to_sprite_center(
                detector, frame_bgr, descriptor, peak_x, peak_y, scale,
            )
            if projected is None:
                return None
            peak_x, peak_y = projected
            bbox = _descriptor_sized_bbox(descriptor, peak_x, peak_y, scale)
            if bbox is not None and peak_val >= _LOCAL_FAST_MIN_HEAT_MULT * min_heat:
                return peak_x, peak_y, peak_val, float(peak_val), bbox
    else:
        accepted, bbox, sim = detector.score_at(
            frame_bgr, descriptor, peak_x, peak_y, scale,
        )
        if accepted and bbox is not None:
            # Native gate returns the best current sprite extract. Carry its
            # center forward instead of retaining the heatmap peak offset.
            bx, by, bw, bh = bbox
            return bx + bw // 2, by + bh // 2, peak_val, sim, bbox
    return None


def _fast_track_accept(
    detector: MobDetector,
    descriptor: MobDescriptor,
) -> bool:
    """True when local follow may accept a peak without the native silhouette gate.

    Modified sprite.grf assets are a single deterministic static frame with a
    distinctive red palette. The heatmap peak is already palette-driven, so the
    native-resolution verify adds little for a static descriptor while costing
    a large part of every tracking tick on big sprites (Anubis). Enabled only
    when the mode flag is set AND the descriptor is truly single-frame, so the
    animated-original path keeps its full verification.
    """
    return (
        detector.use_sprite_grf
        and bool(getattr(detector, "grf_local_track_skip_native_gate", True))
        and detector.descriptor_is_static(descriptor)
    )


def _descriptor_sized_bbox(
    descriptor: MobDescriptor,
    cx: int,
    cy: int,
    scale: float,
) -> tuple[int, int, int, int] | None:
    """Descriptor-sized window around a peak (mirrors ``score_at``'s crop)."""
    w = max(_FAST_BBOX_MIN_PX, int(round(descriptor.avg_width * scale)))
    h = max(_FAST_BBOX_MIN_PX, int(round(descriptor.avg_height * scale)))
    x = int(round(cx - w / 2))
    y = int(round(cy - h / 2))
    return (x, y, w, h)


def _refine_hit_to_sprite_center(
    detector: MobDetector,
    frame_bgr: np.ndarray,
    descriptor: MobDescriptor,
    cx: int,
    cy: int,
    scale: float,
) -> tuple[int, int] | None:
    """Re-anchor a hit to the current sprite-colored bbox center.

    This is one bounded local palette pass. If animation or occlusion leaves no
    local palette pixels, return ``None`` and fail the frame
    closed; the same-template recovery ladder handles the next fresh frame.
    """
    sprite_w = max(_FAST_BBOX_MIN_PX, int(round(descriptor.avg_width * scale)))
    sprite_h = max(_FAST_BBOX_MIN_PX, int(round(descriptor.avg_height * scale)))
    w = max(_FAST_BBOX_MIN_PX, int(round(sprite_w * _REFINE_WINDOW_SCALE)))
    h = max(_FAST_BBOX_MIN_PX, int(round(sprite_h * _REFINE_WINDOW_SCALE)))
    fh, fw = frame_bgr.shape[:2]
    x0 = max(0, int(round(cx - w / 2)))
    y0 = max(0, int(round(cy - h / 2)))
    x1 = min(fw, x0 + w)
    y1 = min(fh, y0 + h)
    x0 = max(0, x1 - w)
    y0 = max(0, y1 - h)
    if x1 <= x0 or y1 <= y0:
        return None

    region = frame_bgr[y0:y1, x0:x1]
    heat = sprite_palette_heatmap(
        region,
        descriptor.match_palette_bgr,
        float(descriptor.max_sprite_palette_distance),
    )
    mask = heat >= float(detector.config["minSpritePaletteMatch"])
    if not np.any(mask):
        return None

    local_cx = int(cx) - x0
    local_cy = int(cy) - y0
    # Restrict ownership to one descriptor-sized window, then select the
    # connected palette component tied to this hit. Separate nearby identical
    # mobs can never contribute pixels to the published center.
    owner_x0 = max(0, local_cx - sprite_w // 2)
    owner_y0 = max(0, local_cy - sprite_h // 2)
    owner_x1 = min(mask.shape[1], owner_x0 + sprite_w)
    owner_y1 = min(mask.shape[0], owner_y0 + sprite_h)
    owner = mask[owner_y0:owner_y1, owner_x0:owner_x1]
    if not np.any(owner):
        return None

    if not (0 <= local_cx < mask.shape[1] and 0 <= local_cy < mask.shape[0]):
        return None
    if not (owner_x0 <= local_cx < owner_x1 and owner_y0 <= local_cy < owner_y1):
        return None
    # Animated sprites commonly have a small palette gap at their visual
    # center. Bridge only that tiny internal gap inside this Track's ownership
    # window; never search for or merge a neighboring component.
    bridge = cv2.dilate(
        owner.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    n_labels, labels, _stats, _centroids = cv2.connectedComponentsWithStats(
        bridge, connectivity=8,
    )
    if n_labels <= 1:
        return None
    hit_x = local_cx - owner_x0
    hit_y = local_cy - owner_y0
    selected_label = int(labels[hit_y, hit_x])
    if selected_label <= 0:
        # A software cursor can cut a large hole through the exact inferred
        # center while leaving one connected sprite component around it. Accept
        # that component only when its bounding box contains the hit and it is
        # clearly the dominant component in this ownership window. This is not
        # a nearby-component guess: the descriptor-sized ownership window and
        # containment rule still bind the correction to this Track's sprite.
        candidates: list[int] = []
        owner_area = int(owner.sum())
        for label in range(1, n_labels):
            component_x, component_y, component_w, component_h, component_area = (
                int(value) for value in _stats[label]
            )
            contains_hit = (
                component_x <= hit_x < component_x + component_w
                and component_y <= hit_y < component_y + component_h
            )
            dominant = component_area >= max(12, int(owner_area * 0.35))
            if contains_hit and dominant:
                # Measure proximity from original palette pixels, not the
                # dilated labels: dilation must not manufacture evidence that
                # a neighboring component actually approaches the hole.
                component_ys, component_xs = np.where(
                    (labels == label) & owner.astype(bool)
                )
                close_to_hit = bool(
                    component_xs.size
                    and np.min(
                        (component_xs - hit_x) ** 2
                        + (component_ys - hit_y) ** 2
                    ) <= _CENTER_HOLE_MAX_GAP_PX ** 2
                )
                if close_to_hit:
                    candidates.append(label)
        if len(candidates) != 1:
            return None
        selected_label = candidates[0]

    selected = (labels == selected_label) & owner
    ys, xs = np.where(selected)
    if xs.size == 0:
        return None
    local_center_x = owner_x0 + int(round((float(xs.min()) + float(xs.max())) / 2.0))
    local_center_y = owner_y0 + int(round((float(ys.min()) + float(ys.max())) / 2.0))
    return x0 + local_center_x, y0 + local_center_y


def _build_local_follow_heatmap(
    heatmap_detector,
    crop_bgr: np.ndarray,
    descriptor: MobDescriptor, scale: float,
) -> np.ndarray:
    sprite = sprite_palette_heatmap(
        crop_bgr, descriptor.match_palette_bgr,
        descriptor.max_sprite_palette_distance,
    )
    body = palette_heatmap(crop_bgr, descriptor.body_palette)
    accent = palette_heatmap(crop_bgr, descriptor.accent_colors)
    color_signal = np.maximum(
        body * _LOCAL_FOLLOW_BODY_W,
        accent * _LOCAL_FOLLOW_ACCENT_W,
    )

    final = np.zeros(crop_bgr.shape[:2], dtype=np.float32)
    # At most 2 scales: the track's scale first, then the nearest distinct
    # centerScale. Preferring the track scale matters — centerScales is often
    # sorted low→high, so a naive [:2] skipped the known size and weakened peaks.
    for track_scale in _local_follow_scales(
        heatmap_detector._center_scales(crop_bgr.shape[1]), scale,
    ):
        window = (
            max(3, int(round(descriptor.avg_width * track_scale)) | 1),
            max(3, int(round(descriptor.avg_height * track_scale)) | 1),
        )
        sprite_heat = cv2.blur(sprite, window)
        color_heat = cv2.blur(color_signal, window)
        combined = np.maximum(
            sprite_heat * _LOCAL_FOLLOW_SPRITE_W,
            color_heat * _LOCAL_FOLLOW_COLOR_W,
        ).astype(np.float32)
        final = np.maximum(final, combined)
    return final


def _local_follow_scales(center_scales: list[float], track_scale: float) -> list[float]:
    """Track scale first, then one nearest distinct entry from centerScales."""
    scales = [float(track_scale)]
    nearest: float | None = None
    nearest_dist = float("inf")
    for candidate in center_scales:
        dist = abs(float(candidate) - float(track_scale))
        if dist < 1e-9:
            continue
        if dist < nearest_dist:
            nearest_dist = dist
            nearest = float(candidate)
    if nearest is not None:
        scales.append(nearest)
    return scales
