"""Local coordinate follower for already-discovered tracks.

Deterministic follow around the predicted position:
1. Build one lightweight local color/sprite heatmap on a small image pyramid.
2. Search the strongest local peak(s) around a one-frame velocity prediction.
3. Verify only the winning peak(s) with the native-resolution silhouette gate.

The expensive gate is deliberately not run at the old center first: that center
is stale for moving mobs and doing so duplicated the largest part of every
tracking tick. Tracking is pure follow for position; discovery owns miss-count
removal.
Opacity is measured on hits for in-place death fade.
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
from pybot.recognition.detector.tracking.opacity_probe import measure_opacity_score

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
# The sprite-recenter window is 1.5x the descriptor size so a hit that sits
# off the sprite edge still contains the full body (a descriptor-sized window
# would clip the palette bbox and bias the center toward the window center).
_REFINE_WINDOW_SCALE = 1.5
_LOCAL_CROSS_TRACK_SUPPRESS_DIV = 2
_LOCAL_FOLLOW_MIN_HEAT_FRAC = 0.5
_LOCAL_FOLLOW_BODY_W = 0.55
_LOCAL_FOLLOW_ACCENT_W = 0.45
_LOCAL_FOLLOW_SPRITE_W = 0.75
_LOCAL_FOLLOW_COLOR_W = 0.55
# Tracking has one deterministic peak decision; failed validation is a miss.
# Large sprites (especially Anubis) make a full-resolution local heatmap
# unnecessarily expensive. A 2x image pyramid keeps the search geometry wide
# while reducing the per-pixel palette/morphology work by roughly 4x. The final
# candidate is still verified at native resolution by score_at().
_LOCAL_HEATMAP_DOWNSCALE = 2
# Temporal-follow correlation runs on a half-resolution grayscale patch. It is
# deliberately independent of the full detector so a confirmed track does not
# pay the discovery/silhouette cost on every frame.
_TEMPLATE_DOWNSCALE = 2
_TEMPLATE_MIN_SCORE = 0.42
_TEMPLATE_MIN_STD = 3.0
# Warm correlation must still contain enough mob-colored/body-colored pixels;
# this rejects a grayscale match on terrain or the player before the cache is
# refreshed from it.
_TEMPLATE_MIN_SPRITE_FRACTION = 0.12
_TEMPLATE_MIN_BODY_FRACTION = 0.035
_TEMPLATE_VERIFY_EVERY = 12
_TEMPLATE_MAX_BYTES = 2_000_000


@dataclass(frozen=True)
class _TrackTemplate:
    image_gray: np.ndarray
    # Native-frame dimensions and the sprite center inside the cached patch.
    width: int
    height: int
    center_x: int
    center_y: int
    scale: float
    verified_hits: int = 0


@dataclass(frozen=True)
class LocalTrackResult:
    track_id: int
    found: bool
    x: int
    y: int
    confidence: float
    miss_reason: str
    opacity_score: float = 0.0


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
    Positive IDs require a transferred cached template and perform the fast
    temporal follow first, with bounded palette recovery after a miss. Zero is
    invalid and returns a miss.

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

    # ``velX``/``velY`` are a smoothed one-frame displacement from the shared
    # track state. Predict one frame ahead so a kiting mob stays near the
    # center of the search crop instead of being found at its outer edge after
    # every update. A long gap means the displacement is stale rather than a
    # reliable one-frame velocity, so stale callers can explicitly disable it
    # with ``prediction_valid=False``.
    prediction_valid = track.get("prediction_valid", True) is not False
    prediction_dx = float(track.get("velX", 0.0)) if prediction_valid else 0.0
    prediction_dy = float(track.get("velY", 0.0)) if prediction_valid else 0.0
    prediction_len = (prediction_dx * prediction_dx + prediction_dy * prediction_dy) ** 0.5
    if prediction_len > float(radius) and prediction_len > 0.0:
        factor = float(radius) / prediction_len
        prediction_dx *= factor
        prediction_dy *= factor
    search_cx = int(round(cx + prediction_dx))
    search_cy = int(round(cy + prediction_dy))

    # Candidate resolution is an explicit one-shot phase. Confirmed tracks use
    # warm-template follow first, with a bounded palette recovery only after a
    # miss so transient animation/occlusion does not break stickiness.
    if track_id < 0:
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
            )
        _peak_x, _peak_y, _heat_score, peak_sim, peak_bbox = peak
        return _finalize_track_hit(
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
    )
    if template_hit is not None:
        return template_hit

    # A temporal match can fail for one animated/occluded frame even though the
    # mob is still nearby. Do not make that one miss permanently destroy the
    # warm template. Pay the more expensive palette follow only on a miss and
    # widen the recovery disk as the store's lost count grows; a later hit
    # re-seeds the same template and makes the normal path fast again.
    # Keep recovery bounded to one extra local disk. Expanding on every miss
    # makes a long-lived miss both expensive and identity-unsafe; discovery is
    # the authority for a genuinely new/relocated object.
    recovery_radius = min(
        radius * 2,
        max(radius, int(getattr(detector, "local_track_max_search_radius_px", radius))),
    )
    peak = _find_local_peak(
        detector,
        frame_bgr,
        descriptor,
        search_cx,
        search_cy,
        scale,
        search_radius_px=recovery_radius,
        suppress_positions=suppress_positions,
    )
    if peak is not None:
        _peak_x, _peak_y, _heat_score, peak_sim, peak_bbox = peak
        template = _template_store(detector).get(track_id)
        # Palette recovery may see a neighboring identical mob. Require the
        # preserved temporal patch to agree before moving the authoritative
        # track; otherwise report a miss and let discovery confirm identity.
        if (
            template is None
            or _template_candidate_agrees(frame_bgr, template, peak_bbox)
        ):
            return _finalize_track_hit(
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
    return _miss_result(
        track_id=track_id,
        x=screen_cx,
        y=screen_cy,
        reason="template_miss",
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
    *,
    track_id: int, x: int, y: int, reason: str,
    confidence: float = 0.0,
) -> LocalTrackResult:
    return LocalTrackResult(
        track_id=track_id, found=False, x=x, y=y,
        confidence=confidence, miss_reason=reason,
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
) -> LocalTrackResult:
    bx, by, bw, bh = bbox
    x = bx + bw // 2 + offset_x
    y = by + bh // 2 + offset_y

    opacity_score = measure_opacity_score(
        frame_bgr,
        descriptor,
        bbox,
        float(descriptor.max_sprite_palette_distance),
        float(detector.config["minSpritePaletteMatch"]),
    )

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
        opacity_score=opacity_score,
    )


def _template_store(detector: MobDetector) -> dict[int, _TrackTemplate]:
    """Get the per-detector temporal cache without global cross-session state."""
    store = getattr(detector, "_local_track_templates", None)
    if store is None:
        store = {}
        setattr(detector, "_local_track_templates", store)
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


def clear_track_templates(detector: MobDetector) -> None:
    """Drop all temporal patches when the detector enters a new screen area."""
    store = getattr(detector, "_local_track_templates", None)
    if store is not None:
        store.clear()


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
    reduced = cv2.resize(
        gray,
        (max(1, gray.shape[1] // _TEMPLATE_DOWNSCALE),
         max(1, gray.shape[0] // _TEMPLATE_DOWNSCALE)),
        interpolation=cv2.INTER_AREA,
    )
    store = _template_store(detector)
    previous = store.get(track_id)
    verified_hits = (
        int(getattr(previous, "verified_hits", 0)) if previous is not None else 0
    )
    store[track_id] = _TrackTemplate(
        image_gray=reduced,
        width=x1 - x0,
        height=y1 - y0,
        center_x=(x0 + x1) // 2,
        center_y=(y0 + y1) // 2,
        scale=scale,
        verified_hits=verified_hits,
    )
    # Track IDs are monotonic in production, but keep a test/restart session
    # from retaining unbounded image memory if a caller reuses one detector.
    total_bytes = sum(int(item.image_gray.nbytes) for item in store.values())
    while total_bytes > _TEMPLATE_MAX_BYTES and store:
        oldest_id = next(iter(store))
        total_bytes -= int(store.pop(oldest_id).image_gray.nbytes)


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
) -> LocalTrackResult | None:
    """Follow a confirmed patch; return None when bounded recovery is needed."""
    template = _template_store(detector).get(track_id)
    if template is None:
        return None
    template_gray = template.image_gray
    if template_gray.size == 0:
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
    th, tw = template_gray.shape[:2]
    if tw > work.shape[1] or th > work.shape[0]:
        return None
    if float(work.std()) < _TEMPLATE_MIN_STD:
        return None
    scores = cv2.matchTemplate(work, template_gray, cv2.TM_CCOEFF_NORMED)
    _min_val, max_val, _min_loc, max_loc = cv2.minMaxLoc(scores)
    if not np.isfinite(max_val) or max_val < _TEMPLATE_MIN_SCORE:
        return None

    hit_x = int(round(x0 + (max_loc[0] + tw / 2.0) * _TEMPLATE_DOWNSCALE))
    hit_y = int(round(y0 + (max_loc[1] + th / 2.0) * _TEMPLATE_DOWNSCALE))
    if suppress_positions:
        suppress_radius = max(_LOCAL_SUPPRESS_RADIUS_FLOOR_PX, search_radius_px // 3)
        if any((hit_x - sx) ** 2 + (hit_y - sy) ** 2 <= suppress_radius ** 2
               for sx, sy in suppress_positions):
            return None

    # Keep the cache aligned to the newly observed location. The patch size is
    # fixed, so animation/movement updates do not introduce geometric drift.
    native_w = max(1, int(round(tw * _TEMPLATE_DOWNSCALE)))
    native_h = max(1, int(round(th * _TEMPLATE_DOWNSCALE)))
    bbox = (
        hit_x - native_w // 2,
        hit_y - native_h // 2,
        native_w,
        native_h,
    )
    if not _template_identity_ok(detector, frame_bgr, descriptor, bbox):
        # Keep the last verified patch for a transient occlusion/animation.
        # The caller performs a one-shot palette recovery on this miss; a
        # single bad frame must not turn a sticky track into a permanently
        # template-less track.
        return None

    # Re-center the follow point on the mob's sprite (palette-CC bbox center)
    # instead of the template patch center, then re-anchor the cached patch so
    # the next match is centered too. The patch keeps its size (no drift).
    hit_x, hit_y = _refine_hit_to_sprite_center(
        detector, frame_bgr, descriptor, hit_x, hit_y, scale,
    )
    bbox = (
        hit_x - native_w // 2,
        hit_y - native_h // 2,
        native_w,
        native_h,
    )

    previous_hits = int(getattr(template, "verified_hits", 0)) + 1
    # Static GRF descriptors skip the periodic native verify entirely — the
    # every-hit ``_template_identity_ok`` color/body check is their identity
    # guarantee, and skipping the gate removes the flicker-miss class.
    if (
        previous_hits % _TEMPLATE_VERIFY_EVERY == 0
        and not _fast_track_accept(detector, descriptor)
    ):
        accepted, _verified_bbox, _similarity = detector.score_at(
            frame_bgr, descriptor, hit_x, hit_y, scale,
        )
        if not accepted:
            # Periodic native verification is a safety check, not proof that
            # the mob died. Preserve the warm patch and let the miss recovery
            # path reacquire on the next frame.
            return None

    _remember_track_template(
        detector,
        track_id=track_id,
        frame_bgr=frame_bgr,
        bbox=bbox,
        scale=scale,
    )
    current = _template_store(detector).get(track_id)
    if current is not None:
        _template_store(detector)[track_id] = _TrackTemplate(
            image_gray=current.image_gray,
            width=current.width,
            height=current.height,
            center_x=current.center_x,
            center_y=current.center_y,
            scale=current.scale,
            verified_hits=previous_hits,
        )
    opacity_score = measure_opacity_score(
        frame_bgr,
        descriptor,
        bbox,
        float(descriptor.max_sprite_palette_distance),
        float(detector.config["minSpritePaletteMatch"]),
    )
    return LocalTrackResult(
        track_id=track_id,
        found=True,
        x=hit_x + offset_x,
        y=hit_y + offset_y,
        confidence=float(max_val),
        miss_reason="",
        opacity_score=opacity_score,
    )


def _template_candidate_agrees(
    frame_bgr: np.ndarray,
    template: _TrackTemplate,
    bbox: tuple[int, int, int, int],
) -> bool:
    """Check a palette-recovery candidate against the preserved warm patch."""
    x, y, width, height = (int(value) for value in bbox)
    frame_h, frame_w = frame_bgr.shape[:2]
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(frame_w, x + max(0, width))
    y1 = min(frame_h, y + max(0, height))
    if x1 <= x0 or y1 <= y0:
        return False
    gray = cv2.cvtColor(frame_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    target_h, target_w = template.image_gray.shape[:2]
    if gray.size == 0 or target_h <= 0 or target_w <= 0:
        return False
    reduced = cv2.resize(
        gray, (target_w, target_h), interpolation=cv2.INTER_AREA,
    )
    if float(reduced.std()) < _TEMPLATE_MIN_STD:
        return False
    score = float(
        cv2.matchTemplate(reduced, template.image_gray, cv2.TM_CCOEFF_NORMED)[0, 0]
    )
    return bool(np.isfinite(score) and score >= _TEMPLATE_MIN_SCORE)


def _template_identity_ok(
    detector: MobDetector,
    frame_bgr: np.ndarray,
    descriptor: MobDescriptor,
    bbox: tuple[int, int, int, int],
) -> bool:
    """Cheap color/body identity check for warm temporal hits."""
    x, y, width, height = bbox
    frame_h, frame_w = frame_bgr.shape[:2]
    x0 = max(0, int(x))
    y0 = max(0, int(y))
    x1 = min(frame_w, int(x + width))
    y1 = min(frame_h, int(y + height))
    if x1 <= x0 or y1 <= y0:
        return False
    region = frame_bgr[y0:y1, x0:x1]
    sprite = sprite_palette_heatmap(
        region,
        descriptor.match_palette_bgr,
        float(descriptor.max_sprite_palette_distance),
    )
    body = palette_heatmap(region, descriptor.body_palette)
    sprite_fraction = float(
        (sprite >= float(detector.config["minSpritePaletteMatch"])).mean()
    )
    body_fraction = float((body >= 0.5).mean())
    return (
        sprite_fraction >= _TEMPLATE_MIN_SPRITE_FRACTION
        and body_fraction >= _TEMPLATE_MIN_BODY_FRACTION
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
            # Static modified sprites (distinctive red, one frame): the local
            # follow heatmap peak is already palette-driven, so accept it via a
            # cheap sprite/body color-fraction check instead of the expensive
            # native-resolution silhouette gate. This is the dominant per-tick
            # cost for large sprites (Anubis) and a common miss source when the
            # gate flickers on a deformed extract. The peak must clear a strong
            # heat multiple (not just the floor) since no silhouette verify runs.
            bbox = _descriptor_sized_bbox(descriptor, peak_x, peak_y, scale)
            if (
                bbox is not None
                and peak_val >= _LOCAL_FAST_MIN_HEAT_MULT * min_heat
                and _template_identity_ok(detector, frame_bgr, descriptor, bbox)
            ):
                # Re-center on the sprite body: the heat peak can sit on the
                # densest color region instead of the sprite center, which made
                # the bot aim off the mob. Heat (palette match strength) doubles
                # as confidence — the fast path has no silhouette similarity.
                peak_x, peak_y = _refine_hit_to_sprite_center(
                    detector, frame_bgr, descriptor, peak_x, peak_y, scale,
                )
                bbox = _descriptor_sized_bbox(descriptor, peak_x, peak_y, scale)
                return peak_x, peak_y, peak_val, float(peak_val), bbox
    else:
        accepted, bbox, sim = detector.score_at(
            frame_bgr, descriptor, peak_x, peak_y, scale,
        )
        if accepted and bbox is not None:
            return peak_x, peak_y, peak_val, sim, bbox
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
) -> tuple[int, int]:
    """Re-center a hit on the mob's sprite (palette-CC bbox center).

    Heat peaks and template-patch centers can sit off the sprite body (a dense
    color region or asymmetric pose). The palette-CC bounding box of the mob
    inside a 1.5x descriptor-sized window around the hit is the same sprite
    anchor discovery and ``score_at`` use, so every consumer sees one
    consistent on-sprite point and the aim click lands on the mob. The
    connected component overlapping the window center (the hit) is isolated so
    an adjacent mob's pixels cannot drag the anchor between two sprites. Falls
    back to the input when no palette match is visible.
    """
    w = max(
        _FAST_BBOX_MIN_PX,
        int(round(descriptor.avg_width * scale * _REFINE_WINDOW_SCALE)),
    )
    h = max(
        _FAST_BBOX_MIN_PX,
        int(round(descriptor.avg_height * scale * _REFINE_WINDOW_SCALE)),
    )
    fh, fw = frame_bgr.shape[:2]
    x0 = max(0, int(round(cx - w / 2)))
    y0 = max(0, int(round(cy - h / 2)))
    x1 = min(fw, x0 + w)
    y1 = min(fh, y0 + h)
    x0 = max(0, x1 - w)
    y0 = max(0, y1 - h)
    if x1 <= x0 or y1 <= y0:
        return int(cx), int(cy)
    region = frame_bgr[y0:y1, x0:x1]
    heat = sprite_palette_heatmap(
        region,
        descriptor.match_palette_bgr,
        float(descriptor.max_sprite_palette_distance),
    )
    mask = heat >= float(detector.config["minSpritePaletteMatch"])
    if not np.any(mask):
        return int(cx), int(cy)

    local_cx = int(cx) - x0
    local_cy = int(cy) - y0
    nlab, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8,
    )
    chosen = 0
    if (
        nlab > 1
        and 0 <= local_cx < mask.shape[1]
        and 0 <= local_cy < mask.shape[0]
    ):
        chosen = int(labels[local_cy, local_cx])
    if chosen <= 0 and nlab > 1:
        # Hit fell between components (e.g. on a gap) — use the largest one.
        areas = {
            label: int(stats[label, cv2.CC_STAT_AREA])
            for label in range(1, nlab)
        }
        chosen = max(areas, key=areas.get)  # type: ignore[arg-type]
    if chosen <= 0:
        return int(cx), int(cy)

    ys, xs = np.where(labels == chosen)
    center_x = x0 + int(round((float(xs.min()) + float(xs.max())) / 2.0))
    center_y = y0 + int(round((float(ys.min()) + float(ys.max())) / 2.0))
    return center_x, center_y


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
