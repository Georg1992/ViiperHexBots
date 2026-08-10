"""Fast centered local follower for already-discovered tracks.

Each fresh tracking frame uses one simple path:
1. Validate the last published center with the detector's silhouette gate.
2. If that misses, search one bounded local heatmap around that center.
3. Publish the accepted bbox center from the current frame.

There is no velocity coasting, optical-flow state, template drift, or per-track
worker. A miss keeps the last coordinate while the next fresh frame retries the
same bounded local recovery path.
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
_LOCAL_CROSS_TRACK_SUPPRESS_DIV = 2
_LOCAL_PEAK_SUPPRESS_DIV = 4
_LOCAL_FOLLOW_MIN_HEAT_FRAC = 0.5
_LOCAL_FOLLOW_BODY_W = 0.55
_LOCAL_FOLLOW_ACCENT_W = 0.45
_LOCAL_FOLLOW_SPRITE_W = 0.75
_LOCAL_FOLLOW_COLOR_W = 0.55
_LOCAL_PEAK_ATTEMPTS = 1


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


# These helpers intentionally keep no visual state. The runtime uses them to
# preserve its explicit area-reset/provisional-ID lifecycle boundary; the
# centered follower has no cached state to transfer or invalidate after teleport.
# The successful return from transfer_track_state means "stateless handoff
# accepted", not "a visual cache was moved".
def transfer_track_state(detector: MobDetector, source_track_id: int, target_track_id: int) -> bool:
    del detector, source_track_id, target_track_id
    return True


def discard_track_state(detector: MobDetector, track_id: int) -> None:
    del detector, track_id


def prune_track_states(detector: MobDetector, active_track_ids: set[int]) -> None:
    del detector, active_track_ids


def clear_track_states(detector: MobDetector) -> None:
    del detector


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
    """Follow one known Track near its last published center."""
    track_id = int(track["trackId"])
    cx = int(track["x"])
    cy = int(track["y"])
    if track_id == 0:
        return _miss_result(
            track_id=track_id,
            x=cx + offset_x,
            y=cy + offset_y,
            reason="invalid_track_id",
        )
    scale = float(track.get("scale", 0.0))
    if scale <= 0.0:
        return _miss_result(
            track_id=track_id,
            x=cx + offset_x,
            y=cy + offset_y,
            reason="invalid_scale",
        )

    radius = (
        int(search_radius_px)
        if search_radius_px is not None
        else int(detector.local_track_moving_search_radius_px)
    )
    descriptor = detector.ensure_descriptor(mob_name)
    screen_cx = cx + offset_x
    screen_cy = cy + offset_y

    static_fast = bool(
        getattr(detector, "use_sprite_grf", False)
        and detector.descriptor_is_static(descriptor)
        and getattr(detector, "grf_local_track_skip_native_gate", True)
    )
    if not static_fast:
        accepted, center_bbox, similarity = detector.score_at(
            frame_bgr, descriptor, cx, cy, scale,
        )
        if accepted and center_bbox is not None:
            return _finalize_track_hit(
                detector=detector,
                frame_bgr=frame_bgr,
                descriptor=descriptor,
                track_id=track_id,
            bbox=center_bbox,
            similarity=similarity,
            scale=scale,
            offset_x=offset_x,
                offset_y=offset_y,
            )

    peak = _find_local_peak(
        detector,
        frame_bgr,
        descriptor,
        cx,
        cy,
        scale,
        search_radius_px=radius,
        suppress_positions=suppress_positions,
    )
    if peak is None:
        # Recovery is still local and bounded. The healthy path never pays this
        # second search; it is only for a temporary center/peak miss.
        expanded_radius = min(
            int(getattr(detector, "local_track_max_search_radius_px", radius)),
            max(radius + 1, radius * 2),
        )
        if expanded_radius > radius:
            peak = _find_local_peak(
                detector,
                frame_bgr,
                descriptor,
                cx,
                cy,
                scale,
                search_radius_px=expanded_radius,
                suppress_positions=suppress_positions,
            )
    if peak is not None:
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

    return _miss_result(
        track_id=track_id,
        x=screen_cx,
        y=screen_cy,
        reason="no_peak",
        confidence=0.0,
    )


def _miss_result(
    *,
    track_id: int,
    x: int,
    y: int,
    reason: str,
    confidence: float = 0.0,
) -> LocalTrackResult:
    return LocalTrackResult(
        track_id=track_id,
        found=False,
        x=x,
        y=y,
        confidence=confidence,
        miss_reason=reason,
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
    offset_x: int,
    offset_y: int,
) -> LocalTrackResult:
    bx, by, bw, bh = bbox
    center_x, center_y = _refine_hit_to_sprite_center(
        detector, frame_bgr, descriptor,
        bx + bw // 2, by + bh // 2, scale,
    )
    x = center_x + offset_x
    y = center_y + offset_y
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
        x=x,
        y=y,
        confidence=similarity,
        miss_reason="",
        opacity_score=opacity_score,
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
    x0 = max(0, cx - pad)
    y0 = max(0, cy - pad)
    x1 = min(frame_w, cx + pad + 1)
    y1 = min(frame_h, cy + pad + 1)
    if x1 <= x0 or y1 <= y0:
        return None

    crop_bgr = frame_bgr[y0:y1, x0:x1]
    local_final = _build_local_follow_heatmap(
        detector.heatmap_detector, crop_bgr, descriptor, scale,
    )
    if local_final.size == 0:
        return None

    if suppress_positions:
        suppress_radius = max(
            _LOCAL_SUPPRESS_RADIUS_FLOOR_PX,
            search_radius_px // _LOCAL_CROSS_TRACK_SUPPRESS_DIV,
        )
        for sx, sy in suppress_positions:
            lsx = sx - x0
            lsy = sy - y0
            if 0 <= lsx < local_final.shape[1] and 0 <= lsy < local_final.shape[0]:
                cv2.circle(
                    local_final,
                    (int(lsx), int(lsy)),
                    suppress_radius,
                    0.0,
                    thickness=-1,
                )

    anchor_x = cx - x0
    anchor_y = cy - y0
    yy, xx = np.ogrid[: local_final.shape[0], : local_final.shape[1]]
    dist_sq = (xx - anchor_x) ** 2 + (yy - anchor_y) ** 2
    mask = dist_sq <= (search_radius_px * search_radius_px)
    work = np.where(mask, local_final, 0.0).copy()
    min_heat = detector.heatmap_detector.min_center_heat * _LOCAL_FOLLOW_MIN_HEAT_FRAC
    suppress_radius = max(
        _LOCAL_SUPPRESS_RADIUS_FLOOR_PX,
        search_radius_px // _LOCAL_PEAK_SUPPRESS_DIV,
    )

    for _ in range(_LOCAL_PEAK_ATTEMPTS):
        peak_val = float(work.max())
        if peak_val < min_heat:
            break
        peak_y_local, peak_x_local = np.unravel_index(int(work.argmax()), work.shape)
        peak_x = int(peak_x_local + x0)
        peak_y = int(peak_y_local + y0)
        static_fast = bool(
            getattr(detector, "use_sprite_grf", False)
            and detector.descriptor_is_static(descriptor)
            and getattr(detector, "grf_local_track_skip_native_gate", True)
        )
        if static_fast:
            bbox = _descriptor_sized_bbox(descriptor, peak_x, peak_y, scale)
            if _local_identity_ok(detector, frame_bgr, descriptor, peak_x, peak_y, scale):
                return peak_x, peak_y, peak_val, peak_val, bbox
        else:
            accepted, bbox, sim = detector.score_at(
                frame_bgr, descriptor, peak_x, peak_y, scale,
            )
            if accepted and bbox is not None:
                return peak_x, peak_y, peak_val, sim, bbox
        cv2.circle(
            work,
            (peak_x_local, peak_y_local),
            suppress_radius,
            0.0,
            thickness=-1,
        )

    return None


def _descriptor_sized_bbox(
    descriptor: MobDescriptor,
    cx: int,
    cy: int,
    scale: float,
) -> tuple[int, int, int, int]:
    width = max(8, int(round(descriptor.avg_width * scale)))
    height = max(8, int(round(descriptor.avg_height * scale)))
    return (
        int(round(cx - width / 2)),
        int(round(cy - height / 2)),
        width,
        height,
    )


def _local_identity_ok(
    detector: MobDetector,
    frame_bgr: np.ndarray,
    descriptor: MobDescriptor,
    cx: int,
    cy: int,
    scale: float,
) -> bool:
    x, y, width, height = _descriptor_sized_bbox(descriptor, cx, cy, scale)
    frame_h, frame_w = frame_bgr.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(frame_w, x + width), min(frame_h, y + height)
    if x1 <= x0 or y1 <= y0:
        return False
    region = frame_bgr[y0:y1, x0:x1]
    sprite = sprite_palette_heatmap(
        region,
        descriptor.match_palette_bgr,
        float(descriptor.max_sprite_palette_distance),
    )
    threshold = float(detector.config["minSpritePaletteMatch"])
    return float((sprite >= threshold).mean()) >= 0.08


def _refine_hit_to_sprite_center(
    detector: MobDetector,
    frame_bgr: np.ndarray,
    descriptor: MobDescriptor,
    cx: int,
    cy: int,
    scale: float,
) -> tuple[int, int]:
    """Return the center of the local sprite-palette component."""
    width = max(8, int(round(descriptor.avg_width * scale * 1.5)))
    height = max(8, int(round(descriptor.avg_height * scale * 1.5)))
    frame_h, frame_w = frame_bgr.shape[:2]
    x0 = max(0, int(cx - width / 2))
    y0 = max(0, int(cy - height / 2))
    x1 = min(frame_w, x0 + width)
    y1 = min(frame_h, y0 + height)
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
    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), 8,
    )
    local_x, local_y = int(cx) - x0, int(cy) - y0
    chosen = (
        int(labels[local_y, local_x])
        if labels_count > 1
        and 0 <= local_x < labels.shape[1]
        and 0 <= local_y < labels.shape[0]
        else 0
    )
    if chosen <= 0 and labels_count > 1:
        chosen = max(
            range(1, labels_count),
            key=lambda label: int(stats[label, cv2.CC_STAT_AREA]),
        )
    if chosen <= 0:
        return int(cx), int(cy)
    ys, xs = np.where(labels == chosen)
    return (
        x0 + int(round((xs.min() + xs.max()) / 2)),
        y0 + int(round((ys.min() + ys.max()) / 2)),
    )


def _build_local_follow_heatmap(
    heatmap_detector,
    crop_bgr: np.ndarray,
    descriptor: MobDescriptor,
    scale: float,
) -> np.ndarray:
    sprite = sprite_palette_heatmap(
        crop_bgr,
        descriptor.match_palette_bgr,
        descriptor.max_sprite_palette_distance,
    )
    body = palette_heatmap(crop_bgr, descriptor.body_palette)
    accent = palette_heatmap(crop_bgr, descriptor.accent_colors)
    color_signal = np.maximum(
        body * _LOCAL_FOLLOW_BODY_W,
        accent * _LOCAL_FOLLOW_ACCENT_W,
    )

    final = np.zeros(crop_bgr.shape[:2], dtype=np.float32)
    for track_scale in _local_follow_scales(
        heatmap_detector._center_scales(crop_bgr.shape[1]), scale,
    ):
        window = (
            max(3, int(round(descriptor.avg_width * track_scale)) | 1),
            max(3, int(round(descriptor.avg_height * track_scale)) | 1),
        )
        combined = np.maximum(
            cv2.blur(sprite, window) * _LOCAL_FOLLOW_SPRITE_W,
            cv2.blur(color_signal, window) * _LOCAL_FOLLOW_COLOR_W,
        ).astype(np.float32)
        final = np.maximum(final, combined)
    return final


def _local_follow_scales(
    center_scales: list[float], track_scale: float,
) -> list[float]:
    """Return the known Track scale first, then one nearest distinct scale."""
    scales = [float(track_scale)]
    nearest: float | None = None
    nearest_dist = float("inf")
    for candidate in center_scales:
        distance = abs(float(candidate) - float(track_scale))
        if distance < 1e-9:
            continue
        if distance < nearest_dist:
            nearest_dist = distance
            nearest = float(candidate)
    if nearest is not None:
        scales.append(nearest)
    return scales
