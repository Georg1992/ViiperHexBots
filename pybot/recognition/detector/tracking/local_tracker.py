"""Local coordinate follower for already-discovered tracks.

Heatmap-based tracking: scores at the predicted center first, falls back
to a heatmap peak search when center misses. Uses the descriptor's sprite
and body palettes with a distance threshold — no exact-match, no sampled
track palette. Tracking is pure follow; discovery handles all liveness
decisions (2-miss removal, stationary timeout, palette decay).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np

from pybot.recognition.detector.descriptors.descriptor import MobDescriptor
from pybot.recognition.detector.scoring.heatmap_detector import palette_heatmap, sprite_palette_heatmap

if TYPE_CHECKING:
    from pybot.recognition.detector.detector import MobDetector

@dataclass(frozen=True)
class LocalTrackResult:
    track_id: int
    found: bool
    x: int
    y: int
    confidence: float
    miss_reason: str


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
    """Follow one known track near its last / predicted center.

    ``suppress_positions``: ROI-relative (x, y) of other tracks whose heat
    signature should be suppressed in the peak search. Prevents track A from
    locking onto mob B when two mobs are close together.
    """
    track_id = int(track["trackId"])
    cx = int(track["x"])
    cy = int(track["y"])
    scale = float(track.get("scale", 1.0))
    moving = bool(track.get("moving", False))
    vel_x = float(track.get("velX", 0.0))
    vel_y = float(track.get("velY", 0.0))
    lost_count = int(track.get("lostCount", 0))

    if search_radius_px is not None:
        radius = int(search_radius_px)
    elif moving or lost_count > 0:
        radius = int(detector.local_track_moving_search_radius_px)
    else:
        radius = int(detector.local_track_search_radius_px)

    # Predicted position: coast velocity when moving
    predicted_x = int(round(cx + vel_x)) if moving else cx
    predicted_y = int(round(cy + vel_y)) if moving else cy
    search_x, search_y = predicted_x, predicted_y

    descriptor = detector.ensure_descriptor(mob_name)
    screen_cx = cx + offset_x
    screen_cy = cy + offset_y

    # Score at predicted center first (fast path)
    accepted, center_bbox, sim = detector.score_at(
        frame_bgr, descriptor, search_x, search_y, scale,
    )
    center_hit = accepted and center_bbox is not None

    if center_hit:
        return _finalize_track_hit(
            track_id=track_id, bbox=center_bbox, similarity=sim,
            offset_x=offset_x, offset_y=offset_y,
        )

    # Center miss → search local heatmap peaks.
    # Use the LAST known position (cx, cy) as the search center, not the
    # predicted position (search_x, search_y). When velocity prediction is
    # wrong (e.g. direction change), the mob is closest to where it was
    # last seen, not where we predicted it to be.
    peak = _find_local_peak(
        detector, frame_bgr, descriptor, cx, cy, scale,
        search_radius_px=radius,
        suppress_positions=suppress_positions,
        lost_count=lost_count,
    )
    if peak is not None:
        _peak_x, _peak_y, _heat_score, peak_sim, peak_bbox = peak
        return _finalize_track_hit(
            track_id=track_id, bbox=peak_bbox, similarity=peak_sim,
            offset_x=offset_x, offset_y=offset_y,
        )

    return _miss_result(
        track_id=track_id, x=screen_cx, y=screen_cy,
        reason="no_peak", confidence=sim,
    )


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
    track_id: int,
    bbox: tuple[int, int, int, int],
    similarity: float,
    offset_x: int, offset_y: int,
) -> LocalTrackResult:
    bx, by, bw, bh = bbox
    x = bx + bw // 2 + offset_x
    y = by + bh // 2 + offset_y

    return LocalTrackResult(
        track_id=track_id, found=True, x=x, y=y,
        confidence=similarity, miss_reason="",
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
    lost_count: int = 0,
) -> tuple[int, int, float] | None:
    frame_h, frame_w = frame_bgr.shape[:2]
    margin_x = int(round(descriptor.avg_width * scale * 0.6))
    margin_y = int(round(descriptor.avg_height * scale * 0.6))
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

    # Suppress heat near other tracks so each track grabs its own mob.
    # Active tracks (lost_count == 0): suppress aggressively — search radius / 2
    # prevents locking onto a neighbor mob.
    # Lost tracks (lost_count > 0): no suppression — locking onto ANY mob is
    # better than remaining lost (discovery will sort out identity later).
    if suppress_positions and lost_count == 0:
        suppress_radius = max(8, search_radius_px // 2)
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
    masked = np.where(mask, local_final, 0.0)
    min_heat = detector.heatmap_detector.min_center_heat * 0.5

    best_peak: tuple[int, int, float, float, tuple[int, int, int, int]] | None = None
    best_living_sim = -1.0
    work = masked.copy()
    suppress_radius = max(8, search_radius_px // 4)
    # Check the single strongest heatmap peak. The center fast path handles
    # the normal case; only the top heatmap candidate is worth the expensive
    # score_at() silhouette gate call. A 2nd iteration would rarely differ.
    for _ in range(1):
        peak_val = float(work.max())
        if peak_val < min_heat:
            break
        peak_y_local, peak_x_local = np.unravel_index(int(work.argmax()), work.shape)
        peak_x = int(peak_x_local + x0)
        peak_y = int(peak_y_local + y0)
        accepted, bbox, sim = detector.score_at(
            frame_bgr, descriptor, peak_x, peak_y, scale,
        )
        if accepted and sim > best_living_sim and bbox is not None:
            best_living_sim = sim
            best_peak = (peak_x, peak_y, peak_val, sim, bbox)
        cv2.circle(work, (peak_x_local, peak_y_local), suppress_radius, 0.0, thickness=-1)

    return best_peak


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
    color_signal = np.maximum(body * 0.55, accent * 0.45)

    final = np.zeros(crop_bgr.shape[:2], dtype=np.float32)
    # Use at most 2 scales — the track's rough scale plus one adjacent.
    # For a 16ms tracking interval, the mob's apparent size doesn't change
    # significantly, so the full multi-scale blend is unnecessary.
    scales = heatmap_detector._center_scales(crop_bgr.shape[1])
    if scale not in scales:
        scales = [scale, *scales]
    for track_scale in scales[:2]:
        window = (
            max(3, int(round(descriptor.avg_width * track_scale)) | 1),
            max(3, int(round(descriptor.avg_height * track_scale)) | 1),
        )
        sprite_heat = cv2.blur(sprite, window)
        color_heat = cv2.blur(color_signal, window)
        combined = np.maximum(sprite_heat * 0.75, color_heat * 0.55).astype(np.float32)
        final = np.maximum(final, combined)
    return final
