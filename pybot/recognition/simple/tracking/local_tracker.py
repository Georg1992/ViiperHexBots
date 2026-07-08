"""Local coordinate follower for already-discovered tracks (not discovery, not death)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np

from pybot.recognition.simple.descriptors.descriptor import SimpleMobDescriptor
from pybot.recognition.simple.scoring.heatmap_detector import HeatmapDetector, palette_heatmap

if TYPE_CHECKING:
    from pybot.recognition.simple.detector import SimpleMobDetector


@dataclass(frozen=True)
class LocalTrackResult:
    track_id: int
    found: bool
    x: int
    y: int
    confidence: float
    miss_reason: str


def track_local(
    detector: SimpleMobDetector,
    frame_bgr: np.ndarray,
    mob_name: str,
    track: dict,
    *,
    offset_x: int = 0,
    offset_y: int = 0,
    search_radius_px: int | None = None,
) -> LocalTrackResult:
    """Follow one known track near its last center. Living-only; no dead/gone output."""
    track_id = int(track["trackId"])
    cx = int(track["x"])
    cy = int(track["y"])
    scale_hint = track.get("scale")
    radius = (
        int(search_radius_px)
        if search_radius_px is not None
        else int(detector.local_track_search_radius_px)
    )

    descriptor = detector.ensure_descriptor(mob_name)
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    scale = _resolve_local_track_scale(
        detector,
        frame_bgr.shape[1],
        float(scale_hint) if scale_hint is not None else None,
    )

    screen_cx = cx + offset_x
    screen_cy = cy + offset_y

    center_score, center_bbox = detector._score_living_only_at(
        frame_bgr,
        hsv,
        descriptor,
        cx,
        cy,
        scale,
    )
    if center_score is not None and center_score.accepted and center_bbox is not None:
        bx, by, bw, bh = center_bbox
        return LocalTrackResult(
            track_id=track_id,
            found=True,
            x=bx + bw // 2 + offset_x,
            y=by + bh // 2 + offset_y,
            confidence=round(center_score.final_score, 4),
            miss_reason="",
        )

    peak = _find_local_peak(
        detector,
        frame_bgr,
        hsv,
        descriptor,
        cx,
        cy,
        scale,
        search_radius_px=radius,
    )
    if peak is None:
        reason = "no_peak"
        if center_score is not None and not center_score.accepted:
            reason = center_score.rejection_reason or "below_threshold"
        return LocalTrackResult(
            track_id=track_id,
            found=False,
            x=screen_cx,
            y=screen_cy,
            confidence=0.0,
            miss_reason=reason,
        )

    peak_x, peak_y, heat_score = peak
    peak_score, peak_bbox = detector._score_living_only_at(
        frame_bgr,
        hsv,
        descriptor,
        peak_x,
        peak_y,
        scale,
    )
    if peak_score is None or not peak_score.accepted or peak_bbox is None:
        reason = peak_score.rejection_reason if peak_score is not None else "no_bbox"
        return LocalTrackResult(
            track_id=track_id,
            found=False,
            x=screen_cx,
            y=screen_cy,
            confidence=round(peak_score.final_score, 4) if peak_score is not None else 0.0,
            miss_reason=reason or "below_threshold",
        )

    bx, by, bw, bh = peak_bbox
    return LocalTrackResult(
        track_id=track_id,
        found=True,
        x=bx + bw // 2 + offset_x,
        y=by + bh // 2 + offset_y,
        confidence=round(peak_score.final_score, 4),
        miss_reason="",
    )


def _resolve_local_track_scale(
    detector: SimpleMobDetector,
    frame_width: int,
    scale_hint: float | None,
) -> float:
    if scale_hint is None:
        return detector._direct_track_scale(frame_width, None)
    scales = detector._scales_for_track(frame_width, float(scale_hint))
    return scales[0]


def _find_local_peak(
    detector: SimpleMobDetector,
    frame_bgr: np.ndarray,
    hsv: np.ndarray,
    descriptor: SimpleMobDescriptor,
    cx: int,
    cy: int,
    scale: float,
    *,
    search_radius_px: int,
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
    crop_hsv = hsv[y0:y1, x0:x1]
    local_final = _build_local_follow_heatmap(detector.heatmap_detector, crop_bgr, crop_hsv, descriptor, scale)
    if local_final.size == 0:
        return None

    anchor_x = cx - x0
    anchor_y = cy - y0
    yy, xx = np.ogrid[: local_final.shape[0], : local_final.shape[1]]
    dist_sq = (xx - anchor_x) ** 2 + (yy - anchor_y) ** 2
    mask = dist_sq <= (search_radius_px * search_radius_px)
    masked = np.where(mask, local_final, 0.0)
    peak_val = float(masked.max())
    if peak_val < detector.heatmap_detector.min_center_heat:
        return None

    peak_y_local, peak_x_local = np.unravel_index(int(masked.argmax()), masked.shape)
    return int(peak_x_local + x0), int(peak_y_local + y0), peak_val


def _build_local_follow_heatmap(
    heatmap_detector: HeatmapDetector,
    crop_bgr: np.ndarray,
    crop_hsv: np.ndarray,
    descriptor: SimpleMobDescriptor,
    scale: float,
) -> np.ndarray:
    body = palette_heatmap(crop_hsv, descriptor.body_palette)
    accent = palette_heatmap(crop_hsv, descriptor.accent_colors)
    rare = palette_heatmap(crop_hsv, descriptor.rare_colors)
    pattern = HeatmapDetector._local_pattern(crop_bgr, accent, body)
    window = (
        max(3, int(round(descriptor.avg_width * scale)) | 1),
        max(3, int(round(descriptor.avg_height * scale)) | 1),
    )
    weights = heatmap_detector.center_weights
    center_body = cv2.blur(body, window)
    center_accent = cv2.blur(accent, window)
    center_rare = cv2.blur(rare, window)
    center_pattern = cv2.blur(pattern, window)
    return (
        center_body * weights["body"]
        + center_accent * weights["accent"]
        + center_rare * weights["rare"]
        + center_pattern * weights["pattern"]
    ).astype(np.float32)
