"""Local coordinate follower for already-discovered tracks.

When death detection is enabled the tracker scores at the heatmap peak (not a
stale center), skips opacity probes while the mob is moving, and only marks
death from in-place opacity decay on stationary hits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np

from pybot.recognition.detector.descriptors.descriptor import MobDescriptor
from pybot.recognition.detector.scoring.heatmap_detector import HeatmapDetector, palette_heatmap
from pybot.recognition.detector.tracking.opacity_probe import (
    evaluate_opacity_death,
    measure_opacity_score,
)
from pybot.recognition.rules import death_movement_thresholds, evaluate_track_moving

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
    dead: bool = False
    opacity_baseline: float = 0.0
    opacity_baseline_samples: int = 0
    opacity_decay_streak: int = 0


def track_local(
    detector: MobDetector,
    frame_bgr: np.ndarray,
    mob_name: str,
    track: dict,
    *,
    offset_x: int = 0,
    offset_y: int = 0,
    search_radius_px: int | None = None,
    death_detection_enabled: bool = False,
) -> LocalTrackResult:
    """Follow one known track near its last center."""
    track_id = int(track["trackId"])
    cx = int(track["x"])
    cy = int(track["y"])
    scale_hint = track.get("scale")
    opacity_baseline = float(track.get("opacityBaseline", 0.0))
    opacity_baseline_samples = int(track.get("opacityBaselineSamples", 0))
    opacity_decay_streak = int(track.get("opacityDecayStreak", 0))
    was_moving = bool(track.get("moving", False))
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
    if (
        not death_detection_enabled
        and center_score is not None
        and center_score.accepted
        and center_bbox is not None
    ):
        return _finalize_track_hit(
            detector,
            frame_bgr,
            hsv,
            descriptor,
            track_id=track_id,
            bbox=center_bbox,
            score=center_score,
            offset_x=offset_x,
            offset_y=offset_y,
            probe_opacity_death=False,
            opacity_baseline=opacity_baseline,
            opacity_baseline_samples=opacity_baseline_samples,
            opacity_decay_streak=opacity_decay_streak,
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
        return _miss_result(
            track_id=track_id,
            x=screen_cx,
            y=screen_cy,
            reason=reason,
            opacity_baseline=opacity_baseline,
            opacity_baseline_samples=opacity_baseline_samples,
            opacity_decay_streak=opacity_decay_streak,
        )

    peak_x, peak_y, _heat_score = peak
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
        return _miss_result(
            track_id=track_id,
            x=screen_cx,
            y=screen_cy,
            reason=reason or "below_threshold",
            confidence=round(peak_score.final_score, 4) if peak_score is not None else 0.0,
            opacity_baseline=opacity_baseline,
            opacity_baseline_samples=opacity_baseline_samples,
            opacity_decay_streak=opacity_decay_streak,
        )

    probe_opacity_death = False
    if death_detection_enabled:
        move_px, stop_px = death_movement_thresholds(detector.config)
        peak_dist_sq = (peak_x - cx) ** 2 + (peak_y - cy) ** 2
        moving = evaluate_track_moving(
            was_moving=was_moving,
            displacement_sq=peak_dist_sq,
            move_threshold_px=move_px,
            stop_threshold_px=stop_px,
        )
        if moving:
            opacity_decay_streak = 0
        else:
            probe_opacity_death = True

    return _finalize_track_hit(
        detector,
        frame_bgr,
        hsv,
        descriptor,
        track_id=track_id,
        bbox=peak_bbox,
        score=peak_score,
        offset_x=offset_x,
        offset_y=offset_y,
        probe_opacity_death=probe_opacity_death,
        opacity_baseline=opacity_baseline,
        opacity_baseline_samples=opacity_baseline_samples,
        opacity_decay_streak=opacity_decay_streak,
    )


def _miss_result(
    *,
    track_id: int,
    x: int,
    y: int,
    reason: str,
    confidence: float = 0.0,
    opacity_baseline: float = 0.0,
    opacity_baseline_samples: int = 0,
    opacity_decay_streak: int = 0,
) -> LocalTrackResult:
    return LocalTrackResult(
        track_id=track_id,
        found=False,
        x=x,
        y=y,
        confidence=confidence,
        miss_reason=reason,
        opacity_baseline=opacity_baseline,
        opacity_baseline_samples=opacity_baseline_samples,
        opacity_decay_streak=opacity_decay_streak,
    )


def _finalize_track_hit(
    detector: MobDetector,
    frame_bgr: np.ndarray,
    hsv: np.ndarray,
    descriptor: MobDescriptor,
    *,
    track_id: int,
    bbox: tuple[int, int, int, int],
    score,
    offset_x: int,
    offset_y: int,
    probe_opacity_death: bool,
    opacity_baseline: float,
    opacity_baseline_samples: int,
    opacity_decay_streak: int,
) -> LocalTrackResult:
    bx, by, bw, bh = bbox
    x = bx + bw // 2 + offset_x
    y = by + bh // 2 + offset_y
    confidence = round(score.final_score, 4)

    if probe_opacity_death:
        opacity_score = measure_opacity_score(
            frame_bgr,
            hsv,
            descriptor,
            bbox,
            detector.region_scorer,
        )
        opacity_baseline, opacity_baseline_samples, opacity_decay_streak, dead = (
            evaluate_opacity_death(
                opacity_score=opacity_score,
                baseline=opacity_baseline,
                baseline_samples=opacity_baseline_samples,
                decay_streak=opacity_decay_streak,
                config=detector.config,
            )
        )
        if dead:
            return LocalTrackResult(
                track_id=track_id,
                found=False,
                x=x,
                y=y,
                confidence=confidence,
                miss_reason="opacity_decay",
                dead=True,
                opacity_baseline=opacity_baseline,
                opacity_baseline_samples=opacity_baseline_samples,
                opacity_decay_streak=opacity_decay_streak,
            )

    return LocalTrackResult(
        track_id=track_id,
        found=True,
        x=x,
        y=y,
        confidence=confidence,
        miss_reason="",
        opacity_baseline=opacity_baseline,
        opacity_baseline_samples=opacity_baseline_samples,
        opacity_decay_streak=opacity_decay_streak,
    )


def _resolve_local_track_scale(
    detector: MobDetector,
    frame_width: int,
    scale_hint: float | None,
) -> float:
    if scale_hint is None:
        return detector._direct_track_scale(frame_width, None)
    scales = detector._scales_for_track(frame_width, float(scale_hint))
    return scales[0]


def _find_local_peak(
    detector: MobDetector,
    frame_bgr: np.ndarray,
    hsv: np.ndarray,
    descriptor: MobDescriptor,
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
    descriptor: MobDescriptor,
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
