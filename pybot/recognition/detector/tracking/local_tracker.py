"""Local coordinate follower for already-discovered tracks.

Scores at the last center first, searches nearby heatmap peaks when that
misses, and measures opacity on every successful hit when death detection is
enabled. Opacity decay confirms death; misses only advance the lost streak.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np

from pybot.recognition.detector.descriptors.descriptor import MobDescriptor
from pybot.recognition.detector.scoring.heatmap_detector import palette_heatmap, sprite_palette_heatmap
from pybot.recognition.detector.tracking.opacity_probe import (
    calibrate_opacity_baseline,
    evaluate_opacity_death,
    is_opacity_calibrated,
    measure_opacity_score,
)

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
    created_tick = int(track.get("createdTick", 0))
    now_tick = int(track.get("nowTick", 0))
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
    center_hit = (
        center_score is not None
        and center_score.accepted
        and center_bbox is not None
    )

    peak_x = cx
    peak_y = cy
    peak_score = None
    peak_bbox = None
    if not center_hit:
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
                confidence=round(center_score.final_score, 4) if center_score is not None else 0.0,
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

    hit_bbox = center_bbox if center_hit else peak_bbox
    hit_score = center_score if center_hit else peak_score
    assert hit_bbox is not None
    assert hit_score is not None

    return _finalize_track_hit(
        detector,
        frame_bgr,
        hsv,
        descriptor,
        track_id=track_id,
        bbox=hit_bbox,
        score=hit_score,
        offset_x=offset_x,
        offset_y=offset_y,
        death_detection_enabled=death_detection_enabled,
        created_tick=created_tick,
        now_tick=now_tick,
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
    dead: bool = False,
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
        dead=dead,
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
    death_detection_enabled: bool,
    created_tick: int,
    now_tick: int,
    opacity_baseline: float,
    opacity_baseline_samples: int,
    opacity_decay_streak: int,
) -> LocalTrackResult:
    bx, by, bw, bh = bbox
    x = bx + bw // 2 + offset_x
    y = by + bh // 2 + offset_y
    confidence = round(score.final_score, 4)

    if death_detection_enabled and _track_old_enough(
        detector.config,
        created_tick=created_tick,
        now_tick=now_tick,
    ):
        opacity_score = measure_opacity_score(
            frame_bgr,
            hsv,
            descriptor,
            bbox,
            detector.region_scorer,
        )
        if not is_opacity_calibrated(
            baseline=opacity_baseline,
            baseline_samples=opacity_baseline_samples,
            config=detector.config,
        ):
            opacity_baseline, opacity_baseline_samples = calibrate_opacity_baseline(
                opacity_score=opacity_score,
                baseline=opacity_baseline,
                baseline_samples=opacity_baseline_samples,
                config=detector.config,
            )
            opacity_decay_streak = 0
        else:
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


def _track_old_enough(
    config: dict,
    *,
    created_tick: int,
    now_tick: int,
) -> bool:
    """Death confirmation only after the track has lived long enough."""
    min_age_ms = int(config["deathOpacityMinTrackAgeMs"])
    if (
        now_tick > 0
        and created_tick > 0
        and (now_tick - created_tick) < min_age_ms
    ):
        return False
    return True


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
    local_final = _build_local_follow_heatmap(
        detector.heatmap_detector,
        crop_bgr,
        crop_hsv,
        descriptor,
        scale,
    )
    if local_final.size == 0:
        return None

    anchor_x = cx - x0
    anchor_y = cy - y0
    yy, xx = np.ogrid[: local_final.shape[0], : local_final.shape[1]]
    dist_sq = (xx - anchor_x) ** 2 + (yy - anchor_y) ** 2
    mask = dist_sq <= (search_radius_px * search_radius_px)
    masked = np.where(mask, local_final, 0.0)
    min_heat = detector.heatmap_detector.min_center_heat * 0.5

    best_peak: tuple[int, int, float] | None = None
    best_living_score = -1.0
    work = masked.copy()
    suppress_radius = max(8, search_radius_px // 4)
    for _ in range(3):
        peak_val = float(work.max())
        if peak_val < min_heat:
            break
        peak_y_local, peak_x_local = np.unravel_index(int(work.argmax()), work.shape)
        peak_x = int(peak_x_local + x0)
        peak_y = int(peak_y_local + y0)
        living_score, _bbox = detector._score_living_only_at(
            frame_bgr,
            hsv,
            descriptor,
            peak_x,
            peak_y,
            scale,
        )
        living_val = living_score.final_score if living_score is not None else 0.0
        if living_val > best_living_score:
            best_living_score = living_val
            best_peak = (peak_x, peak_y, peak_val)
        cv2.circle(work, (peak_x_local, peak_y_local), suppress_radius, 0.0, thickness=-1)

    if best_peak is None:
        return None
    peak_score, peak_bbox = detector._score_living_only_at(
        frame_bgr,
        hsv,
        descriptor,
        best_peak[0],
        best_peak[1],
        scale,
    )
    if peak_score is None or not peak_score.accepted or peak_bbox is None:
        return None
    return best_peak


def _build_local_follow_heatmap(
    heatmap_detector,
    crop_bgr: np.ndarray,
    crop_hsv: np.ndarray,
    descriptor: MobDescriptor,
    scale: float,
) -> np.ndarray:
    sprite = sprite_palette_heatmap(
        crop_bgr,
        descriptor.match_palette_bgr,
        heatmap_detector.max_sprite_palette_distance,
    )
    body = palette_heatmap(crop_hsv, descriptor.body_palette)
    accent = palette_heatmap(crop_hsv, descriptor.accent_colors)
    color_signal = np.maximum(body * 0.55, accent * 0.45)

    body_sprite = accent_sprite = None
    structural_pairs = descriptor.structural_pixel_pairs()
    if structural_pairs:
        body_sprite = np.zeros(crop_bgr.shape[:2], dtype=np.float32)
        accent_sprite = np.zeros(crop_bgr.shape[:2], dtype=np.float32)
        for dominant, accent in structural_pairs:
            body_sprite = np.maximum(
                body_sprite,
                sprite_palette_heatmap(
                    crop_bgr,
                    [tuple(dominant)],
                    heatmap_detector.max_sprite_palette_distance,
                ),
            )
            accent_sprite = np.maximum(
                accent_sprite,
                sprite_palette_heatmap(
                    crop_bgr,
                    [tuple(accent)],
                    heatmap_detector.accent_structural_distance,
                ),
            )

    final = np.zeros(crop_bgr.shape[:2], dtype=np.float32)
    scales = heatmap_detector._center_scales(crop_bgr.shape[1])
    if scale not in scales:
        scales = [scale, *scales]
    for track_scale in scales:
        window = (
            max(3, int(round(descriptor.avg_width * track_scale)) | 1),
            max(3, int(round(descriptor.avg_height * track_scale)) | 1),
        )
        sprite_heat = cv2.blur(sprite, window)
        color_heat = cv2.blur(color_signal, window)
        combined = np.maximum(sprite_heat * 0.75, color_heat * 0.55).astype(np.float32)
        final = np.maximum(final, combined)
        if body_sprite is not None and accent_sprite is not None:
            body_heat = cv2.blur(body_sprite, window)
            accent_heat = cv2.blur(accent_sprite, window)
            structural_heat = np.sqrt(body_heat * accent_heat).astype(np.float32)
            final = np.maximum(final, structural_heat)
    return final
