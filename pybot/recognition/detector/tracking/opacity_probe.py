"""Opacity decay probe — death detection during local tracking."""

from __future__ import annotations

import cv2
import numpy as np

from pybot.recognition.detector.descriptors.descriptor import MobDescriptor
from pybot.recognition.detector.scoring.heatmap_detector import palette_heatmap, sprite_palette_heatmap


def _top_match_score(heatmap: np.ndarray, fraction: float) -> float:
    if heatmap.size == 0:
        return 0.0
    flat = heatmap.reshape(-1)
    keep = max(1, int(round(len(flat) * fraction)))
    top = np.partition(flat, len(flat) - keep)[-keep:]
    return float(np.clip(top.mean(), 0.0, 1.0))


def measure_opacity_score(
    frame_bgr: np.ndarray,
    descriptor: MobDescriptor,
    bbox: tuple[int, int, int, int],
    max_sprite_palette_distance: float,
    min_sprite_palette_match: float,
) -> float:
    """Estimate sprite opacity/solidity in a tracked mob window.

    Fading corpses blend with the background, lowering informative pixel
    coverage and body signal relative to a living mob.
    """
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        return 0.0
    frame_h, frame_w = frame_bgr.shape[:2]
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(frame_w, x + w)
    y1 = min(frame_h, y + h)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    region_bgr = frame_bgr[y0:y1, x0:x1]
    if region_bgr.size == 0:
        return 0.0

    body_heat = palette_heatmap(region_bgr, descriptor.body_palette)
    sprite_palette_heat = sprite_palette_heatmap(
        region_bgr,
        descriptor.match_palette_bgr,
        max_sprite_palette_distance,
    )

    body = _top_match_score(body_heat, 0.22)
    sprite_pixels = sprite_palette_heat >= min_sprite_palette_match
    informative_fraction = float(sprite_pixels.mean()) if sprite_pixels.size else 0.0
    body_coverage = float((body_heat >= 0.25).mean()) if body_heat.size else 0.0

    gray = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2GRAY)
    contrast = float(np.std(gray)) / 128.0

    # Coverage of sprite/body colors drops first as the corpse fades; contrast
    # is a light tie-breaker so flat background windows score near zero.
    opacity = (
        0.50 * informative_fraction
        + 0.30 * body_coverage
        + 0.12 * body
        + 0.08 * min(contrast, 1.0)
    )
    return float(np.clip(opacity, 0.0, 1.0))


def calibrate_opacity_baseline(
    *,
    opacity_score: float,
    baseline: float,
    baseline_samples: int,
    config: dict,
) -> tuple[float, int]:
    """Accumulate baseline samples while the mob is alive."""
    min_samples = int(config["deathOpacityBaselineSamples"])
    min_baseline = float(config["deathOpacityMinBaseline"])

    if baseline_samples < min_samples:
        return max(baseline, opacity_score), baseline_samples + 1

    if baseline < min_baseline:
        return max(baseline, opacity_score), baseline_samples

    return baseline, baseline_samples


def is_opacity_calibrated(
    *,
    baseline: float,
    baseline_samples: int,
    config: dict,
) -> bool:
    min_samples = int(config["deathOpacityBaselineSamples"])
    min_baseline = float(config["deathOpacityMinBaseline"])
    return baseline_samples >= min_samples and baseline >= min_baseline


def evaluate_opacity_death(
    *,
    opacity_score: float,
    baseline: float,
    baseline_samples: int,
    decay_streak: int,
    config: dict,
    moving: bool = False,
    now_tick: int,
) -> tuple[float, int, int, bool]:
    """Update opacity baseline state and return whether death is confirmed.

    ``decay_streak`` stores the monotonic tick when a meaningful fade began
    (0 = not fading). A meaningful drop is ``opacity_score`` below
    ``baseline * deathOpacityDropRatio``. Dead sprites stop moving, so the
    fade clock only runs while stationary; a drop while ``moving`` holds the
    start tick (not opacity loss). Recovery clears it. Death requires the fade
    to last ``deathOpacityConfirmMs`` wall-clock while stationary.

    Tracking runs unbound, so tick-count confirms are not used.
    """
    min_samples = int(config["deathOpacityBaselineSamples"])
    min_baseline = float(config["deathOpacityMinBaseline"])
    drop_ratio = float(config["deathOpacityDropRatio"])
    confirm_ms = int(config["deathOpacityConfirmMs"])

    if baseline_samples < min_samples:
        baseline = max(baseline, opacity_score)
        baseline_samples += 1
        return baseline, baseline_samples, 0, False

    if baseline < min_baseline:
        baseline = max(baseline, opacity_score)
        return baseline, baseline_samples, decay_streak, False

    dropped = opacity_score < (baseline * drop_ratio)
    if dropped and not moving:
        if decay_streak <= 0:
            decay_streak = now_tick
        dead = (now_tick - decay_streak) >= confirm_ms
        if dead:
            return baseline, baseline_samples, 0, True
        return baseline, baseline_samples, decay_streak, False
    if dropped and moving:
        # Walk / attack blur can look like a drop — hold the fade clock.
        return baseline, baseline_samples, decay_streak, False

    return baseline, baseline_samples, 0, False
