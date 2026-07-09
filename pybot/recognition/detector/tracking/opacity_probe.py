"""Opacity decay probe — death detection during local tracking."""

from __future__ import annotations

import cv2
import numpy as np

from pybot.recognition.detector.descriptors.descriptor import MobDescriptor
from pybot.recognition.detector.scoring.heatmap_detector import palette_heatmap, sprite_palette_heatmap
from pybot.recognition.detector.scoring.region_scorer import RegionScorer


def measure_opacity_score(
    frame_bgr: np.ndarray,
    hsv: np.ndarray,
    descriptor: MobDescriptor,
    bbox: tuple[int, int, int, int],
    region_scorer: RegionScorer,
) -> float:
    """Estimate sprite opacity/solidity in a tracked mob window.

    Fading corpses blend with the background, lowering informative pixel
    coverage, palette purity, and body signal relative to a living mob.
    """
    x, y, w, h = bbox
    region_bgr = frame_bgr[y : y + h, x : x + w]
    region_hsv = hsv[y : y + h, x : x + w]
    if region_bgr.size == 0:
        return 0.0

    body_heat = palette_heatmap(region_hsv, descriptor.body_palette)
    accent_heat = palette_heatmap(region_hsv, descriptor.accent_colors)
    rare_heat = palette_heatmap(region_hsv, descriptor.rare_colors)
    descriptor_heat = np.maximum.reduce([body_heat, accent_heat, rare_heat])
    sprite_palette_heat = sprite_palette_heatmap(
        region_bgr,
        descriptor.match_palette_bgr,
        region_scorer.max_sprite_palette_distance,
    )

    body = RegionScorer._top_match_score(body_heat, 0.22)
    sprite_pixels = sprite_palette_heat >= region_scorer.min_sprite_palette_match
    informative_fraction = float(sprite_pixels.mean()) if sprite_pixels.size else 0.0
    purity = informative_fraction

    gray = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2GRAY)
    contrast = float(np.std(gray)) / 128.0

    opacity = (
        0.40 * informative_fraction
        + 0.35 * purity
        + 0.15 * body
        + 0.10 * min(contrast, 1.0)
    )
    return float(np.clip(opacity, 0.0, 1.0))


def calibrate_opacity_baseline(
    *,
    opacity_score: float,
    baseline: float,
    baseline_samples: int,
    config: dict,
) -> tuple[float, int]:
    """Accumulate baseline samples while the mob is alive and stationary."""
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
    decay_ratio_limit: float | None = None,
) -> tuple[float, int, int, bool]:
    """Update opacity baseline state and return whether death is confirmed."""
    min_samples = int(config["deathOpacityBaselineSamples"])
    min_baseline = float(config["deathOpacityMinBaseline"])
    decay_ratio_limit = (
        float(decay_ratio_limit)
        if decay_ratio_limit is not None
        else float(config["deathOpacityDecayRatio"])
    )
    confirm_ticks = int(config["deathOpacityConfirmTicks"])

    if baseline_samples < min_samples:
        baseline = max(baseline, opacity_score)
        baseline_samples += 1
        decay_streak = 0
        return baseline, baseline_samples, decay_streak, False

    if baseline < min_baseline:
        baseline = max(baseline, opacity_score)
        return baseline, baseline_samples, decay_streak, False

    ratio = opacity_score / baseline if baseline > 0.0 else 1.0
    if ratio <= decay_ratio_limit + 1e-6:
        decay_streak += 1
    else:
        decay_streak = 0

    dead = decay_streak >= confirm_ticks
    if dead:
        decay_streak = 0
    return baseline, baseline_samples, decay_streak, dead
