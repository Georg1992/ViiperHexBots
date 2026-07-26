"""Opacity decay probe — death detection during local tracking.

Living sprites have dense body/sprite palette coverage in their bbox.
On death the game fades the sprite in place; coverage drops while the
heatmap can still follow residual color. Baseline the living score, then
confirm death after a sustained drop while stationary (see
``evaluate_opacity_death`` in ``rules``).
"""

from __future__ import annotations

import cv2
import numpy as np

from pybot.recognition.detector.descriptors.descriptor import MobDescriptor
from pybot.recognition.detector.scoring.heatmap_detector import (
    palette_heatmap,
    sprite_palette_heatmap,
)


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
