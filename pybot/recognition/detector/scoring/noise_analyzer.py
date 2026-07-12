"""Detect diffuse sprite-heatmap noise before discovery heatmap build."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pybot.recognition.detector.descriptors.descriptor import MobDescriptor
from pybot.recognition.detector.scoring.heatmap_detector import sprite_palette_heatmap


@dataclass(frozen=True)
class NoiseSignals:
    hot_frac: float
    is_noisy: bool


def analyze_heatmap_noise(
    frame_bgr: np.ndarray,
    descriptor: MobDescriptor,
    *,
    max_sprite_palette_distance: float,
    hot_frac_min: float,
    raw_heat_threshold: float,
) -> NoiseSignals:
    """Return frame-level noise metrics for gated background correction.

    Uses the raw sprite-palette distance heatmap only (no boosts/blur) so the
    gate is cheap and does not depend on correction already being applied.
    """
    raw = sprite_palette_heatmap(
        frame_bgr,
        descriptor.match_palette_bgr,
        max_sprite_palette_distance,
    )
    hot_frac = float(np.mean(raw >= raw_heat_threshold))
    is_noisy = hot_frac >= hot_frac_min
    return NoiseSignals(hot_frac=hot_frac, is_noisy=is_noisy)
