"""Local background analysis for noisy discovery heatmaps.

Texture filtering is applied as a heatmap mask (not a frame rewrite) so the
silhouette gate still sees the original screenshot colours.
"""

from __future__ import annotations

from collections import Counter

import cv2
import numpy as np


def _quantize_bgr(color: tuple[int, int, int], step: int = 16) -> tuple[int, int, int]:
    return tuple(int(channel) // step * step for channel in color)


def smooth_background_field(frame_bgr: np.ndarray, cell_px: int) -> np.ndarray:
    """Interpolate a coarse grid of per-cell BGR medians to full-frame size."""
    if cell_px < 8:
        raise ValueError(f"backgroundGridCellPx must be >= 8, got {cell_px}")

    height, width = frame_bgr.shape[:2]
    grid_h = (height + cell_px - 1) // cell_px
    grid_w = (width + cell_px - 1) // cell_px
    background_grid = np.zeros((grid_h, grid_w, 3), dtype=np.float32)

    for row in range(grid_h):
        y0 = row * cell_px
        y1 = min(height, y0 + cell_px)
        for col in range(grid_w):
            x0 = col * cell_px
            x1 = min(width, x0 + cell_px)
            background_grid[row, col] = np.median(
                frame_bgr[y0:y1, x0:x1].reshape(-1, 3),
                axis=0,
            )

    return cv2.resize(
        background_grid,
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )


def texture_deviation_mask(frame_bgr: np.ndarray, cell_px: int) -> np.ndarray:
    """Return a 0–1 mask that suppresses pixels matching local background texture."""
    background = smooth_background_field(frame_bgr, cell_px)
    residual = np.clip(frame_bgr.astype(np.float32) - background, 0.0, 255.0)
    magnitude = np.sqrt(np.sum(residual * residual, axis=2))

    low = float(np.percentile(magnitude, 40))
    high = float(np.percentile(magnitude, 90))
    if high <= low + 1e-6:
        return np.ones(magnitude.shape, dtype=np.float32)

    return np.clip((magnitude - low) / (high - low), 0.0, 1.0).astype(np.float32)


def dominant_background_colors(
    frame_bgr: np.ndarray,
    cell_px: int,
    *,
    max_colors: int = 8,
) -> list[tuple[int, int, int]]:
    """Collect the most frequent coarse-grid median colours (terrain palette)."""
    if cell_px < 8:
        raise ValueError(f"backgroundGridCellPx must be >= 8, got {cell_px}")

    height, width = frame_bgr.shape[:2]
    counts: Counter[tuple[int, int, int]] = Counter()

    for y0 in range(0, height, cell_px):
        y1 = min(height, y0 + cell_px)
        for x0 in range(0, width, cell_px):
            x1 = min(width, x0 + cell_px)
            median = tuple(
                np.median(frame_bgr[y0:y1, x0:x1].reshape(-1, 3), axis=0).astype(int),
            )
            counts[_quantize_bgr(median)] += 1

    return [color for color, _ in counts.most_common(max_colors)]
