"""Silhouette extraction for noisy discovery heatmaps.

Uses the heatmap blob outline to bound palette matching so grass pixels
outside the discovered object are excluded.
"""

from __future__ import annotations

import cv2
import numpy as np

from pybot.recognition.detector.scoring.heatmap_detector import sprite_palette_heatmap

NOISY_MIN_SILHOUETTE_SIMILARITY = 0.45


def _select_best_contour(
    contours: list[np.ndarray],
    heatmap_crop: np.ndarray,
    target_x: int,
    target_y: int,
) -> np.ndarray | None:
    best_contour = None
    best_score = float("-inf")
    for contour in contours:
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue
        center_x = int(moments["m10"] / moments["m00"])
        center_y = int(moments["m01"] / moments["m00"])
        filled = np.zeros(heatmap_crop.shape[:2], dtype=np.uint8)
        cv2.drawContours(filled, [contour], -1, 1, thickness=-1)
        heat_mass = float(heatmap_crop[filled.astype(bool)].sum())
        distance_sq = (center_x - target_x) ** 2 + (center_y - target_y) ** 2
        score = heat_mass - distance_sq * 0.002
        if score > best_score:
            best_score = score
            best_contour = contour
    return best_contour


def _trim_background_fringe(
    occupancy: np.ndarray,
    background_heat: np.ndarray,
    *,
    fringe_threshold: float = 0.45,
) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    outer_band = (
        cv2.dilate(occupancy.astype(np.uint8), kernel, iterations=2).astype(bool)
        & ~cv2.erode(occupancy.astype(np.uint8), kernel, iterations=1).astype(bool)
    )
    trimmed = occupancy.copy()
    trimmed[outer_band & (background_heat >= fringe_threshold)] = False
    return trimmed


def _mask_plausible(
    comp_mask: np.ndarray,
    *,
    avg_width: float,
    avg_height: float,
) -> bool:
    height, width = comp_mask.shape[:2]
    if height < 4 or width < 4:
        return False
    if width > avg_width * 1.4 or height > avg_height * 1.4:
        return False
    if width < avg_width * 0.35 or height < avg_height * 0.35:
        return False
    aspect = width / max(height, 1)
    expected_aspect = avg_width / max(avg_height, 1.0)
    if aspect < expected_aspect * 0.45 or aspect > expected_aspect * 2.0:
        return False
    return True


def extract_heatmap_outline_mask(
    search_bgr: np.ndarray,
    heatmap_crop: np.ndarray,
    *,
    local_ref_left: int,
    local_ref_top: int,
    local_ref_width: int,
    local_ref_height: int,
    exclude_palette_bgr: list[tuple[int, int, int]],
    max_sprite_palette_distance: float,
    min_heat: float = 0.04,
    peak_relative_threshold: float = 0.45,
    background_match_threshold: float = 0.55,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (mob_region_bgr, occupancy_mask) from a heatmap-seeded object outline."""
    if search_bgr.size == 0 or heatmap_crop.size == 0:
        return None

    background_heat = sprite_palette_heatmap(
        search_bgr,
        exclude_palette_bgr,
        max_sprite_palette_distance,
    )
    non_background = background_heat < background_match_threshold

    threshold = max(float(heatmap_crop.max()) * peak_relative_threshold, min_heat)
    seed = ((heatmap_crop >= threshold) & non_background).astype(np.uint8)
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    seed = cv2.morphologyEx(seed, cv2.MORPH_CLOSE, kernel_large, iterations=2)
    dilated = cv2.dilate(seed, kernel_large, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    target_x = local_ref_left + local_ref_width // 2
    target_y = local_ref_top + local_ref_height // 2
    best_contour = _select_best_contour(contours, heatmap_crop, target_x, target_y)
    if best_contour is None:
        return None

    filled = np.zeros(search_bgr.shape[:2], dtype=np.uint8)
    cv2.drawContours(filled, [best_contour], -1, 1, thickness=-1)
    occupancy = filled.astype(bool) & non_background
    occupancy = cv2.morphologyEx(
        occupancy.astype(np.uint8),
        cv2.MORPH_CLOSE,
        kernel_large,
        iterations=2,
    ).astype(bool) & non_background
    occupancy = cv2.morphologyEx(
        occupancy.astype(np.uint8),
        cv2.MORPH_OPEN,
        kernel_small,
        iterations=1,
    ).astype(bool)
    occupancy = _trim_background_fringe(occupancy, background_heat)
    if not np.any(occupancy):
        return None

    ys, xs = np.where(occupancy)
    left = int(xs.min())
    right = int(xs.max()) + 1
    top = int(ys.min())
    bottom = int(ys.max()) + 1
    mob_region = search_bgr[top:bottom, left:right]
    comp_mask = occupancy[top:bottom, left:right]
    if mob_region.size == 0 or not np.any(comp_mask):
        return None

    return mob_region, comp_mask
