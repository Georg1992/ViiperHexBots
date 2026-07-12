"""Normalized occupancy grid and silhouette helpers."""

from __future__ import annotations

import cv2
import numpy as np

from pybot.recognition.detector.descriptors.descriptor import ColorCluster


def frame_alpha_occupancy_grid(alpha: np.ndarray, grid_size: int) -> np.ndarray:
    height, width = alpha.shape[:2]
    if height <= 0 or width <= 0:
        return np.zeros((grid_size, grid_size), dtype=np.float32)
    resized = cv2.resize(alpha.astype(np.float32), (grid_size, grid_size), interpolation=cv2.INTER_AREA)
    return np.clip(resized, 0.0, 1.0)


def frame_palette_coverage_grid(
    bgr: np.ndarray,
    alpha: np.ndarray,
    palette_bgr: list[tuple[int, int, int]],
    grid_size: int,
    max_distance: float,
) -> np.ndarray:
    if not palette_bgr:
        return np.zeros((grid_size, grid_size), dtype=np.float32)
    palette = np.asarray(palette_bgr, dtype=np.float32)
    pixels = bgr.reshape(-1, 3).astype(np.float32)
    opaque = (alpha.reshape(-1) >= 128)
    match = np.zeros(pixels.shape[0], dtype=bool)
    if np.any(opaque):
        visible = pixels[opaque]
        min_dist_sq = np.full(visible.shape[0], np.inf, dtype=np.float32)
        for start in range(0, len(palette), 64):
            chunk = palette[start : start + 64]
            diff = visible[:, None, :] - chunk[None, :, :]
            dist_sq = np.sum(diff * diff, axis=2)
            min_dist_sq = np.minimum(min_dist_sq, dist_sq.min(axis=1))
        match[opaque] = min_dist_sq <= max_distance * max_distance
    match_img = match.reshape(bgr.shape[:2]).astype(np.float32)
    return frame_alpha_occupancy_grid(match_img * (alpha >= 128).astype(np.float32), grid_size)


def frame_cluster_grid(
    bgr: np.ndarray,
    alpha: np.ndarray,
    clusters: list[ColorCluster],
    grid_size: int,
) -> np.ndarray:
    grid = np.full((grid_size, grid_size), -1, dtype=np.int32)
    if not clusters or not np.any(alpha >= 128):
        return grid
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    height, width = alpha.shape[:2]
    cell_h = height / grid_size
    cell_w = width / grid_size
    centers = np.asarray([cluster.hsv for cluster in clusters], dtype=np.float32)
    tolerances = np.asarray([cluster.tolerance for cluster in clusters], dtype=np.float32)
    for gy in range(grid_size):
        y0 = int(round(gy * cell_h))
        y1 = int(round((gy + 1) * cell_h))
        for gx in range(grid_size):
            x0 = int(round(gx * cell_w))
            x1 = int(round((gx + 1) * cell_w))
            cell_alpha = alpha[y0:y1, x0:x1] >= 128
            if not np.any(cell_alpha):
                continue
            cell_hsv = hsv[y0:y1, x0:x1][cell_alpha]
            best_index = -1
            best_count = 0
            for index, center in enumerate(centers):
                tol = tolerances[index]
                hue_diff = np.abs(cell_hsv[:, 0] - center[0])
                hue_diff = np.minimum(hue_diff, 180.0 - hue_diff)
                sat_diff = np.abs(cell_hsv[:, 1] - center[1])
                val_diff = np.abs(cell_hsv[:, 2] - center[2])
                matched = (
                    (hue_diff <= tol[0]) & (sat_diff <= tol[1]) & (val_diff <= tol[2])
                )
                count = int(np.sum(matched))
                if count > best_count:
                    best_count = count
                    best_index = index
            grid[gy, gx] = best_index
    return grid


def frame_silhouette(alpha: np.ndarray, width: int, height: int) -> np.ndarray:
    if alpha.size == 0:
        return np.zeros((height, width), dtype=np.float32)
    return cv2.resize((alpha >= 128).astype(np.float32), (width, height), interpolation=cv2.INTER_AREA)


def candidate_palette_layout(
    region_bgr: np.ndarray,
    palette_bgr: np.ndarray,
    max_distance: float,
    grid_size: int,
) -> np.ndarray:
    pixels = region_bgr.reshape(-1, 3).astype(np.float32)
    min_dist_sq = np.full(pixels.shape[0], np.inf, dtype=np.float32)
    for start in range(0, len(palette_bgr), 64):
        chunk = palette_bgr[start : start + 64]
        diff = pixels[:, None, :] - chunk[None, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        min_dist_sq = np.minimum(min_dist_sq, dist_sq.min(axis=1))
    match = (min_dist_sq <= max_distance * max_distance).reshape(region_bgr.shape[:2]).astype(np.float32)
    return frame_alpha_occupancy_grid(match, grid_size)


def candidate_silhouette(
    region_bgr: np.ndarray,
    palette_bgr: np.ndarray,
    max_distance: float,
    width: int,
    height: int,
    *,
    occupancy_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Build a 16×16 occupancy grid from palette-matched pixels in a crop.

    Uses Euclidean BGR distance (same metric family as the discovery heatmap).
    Callers may pass a wider ``max_distance`` than the heatmap stage so weak
    low-contrast body pixels are retained without loosening blob discovery.
    """
    region = region_bgr
    if occupancy_mask is not None:
        region = region_bgr.copy()
        region[~occupancy_mask.reshape(region.shape[:2])] = 0

    pixels = region.reshape(-1, 3).astype(np.float32)
    min_dist_sq = np.full(pixels.shape[0], np.inf, dtype=np.float32)
    for start in range(0, len(palette_bgr), 64):
        chunk = palette_bgr[start : start + 64]
        diff = pixels[:, None, :] - chunk[None, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        min_dist_sq = np.minimum(min_dist_sq, dist_sq.min(axis=1))
    match = (min_dist_sq <= max_distance * max_distance).reshape(region.shape[:2])
    match_uint8 = (match.astype(np.uint8)) * 255
    return frame_silhouette(match_uint8, width, height)


def layout_similarity(
    candidate_grid: np.ndarray,
    reference_avg: np.ndarray,
    stable_mask: np.ndarray,
) -> float:
    if reference_avg.size == 0 or not np.any(stable_mask):
        return 1.0
    reference = reference_avg.reshape(-1)[stable_mask.reshape(-1)]
    observed = candidate_grid.reshape(-1)[stable_mask.reshape(-1)]
    if reference.size == 0:
        return 1.0
    denom = max(float(np.max(reference)), 0.05)
    return float(np.clip(1.0 - np.mean(np.abs(observed - reference)) / denom, 0.0, 1.0))


def silhouette_similarity(candidate: np.ndarray, reference: np.ndarray, stable_mask: np.ndarray) -> float:
    """Asymmetric overlap score on the 16×16 silhouette grid.

    Candidate pixels outside the reference silhouette are penalized more heavily
    than reference pixels the candidate misses. Sparse extraction in low-contrast
    scenes therefore still passes at the 0.50 gate, while viewport-filling blobs
    cannot inflate their score by covering unrelated cells.
    """
    if reference.size == 0 or not np.any(stable_mask):
        return 1.0

    stable = stable_mask.reshape(reference.shape)
    ref_bin = ((reference >= 0.5) & stable).astype(np.float32)
    cand_bin = (candidate >= 0.5).astype(np.float32)
    intersection = float(np.sum(ref_bin * cand_bin))
    if intersection <= 0.0:
        return 0.0
    miss = float(np.sum(ref_bin * (1.0 - cand_bin)))
    extra = float(np.sum((1.0 - ref_bin) * cand_bin))
    miss_weight = 0.5
    extra_weight = 1.5
    denom = intersection + miss_weight * miss + extra_weight * extra
    return float(np.clip(intersection / denom, 0.0, 1.0))


def best_silhouette_similarity(
    candidate: np.ndarray,
    references: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[float, int, list[float]]:
    """Score candidate against multiple refs; return best score, index, and all scores."""
    if not references:
        return 1.0, 0, []
    scores: list[float] = []
    best_sim = -1.0
    best_idx = 0
    for idx, (reference, stable_mask) in enumerate(references):
        sim = silhouette_similarity(candidate, reference, stable_mask)
        scores.append(sim)
        if sim > best_sim:
            best_sim = sim
            best_idx = idx
    return float(best_sim), best_idx, scores
