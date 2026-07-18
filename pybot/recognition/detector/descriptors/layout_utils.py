"""Silhouette helpers for descriptor build and silhouette gate."""

from __future__ import annotations

import cv2
import numpy as np


def frame_silhouette(alpha: np.ndarray, width: int, height: int) -> np.ndarray:
    if alpha.size == 0:
        return np.zeros((height, width), dtype=np.float32)
    return cv2.resize((alpha >= 128).astype(np.float32), (width, height), interpolation=cv2.INTER_AREA)


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
