"""Silhouette helpers for descriptor build and silhouette gate."""

from __future__ import annotations

import cv2
import numpy as np

# Occupancy at/above this is a hard cell (same cutoff as soft-membership cores).
HARD_OCCUPANCY = 0.5
# Soft Tversky FP weight for hard cells outside the soft ref:
# 1× soft-Jaccard mass + the gap below full occupancy at the hard cutoff.
SILHOUETTE_HARD_FP_WEIGHT = 1.0 + HARD_OCCUPANCY


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


def _soft_membership(occupancy: np.ndarray, *, radius: float = 1.0) -> np.ndarray:
    """Occupancy in [0, 1], with a 1-cell soft halo around hard cores.

    Interior hard cells stay 1. Soft-gray keeps its value. Background cells
    within ``radius`` of a hard core fall off as ``1 - dist/radius``.
    """
    occ = np.clip(np.asarray(occupancy, dtype=np.float32), 0.0, 1.0)
    hard = (occ >= np.float32(HARD_OCCUPANCY)).astype(np.uint8)
    if not np.any(hard):
        return occ
    # distanceTransform: zeros stay 0; nonzeros = distance to nearest zero.
    dist = cv2.distanceTransform((1 - hard).astype(np.uint8), cv2.DIST_L2, 3)
    halo = np.clip(
        np.float32(1.0) - dist / np.float32(radius),
        0.0,
        1.0,
    ).astype(np.float32)
    return np.maximum(occ, halo)


def silhouette_similarity(candidate: np.ndarray, reference: np.ndarray, stable_mask: np.ndarray) -> float:
    """Soft Tversky similarity of reference and candidate occupancy.

    .. math::

        T = \\frac{\\sum_i \\min(A_i, B_i)}
                  {\\sum_i \\min(A_i, B_i)
                   + w \\sum_i \\max(H_i - A_i, 0)
                   + \\sum_i \\max(A_i - B_i, 0)}

    ``A`` is the stable reference (hard cells + 1-cell soft halo).
    ``B`` is the candidate (soft-gray kept, hard cores + same 1-cell halo).
    ``H`` is hard candidate occupancy (``>= HARD_OCCUPANCY``) — soft-gray
    outside the ref is not charged as false mass; solid fill-in is.
    ``w = SILHOUETTE_HARD_FP_WEIGHT`` so hard extras cost more than holes.

    Scores are absolute-grid (no per-ref translation search): independently
    maximizing shift per facing flattens wrong-pose scores into the true match.
    One-cell soft halo already absorbs small framing jitter.
    """
    if reference.size == 0 or not np.any(stable_mask):
        return 1.0

    shape = reference.shape
    stable = stable_mask.reshape(shape)
    ref_hard = ((reference >= HARD_OCCUPANCY) & stable).astype(np.float32)
    cand_raw = np.asarray(candidate, dtype=np.float32).reshape(shape)

    ref = _soft_membership(ref_hard, radius=1.0)
    cand = _soft_membership(cand_raw, radius=1.0)
    inter = float(np.sum(np.minimum(ref, cand)))
    hard = (cand_raw >= HARD_OCCUPANCY).astype(np.float32)
    false_pos = float(np.sum(np.maximum(hard - ref, 0.0)))
    false_neg = float(np.sum(np.maximum(ref - cand, 0.0)))
    denom = inter + SILHOUETTE_HARD_FP_WEIGHT * false_pos + false_neg
    if denom <= 0.0:
        return 0.0
    return float(np.clip(inter / denom, 0.0, 1.0))


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
