"""Silhouette helpers for descriptor build and silhouette gate."""

from __future__ import annotations

import cv2
import numpy as np

# Occupancy at/above this is a hard cell (same cutoff as soft-membership cores).
HARD_OCCUPANCY = 0.5


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


def silhouette_match(
    candidate: np.ndarray,
    reference: np.ndarray,
    stable_mask: np.ndarray,
) -> tuple[float, float, float]:
    """Soft occupancy match: (jaccard, precision, recall).

    ``A`` is the stable reference (hard cells + 1-cell soft halo).
    ``B`` is the candidate (soft-gray kept, hard cores + same 1-cell halo).
    ``H`` is hard candidate occupancy (``>= HARD_OCCUPANCY``).

    Soft-gray outside the ref is not charged as false mass; solid fill-in is.

    .. math::

        inter = \\sum_i \\min(A_i, B_i)
        FP = \\sum_i \\max(H_i - A_i, 0)
        FN = \\sum_i \\max(A_i - B_i, 0)
        precision = inter / (inter + FP)
        recall = inter / (inter + FN)
        jaccard = inter / (inter + FP + FN)

    Scores are absolute-grid (no per-ref translation search): independently
    maximizing shift per facing flattens wrong-pose scores into the true match.
    One-cell soft halo already absorbs small framing jitter.
    """
    if reference.size == 0 or not np.any(stable_mask):
        return 0.0, 0.0, 0.0

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

    precision = inter / (inter + false_pos) if (inter + false_pos) > 0.0 else 0.0
    recall = inter / (inter + false_neg) if (inter + false_neg) > 0.0 else 0.0
    denom = inter + false_pos + false_neg
    jaccard = float(np.clip(inter / denom, 0.0, 1.0)) if denom > 0.0 else 0.0
    return jaccard, float(precision), float(recall)


def silhouette_similarity(candidate: np.ndarray, reference: np.ndarray, stable_mask: np.ndarray) -> float:
    """Soft Jaccard similarity of reference and candidate occupancy."""
    jaccard, _precision, _recall = silhouette_match(candidate, reference, stable_mask)
    return jaccard


def best_silhouette_match(
    candidate: np.ndarray,
    references: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[float, int, list[float], float, float]:
    """Score candidate against refs; return best jaccard, index, all scores, prec, recall."""
    if not references:
        return 0.0, 0, [], 0.0, 0.0
    scores: list[float] = []
    best_sim = -1.0
    best_idx = 0
    best_precision = 0.0
    best_recall = 0.0
    for idx, (reference, stable_mask) in enumerate(references):
        jaccard, precision, recall = silhouette_match(candidate, reference, stable_mask)
        scores.append(jaccard)
        if jaccard > best_sim:
            best_sim = jaccard
            best_idx = idx
            best_precision = precision
            best_recall = recall
    return float(best_sim), best_idx, scores, float(best_precision), float(best_recall)


def best_silhouette_similarity(
    candidate: np.ndarray,
    references: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[float, int, list[float]]:
    """Score candidate against multiple refs; return best score, index, and all scores."""
    best_sim, best_idx, scores, _precision, _recall = best_silhouette_match(candidate, references)
    return best_sim, best_idx, scores
