"""Cluster match-palette BGR entries into perceptually distinct Lab groups."""

from __future__ import annotations

import cv2
import numpy as np

_MAX_GROUPS = 5
_LAB_MERGE_THRESHOLD = 18.0


def _palette_lab(palette_bgr: list[tuple[int, int, int]]) -> np.ndarray:
    bgr = np.asarray(palette_bgr, dtype=np.uint8).reshape(-1, 1, 3)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)


def _closest_pair(groups: list[list[int]], lab: np.ndarray) -> tuple[int, int, float]:
    def centroid(indices: list[int]) -> np.ndarray:
        return lab[indices].mean(axis=0)

    best_i, best_j, best_d = 0, 1, float("inf")
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            d = float(np.linalg.norm(centroid(groups[i]) - centroid(groups[j])))
            if d < best_d:
                best_i, best_j, best_d = i, j, d
    return best_i, best_j, best_d


def cluster_match_palette_groups(
    palette_bgr: list[tuple[int, int, int]],
    *,
    max_groups: int = _MAX_GROUPS,
    lab_merge_threshold: float = _LAB_MERGE_THRESHOLD,
) -> list[list[int]]:
    """Cluster palette indices into at most ``max_groups`` Lab-similar families.

    Nearby shades (Lab distance < threshold) share a group so several pink
    entries count as one coverage family. Remaining groups are merged by
    closest Lab centroid until ``max_groups``.
    """
    n = len(palette_bgr)
    if n == 0:
        return []
    if n == 1:
        return [[0]]

    lab = _palette_lab(palette_bgr)
    groups: list[list[int]] = [[i] for i in range(n)]

    while len(groups) > 1:
        i, j, dist = _closest_pair(groups, lab)
        if dist >= lab_merge_threshold:
            break
        merged = groups[i] + groups[j]
        groups = [g for k, g in enumerate(groups) if k not in (i, j)]
        groups.append(merged)

    while len(groups) > max_groups:
        i, j, _dist = _closest_pair(groups, lab)
        merged = groups[i] + groups[j]
        groups = [g for k, g in enumerate(groups) if k not in (i, j)]
        groups.append(merged)

    groups = [sorted(g) for g in groups]
    groups.sort(key=lambda g: g[0])
    return groups


def split_palette_groups_by_required(
    groups: list[list[int]],
    color_required: list[bool],
) -> tuple[list[list[int]], list[list[int]]]:
    """Split Lab groups into required vs optional color index sets.

    Frame-stable palette colors form required groups (hard gate + diversity
    bar). Frame-intermittent colors (eyes, blinks, rare accents) form optional
    groups even when Lab-merged with a required family — they must not raise
    the diversity bar, but soft heatmap diversity can still boost when they
    are locally present.

    Mixed Lab families are peeled: required indices stay in the required set,
    optional indices become their own optional group (never swallowed).
    """
    required: list[list[int]] = []
    optional: list[list[int]] = []
    for group in groups:
        req_idx = [
            idx for idx in group
            if idx < len(color_required) and color_required[idx]
        ]
        opt_idx = [
            idx for idx in group
            if idx < len(color_required) and not color_required[idx]
        ]
        # Indices outside color_required (should not happen) stay required-side
        # so they are not silently dropped.
        stray = [
            idx for idx in group
            if idx >= len(color_required)
        ]
        if req_idx or stray:
            required.append(sorted(req_idx + stray))
        if opt_idx:
            optional.append(sorted(opt_idx))
    return required, optional
