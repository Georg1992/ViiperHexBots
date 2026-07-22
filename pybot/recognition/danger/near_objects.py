"""Detect non-player sprites very near the character (center of client).

The player (body + Hunter falcon + nameplate) is treated as one stack and
excluded. Any other sizable foreground blob whose box reaches within
``near_cells`` of the character center counts as danger.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# Defaults match RO cell size and "melee / very near" feel.
DEFAULT_NEAR_CELLS = 1.5
_MIN_BLOB_AREA = 80
_BODY_MIN_HEIGHT = 24
_BIRD_UP_PX = 56
_NAME_DOWN_PX = 24
_CHAR_PAD_X = 16
_CENTER_DY = 8
_SELF_INTERSECTION_FRAC = 0.35


@dataclass(frozen=True)
class NearObjectsResult:
    count: int


def count_near_objects(
    frame_bgr: np.ndarray,
    *,
    cell_size_px: int = 64,
    near_cells: float = DEFAULT_NEAR_CELLS,
) -> NearObjectsResult:
    """Return how many foreign blobs sit within *near_cells* of the player."""
    if frame_bgr is None or frame_bgr.size == 0:
        return NearObjectsResult(count=0)
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] < 3:
        return NearObjectsResult(count=0)

    h, w = frame_bgr.shape[:2]
    cx, cy = w // 2, h // 2 + _CENTER_DY
    near_r = max(1, int(near_cells * cell_size_px))
    pad = 48
    x0, y0 = max(0, cx - near_r - pad), max(0, cy - near_r - pad)
    x1, y1 = min(w, cx + near_r + pad), min(h, cy + near_r + pad)
    crop = frame_bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return NearObjectsResult(count=0)

    mask = _foreground_mask(crop)
    num, _labels, stats, cents = cv2.connectedComponentsWithStats(mask * 255, 8)
    lcx, lcy = cx - x0, cy - y0

    comps: list[tuple[int, tuple[int, int, int, int], tuple[float, float]]] = []
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < _MIN_BLOB_AREA:
            continue
        box = (
            int(stats[i, cv2.CC_STAT_LEFT]),
            int(stats[i, cv2.CC_STAT_TOP]),
            int(stats[i, cv2.CC_STAT_WIDTH]),
            int(stats[i, cv2.CC_STAT_HEIGHT]),
        )
        cent = (float(cents[i, 0]), float(cents[i, 1]))
        comps.append((area, box, cent))

    body = _pick_body(comps, lcx, lcy)
    if body is None:
        return NearObjectsResult(count=0)

    _area, (bx, by, bw, bh), _cent = body
    stack = (
        bx - _CHAR_PAD_X,
        by - _BIRD_UP_PX,
        bw + 2 * _CHAR_PAD_X,
        bh + _BIRD_UP_PX + _NAME_DOWN_PX,
    )

    foreign = 0
    for area, box, cent in comps:
        if _is_self(box, cent, stack, area):
            continue
        if _box_distance_to_point(box, lcx, lcy) <= near_r:
            foreign += 1
    return NearObjectsResult(count=foreign)


def _foreground_mask(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    mask = (((sat > 28) & (val > 40)) | ((sat > 18) & (val > 90))).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return mask


def _pick_body(
    comps: list[tuple[int, tuple[int, int, int, int], tuple[float, float]]],
    lcx: int,
    lcy: int,
) -> tuple[int, tuple[int, int, int, int], tuple[float, float]] | None:
    """Prefer a tall center component (skip flat nameplate streaks)."""
    best: tuple[int, tuple[int, int, int, int], tuple[float, float]] | None = None
    best_score = -1.0
    for area, box, cent in comps:
        bx, by, bw, bh = box
        contains = bx <= lcx < bx + bw and by <= lcy < by + bh
        dist = ((cent[0] - lcx) ** 2 + (cent[1] - lcy) ** 2) ** 0.5
        if not contains and dist > 50:
            continue
        if bh < _BODY_MIN_HEIGHT:
            continue
        score = float(area) + (500.0 if contains else 0.0) + bh * 5.0
        if score > best_score:
            best_score = score
            best = (area, box, cent)
    if best is not None:
        return best
    for area, box, cent in sorted(comps, key=lambda c: -c[0]):
        dist = ((cent[0] - lcx) ** 2 + (cent[1] - lcy) ** 2) ** 0.5
        if dist < 70:
            return (area, box, cent)
    return None


def _is_self(
    box: tuple[int, int, int, int],
    cent: tuple[float, float],
    stack: tuple[int, int, int, int],
    area: int,
) -> bool:
    sx, sy, sw, sh = stack
    if sx <= cent[0] < sx + sw and sy <= cent[1] < sy + sh:
        return True
    inter = _box_intersection_area(box, stack)
    return inter >= _SELF_INTERSECTION_FRAC * area


def _box_intersection_area(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> int:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix0, iy0 = max(ax, bx), max(ay, by)
    ix1, iy1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0
    return (ix1 - ix0) * (iy1 - iy0)


def _box_distance_to_point(
    box: tuple[int, int, int, int],
    px: int,
    py: int,
) -> float:
    x, y, w, h = box
    cx = min(max(px, x), x + w - 1)
    cy = min(max(py, y), y + h - 1)
    return float(((cx - px) ** 2 + (cy - py) ** 2) ** 0.5)
