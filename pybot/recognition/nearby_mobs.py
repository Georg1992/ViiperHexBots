"""Count non-character sprites near the player in a hunt frame.

Used by ``CharacterStateMonitor`` to publish ``nearby_any_mobs_count``.
``DangerDetector`` only reads that published count — it does not run CV.
"""

from __future__ import annotations

import cv2
import numpy as np

# 250x250 crop around the hunt-frame center (character).
_NEARBY_CROP_HALF = 125
_NEARBY_MIN_BLOB_AREA = 30
# Exclude blobs this close to center (character + falcon).
_NEARBY_CHAR_RADIUS = 80
_CENTER_DY = 8  # feet sit slightly below geometric center
_MORPH_KERNEL = np.ones((3, 3), np.uint8)


def detect_nearby_any_mobs(frame_bgr: np.ndarray) -> int:
    """Count distinct non-character sprite blobs near frame center.

    Returns ``0`` when no other sprites are nearby.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return 0
    h, w = frame_bgr.shape[:2]
    cx, cy = w // 2, h // 2 + _CENTER_DY

    x1 = max(0, cx - _NEARBY_CROP_HALF)
    y1 = max(0, cy - _NEARBY_CROP_HALF)
    x2 = min(w, cx + _NEARBY_CROP_HALF)
    y2 = min(h, cy + _NEARBY_CROP_HALF)
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return 0

    mask = _foreground_mask(crop)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _MORPH_KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _MORPH_KERNEL)

    num_labels, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 4)
    if num_labels <= 1:
        return 0

    crop_cx = crop.shape[1] // 2
    crop_cy = crop.shape[0] // 2
    radius_sq = _NEARBY_CHAR_RADIUS * _NEARBY_CHAR_RADIUS

    nearby = 0
    for lbl in range(1, num_labels):
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        if area < _NEARBY_MIN_BLOB_AREA:
            continue
        bx = int(centroids[lbl, 0])
        by = int(centroids[lbl, 1])
        dx = bx - crop_cx
        dy = by - crop_cy
        if dx * dx + dy * dy <= radius_sq:
            continue  # character / falcon
        nearby += 1

    return nearby


def _foreground_mask(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    return (((sat > 28) & (val > 40)) | ((sat > 18) & (val > 90))).astype(np.uint8)
