"""Center-screen character pose (sitting vs standing) from client frames.

RO keeps the player at the client center. Sitting shrinks the body sprite;
the Hunter falcon floats above and is ignored by taking the lowest contiguous
vertical occupancy run (>= 20px) in a narrow center strip (bird sits in a
separate run when there is a gap above the body).

Danger detection is handled by :class:`DangerDetector`, not this module.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# Crop around client center (px). Tall enough for standing + falcon, narrow
# enough that nearby NPCs/mobs stay outside the body strip.
_CENTER_HALF_W = 64
_CENTER_HALF_H = 96
_CENTER_DY = 8  # feet sit slightly below geometric center
_MORPH_OPEN_KERNEL = np.ones((2, 2), np.uint8)
_MORPH_CLOSE_KERNEL = np.ones((3, 3), np.uint8)


@dataclass(frozen=True)
class CharacterPose:
    """Body occupancy measured at the client center."""

    body_height: int
    fg_count: int


def measure_center_pose(frame_bgr: np.ndarray) -> CharacterPose | None:
    """Return center body pose, or None if no reliable sprite is found."""
    if frame_bgr is None or frame_bgr.size == 0:
        return None
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] < 3:
        return None
    h, w = frame_bgr.shape[:2]
    if w < 2 * _CENTER_HALF_W + 1 or h < 2 * _CENTER_HALF_H + 1:
        return None

    cx, cy = w // 2, h // 2 + _CENTER_DY
    crop = frame_bgr[
        cy - _CENTER_HALF_H : cy + _CENTER_HALF_H,
        cx - _CENTER_HALF_W : cx + _CENTER_HALF_W,
    ]
    mask = _foreground_mask(crop)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _MORPH_OPEN_KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _MORPH_CLOSE_KERNEL)

    profile = _central_column_profile(mask)
    thr = max(4, int(0.08 * (crop.shape[1] * 0.40)))
    body_h = _body_run_height(profile, thr)
    if body_h is None or body_h < 20:
        return None
    return CharacterPose(body_height=body_h, fg_count=int(mask.sum()))


def _foreground_mask(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    # Dark grid floor is low-sat; sprites are colored or brighter edged.
    return (((sat > 28) & (val > 40)) | ((sat > 18) & (val > 90))).astype(np.uint8)


def _central_column_profile(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    x0, x1 = int(w * 0.30), int(w * 0.70)
    return mask[:, x0:x1].sum(axis=1)


# ── Sit/Stand threshold ───────────────────────────────────────────
# Body-height thresholds calibrated for 128x192 center crop.
# Sitting: ~55-73px, Standing: ~88-108px (with or without falcon).
_SIT_BODY_HEIGHT_THRESHOLD = 80


def check_is_sitting(pose: CharacterPose | None) -> bool | None:
    """Returns True if pose indicates sitting, False if standing, None if unknown."""
    if pose is None:
        return None
    return pose.body_height < _SIT_BODY_HEIGHT_THRESHOLD


def check_is_standing(pose: CharacterPose | None) -> bool | None:
    """Returns True if pose indicates standing, False if sitting, None if unknown."""
    if pose is None:
        return None
    return pose.body_height >= _SIT_BODY_HEIGHT_THRESHOLD


# ── Generic nearby-mob detection (250x250 center crop) ────────────
# Crop used to detect ANY sprite blobs near the character (not just hunted ones).
_NEARBY_CROP_HALF = 125  # 250x250
_NEARBY_MIN_BLOB_AREA = 30
_NEARBY_CHAR_RADIUS = 80  # px from center — includes character + falcon


def detect_nearby_any_mobs(frame_bgr: np.ndarray) -> int:
    """Count distinct non-character sprite blobs in a 250x250 center crop.

    Any mob sprite that walks near the character appears as a saturated
    foreground blob in the center crop.  The character sprite itself is the
    largest blob near the center and is excluded.

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
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _MORPH_CLOSE_KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _MORPH_CLOSE_KERNEL)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 4)
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
        # Centroids: [col, row] (x, y) in crop coordinates
        bx = int(centroids[lbl, 0])
        by = int(centroids[lbl, 1])
        dx = bx - crop_cx
        dy = by - crop_cy
        if dx * dx + dy * dy <= radius_sq:
            continue  # Too close to center — likely the character sprite
        nearby += 1

    return nearby


def _body_run_height(profile: np.ndarray, thr: int) -> int | None:
    """Height of the body sprite — lowest run that is tall enough to be a body.

    The hunter falcon floats above the character's head.  When there is a gap
    the falcon forms a separate, shorter run above the body.  Floor noise at
    the very bottom of the crop produces very short runs (< 20px).

    Strategy: filter to runs >= 20px tall (body), then take the lowest one
    (largest start row). This picks the body over the falcon (higher up) and
    ignores floor noise.
    """
    runs: list[tuple[int, int]] = []  # (start_row, height)
    start: int | None = None
    for i, occupied in enumerate(profile >= thr):
        if occupied and start is None:
            start = i
        elif not occupied and start is not None:
            h = i - start
            if h >= 20:
                runs.append((start, h))
            start = None
    if start is not None:
        h = len(profile) - start
        if h >= 20:
            runs.append((start, h))
    if not runs:
        return None
    # Lowest on screen = largest start_row.
    _start, height = max(runs, key=lambda item: item[0])
    return height
