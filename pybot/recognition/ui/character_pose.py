"""Center-screen character pose (sitting vs standing) from hunt frames.

RO keeps the player at the client center. Sitting shrinks the body sprite;
the Hunter falcon floats above and is ignored by taking the lowest contiguous
vertical occupancy run (>= 20px) in a narrow center strip (bird sits in a
separate run when there is a gap above the body).

Nearby-sprite counting lives in ``pybot.recognition.nearby_mobs``.
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


# ── Sit/Stand thresholds ──────────────────────────────────────────
# Calibrated for 128x192 center crop.
# Sitting body alone: ~55-73px. Standing: ~88-108px.
# When the falcon glues onto a sitting sprite the merged run can land
# near the old single threshold (80) and look "standing". Keep a dead
# band so those frames are unknown instead of false standing.
_SIT_BODY_HEIGHT_MAX = 76
_STAND_BODY_HEIGHT_MIN = 86


def check_is_sitting(pose: CharacterPose | None) -> bool | None:
    """True if sitting, False if standing, None if unknown/ambiguous."""
    if pose is None:
        return None
    h = pose.body_height
    if h <= _SIT_BODY_HEIGHT_MAX:
        return True
    if h >= _STAND_BODY_HEIGHT_MIN:
        return False
    return None


def check_is_standing(pose: CharacterPose | None) -> bool | None:
    """True if standing, False if sitting, None if unknown/ambiguous."""
    sitting = check_is_sitting(pose)
    if sitting is None:
        return None
    return not sitting


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
