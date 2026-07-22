"""Center-screen character pose (sitting vs standing) from client frames.

RO keeps the player at the client center. Sitting shrinks the body sprite;
the Hunter falcon floats above and is ignored by taking the largest contiguous
vertical occupancy run in a narrow center strip (bird sits in a separate run
when there is a gap above the body).

Sit interrupt / combat threat uses ``pybot.recognition.danger``, not this module.
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
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    profile = _central_column_profile(mask)
    thr = max(4, int(0.08 * (crop.shape[1] * 0.40)))
    body_h = _largest_run_height(profile, thr)
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


def _largest_run_height(profile: np.ndarray, thr: int) -> int | None:
    runs: list[int] = []
    start: int | None = None
    for i, occupied in enumerate(profile >= thr):
        if occupied and start is None:
            start = i
        elif not occupied and start is not None:
            runs.append(i - start)
            start = None
    if start is not None:
        runs.append(len(profile) - start)
    if not runs:
        return None
    return max(runs)
