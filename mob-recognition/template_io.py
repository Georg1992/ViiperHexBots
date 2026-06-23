"""PNG template load/save helpers with alpha channel support."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


def load_template_image(image_path: Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Load a template image for OpenCV matching.

    Returns:
        bgr: BGR uint8 image
        mask: single-channel uint8 mask (255 = opaque) or None if no alpha
    """
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"failed to read image: {image_path}")

    if image.ndim == 2:
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return bgr, None

    if image.shape[2] == 4:
        bgr = image[:, :, :3]
        alpha = image[:, :, 3]
        mask = np.where(alpha > 0, 255, 0).astype(np.uint8)
        return bgr, mask

    return image[:, :, :3], None


def save_template_png(output_path: Path, rgba: np.ndarray) -> Path:
    """Save a BGRA/RGBA frame as PNG with alpha."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rgba = tight_crop_rgba(rgba)

    if rgba.ndim != 3 or rgba.shape[2] not in (3, 4):
        raise ValueError("rgba must be HxWx3 or HxWx4")

    if rgba.shape[2] == 3:
        image = rgba
    else:
        image = rgba

    if not cv2.imwrite(str(output_path), image):
        raise IOError(f"failed to write PNG: {output_path}")
    return output_path


def tight_crop_rgba(rgba: np.ndarray) -> np.ndarray:
    """Crop transparent padding so templates match on-screen sprites tightly."""
    if rgba.ndim != 3 or rgba.shape[2] < 4:
        return rgba

    alpha = rgba[:, :, 3]
    rows = np.any(alpha > 0, axis=1)
    cols = np.any(alpha > 0, axis=0)
    if not rows.any() or not cols.any():
        return rgba

    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    return rgba[y1 : y2 + 1, x1 : x2 + 1]
