"""Screen region capture for template matching."""

import numpy as np
import cv2
import mss


def capture_region(x: int, y: int, width: int, height: int) -> np.ndarray:
    """Capture a screen rectangle and return a BGR image."""
    if width <= 0 or height <= 0:
        raise ValueError("capture width and height must be positive")

    with mss.mss() as sct:
        shot = sct.grab({"left": x, "top": y, "width": width, "height": height})
        frame = np.array(shot)
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
