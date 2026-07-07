"""Screen region capture for template matching."""

import numpy as np
import cv2
import mss

_sct = None


def capture_region(x: int, y: int, width: int, height: int) -> np.ndarray:
    """Capture a screen rectangle and return a BGR image."""
    if width <= 0 or height <= 0:
        raise ValueError("capture width and height must be positive")

    global _sct
    if _sct is None:
        _sct = mss.mss()
    shot = _sct.grab({"left": x, "top": y, "width": width, "height": height})
    frame = np.array(shot)
    return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
