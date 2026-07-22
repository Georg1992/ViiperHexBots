"""Screen region capture for template matching."""

from __future__ import annotations

import threading

import cv2
import mss
import numpy as np
from mss.exception import ScreenShotError

_sct: mss.mss | None = None
_capture_lock = threading.Lock()


def reset_capture_session() -> None:
    """Drop the shared mss session (e.g. after a failed grab or runtime restart).

    If a previous hunt's capture thread is stuck holding the lock, rotate to a
    fresh lock so the next hunt is not blocked forever.
    """
    global _sct, _capture_lock
    lock = _capture_lock
    acquired = lock.acquire(timeout=0.5)
    try:
        if acquired and _sct is not None:
            try:
                _sct.close()
            except Exception:
                pass
        _sct = None
    finally:
        if acquired:
            lock.release()
        _capture_lock = threading.Lock()


def capture_region(x: int, y: int, width: int, height: int) -> np.ndarray | None:
    """Capture a screen rectangle and return a BGR image, or None on failure."""
    if width <= 0 or height <= 0:
        raise ValueError("capture width and height must be positive")

    monitor = {"left": int(x), "top": int(y), "width": int(width), "height": int(height)}
    for attempt in range(2):
        try:
            global _sct
            with _capture_lock:
                if _sct is None:
                    _sct = mss.MSS()
                shot = _sct.grab(monitor)
            frame = np.array(shot)
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        except ScreenShotError:
            reset_capture_session()
            if attempt == 0:
                continue
            return None
    return None
