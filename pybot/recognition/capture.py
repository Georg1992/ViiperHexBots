"""Screen region capture for template matching."""

from __future__ import annotations

import threading

import cv2
import mss
import numpy as np
from mss.exception import ScreenShotError

_sct: mss.mss | None = None
_capture_lock = threading.Lock()
# Protects the lock/session pair when a reset has to retire a busy capture
# session. Never close or discard a session that another thread may still be
# using; retire it and let that thread close it after releasing its lock.
_capture_state_lock = threading.Lock()
_retired_sessions: dict[int, tuple[threading.Lock, mss.mss | None]] = {}


def _close_capture_session(sct: mss.mss | None) -> None:
    if sct is None:
        return
    try:
        sct.close()
    except Exception:
        pass


def _close_retired_session(lock: threading.Lock) -> None:
    """Close a session whose lock was rotated while a grab was in flight."""
    with _capture_state_lock:
        retired = _retired_sessions.pop(id(lock), None)
    if retired is not None:
        _close_capture_session(retired[1])


def reset_capture_session() -> None:
    """Drop the shared mss session without orphaning a busy native handle.

    A reset can race with a screen grab. If the old lock is busy, rotate the
    lock so new captures can proceed, but retain the old MSS object until its
    owner releases the old lock. The previous implementation discarded that
    reference immediately, leaking the native capture handle and allowing the
    old and new sessions to overlap unsafely.

    Lock ordering is capture lock first, then state lock. The state lock is
    never held while waiting for a capture, so reset cannot deadlock with a
    capture that is validating its lock/session pair.
    """
    global _sct, _capture_lock
    for _ in range(8):
        # Snapshot only the lock identity. Do not hold the state lock while
        # waiting: capture_region() may already hold this lock and then take
        # the state lock to validate the pair.
        with _capture_state_lock:
            lock = _capture_lock

        if lock.acquire(timeout=0.5):
            try:
                with _capture_state_lock:
                    # Another reset may have rotated the pair while this
                    # reset waited. Do not touch a superseded session.
                    if lock is not _capture_lock:
                        continue
                    sct = _sct
                    _sct = None
                _close_capture_session(sct)
                return
            finally:
                lock.release()

        # A grab is still in flight. Retire the session under the state lock;
        # capture_region() closes it after releasing this old lock. New callers
        # receive a clean, independent session.
        with _capture_state_lock:
            if lock is not _capture_lock:
                continue
            sct = _sct
            if sct is not None:
                _retired_sessions[id(lock)] = (lock, sct)
            _sct = None
            _capture_lock = threading.Lock()
            return

    # Another reset won every bounded handoff. Its current session remains the
    # live owner of the shared pair, so do not discard it or spin indefinitely.
    return


def capture_region(x: int, y: int, width: int, height: int) -> np.ndarray | None:
    """Capture a screen rectangle and return a BGR image, or None on failure."""
    if width <= 0 or height <= 0:
        raise ValueError("capture width and height must be positive")

    global _sct
    monitor = {"left": int(x), "top": int(y), "width": int(width), "height": int(height)}
    grab_failures = 0
    pair_retries = 0
    while grab_failures < 2 and pair_retries < 4:
        # Snapshot the lock/session pair while creating a missing session, but
        # do not hold _capture_state_lock while waiting for the capture lock.
        # reset_capture_session() must be able to rotate a lock whose grab is
        # stuck. The local sct reference is essential: after rotation, the
        # old lock must never look up the new global session.
        with _capture_state_lock:
            lock = _capture_lock
            if _sct is None:
                _sct = mss.MSS()
            sct = _sct

        acquired = lock.acquire(timeout=0.5)
        if not acquired:
            # A reset may have retired this busy lock. Retry against the
            # current lock/session pair rather than spending a screenshot
            # retry on the retired one.
            pair_retries += 1
            continue
        try:
            # If reset rotated the pair while we waited, do not use the old
            # session. It will be closed by its owner when that grab finishes.
            with _capture_state_lock:
                if lock is not _capture_lock:
                    pair_retries += 1
                    continue
                sct = _sct
            if sct is None:
                pair_retries += 1
                continue
            shot = sct.grab(monitor)
            frame = np.array(shot)
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        except ScreenShotError:
            grab_failures += 1
            reset_capture_session()
            pair_retries = 0
            continue
        finally:
            lock.release()
            # If reset_capture_session rotated this lock while grab() was in
            # progress, release of the old lock is now the safe close point.
            _close_retired_session(lock)
    return None
