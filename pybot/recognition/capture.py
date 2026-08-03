"""Screen region capture for template matching."""

from __future__ import annotations

import queue
import threading
import time

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

# A native ``mss.grab`` can block forever (e.g. the game client wedged on a
# loading frame). Workers must never freeze behind it, so every grab runs on a
# single daemon worker and callers wait at most ``_CAPTURE_GRAB_TIMEOUT_S``.
# A permanently stuck grab orphaned the worker until the client recovers;
# queued grabs then drain and captures resume. The daemon thread cannot block
# process shutdown.
_CAPTURE_GRAB_TIMEOUT_S = 2.0
_grab_queue: "queue.Queue[_GrabRequest | None]" = queue.Queue(maxsize=16)
_grab_worker_started = False
_grab_state_lock = threading.Lock()


class _GrabRequest:
    """One native grab with a completion event and result slot."""

    __slots__ = ("monitor", "sct", "done", "result", "error")

    def __init__(self, monitor: dict, sct: mss.mss) -> None:
        self.monitor = monitor
        self.sct = sct
        self.done = threading.Event()
        self.result = None
        self.error: BaseException | None = None


def _grab_worker_loop() -> None:
    while True:
        request = _grab_queue.get()
        if request is None:
            return
        try:
            request.result = request.sct.grab(request.monitor)
        except BaseException as exc:  # noqa: BLE001 - surface any native failure
            request.error = exc
        finally:
            request.done.set()
            # The worker is the only thread that ever touches this session.
            # If reset rotated the pair while this grab was in flight, the
            # waiting caller may have timed out and returned; closing the
            # retired session here (after grab completes) is the only safe
            # close point. Callers must never close a session this thread
            # is still using.
            _close_retired_session_for_sct(request.sct)


def _ensure_grab_worker() -> None:
    global _grab_worker_started
    with _grab_state_lock:
        if _grab_worker_started:
            return
        _grab_worker_started = True
    threading.Thread(
        target=_grab_worker_loop,
        name="mss-grab",
        daemon=True,
    ).start()


def _close_capture_session(sct: mss.mss | None) -> None:
    if sct is None:
        return
    try:
        sct.close()
    except Exception:
        pass


def _close_retired_session(lock: threading.Lock) -> None:
    """Close a session whose lock was rotated while a grab was in flight.

    Safe only when the caller's grab finished (or was never queued): the
    caller thread no longer owns any in-flight grab on that session.
    """
    with _capture_state_lock:
        retired = _retired_sessions.pop(id(lock), None)
    if retired is not None:
        _close_capture_session(retired[1])


def _close_retired_session_for_sct(sct: mss.mss) -> None:
    """Close a retired session from the worker thread that owns it.

    A session is retired when reset rotates the pair while a grab is in
    flight. Only the thread that ran the grab may close it afterwards;
    closing from the waiting caller could destroy the native handle mid-grab.
    """
    retired = None
    with _capture_state_lock:
        for key, (_lock, retired_sct) in list(_retired_sessions.items()):
            if retired_sct is sct:
                retired = _retired_sessions.pop(key)
                break
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
        reset_needed = False
        timed_out = False
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
            _ensure_grab_worker()
            request = _GrabRequest(monitor, sct)
            try:
                _grab_queue.put_nowait(request)
            except queue.Full:
                # A wedged grab has backed the queue up. Fail the frame
                # instead of blocking; the backlog drains once grabs recover.
                return None
            if not request.done.wait(_CAPTURE_GRAB_TIMEOUT_S):
                # The native grab hung (client/desktop wedge). Do not freeze
                # the calling worker: fail this frame. The request stays owned
                # by the daemon worker; when the client recovers, queued grabs
                # drain and captures resume. The next grab after a recovery
                # re-validates the session and retries.
                timed_out = True
                return None
            if request.error is not None:
                raise request.error
            shot = request.result
            frame = np.array(shot)
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        except ScreenShotError:
            grab_failures += 1
            pair_retries = 0
            reset_needed = True
        finally:
            lock.release()
            # If reset_capture_session rotated this lock while grab() was in
            # progress, release of the old lock is the safe close point — but
            # only when this grab finished (or was never queued). On a timeout
            # the worker may still be inside grab() for this session, so it
            # closes the retired session after the grab completes.
            if not timed_out:
                _close_retired_session(lock)
        # Replace the broken session only after this capture released its
        # lock: reset_capture_session() must acquire the lock to close the
        # session it rotates out.
        if reset_needed:
            reset_capture_session()
    return None
