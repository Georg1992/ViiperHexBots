"""Screen region capture for template matching."""

from __future__ import annotations

import queue
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


def _grab_worker_loop(
    work_queue: "queue.Queue[_GrabRequest | None]",
) -> None:
    while True:
        request = work_queue.get()
        if request is None:
            return
        try:
            request.result = request.sct.grab(request.monitor)
        except BaseException as exc:  # noqa: BLE001 - surface any native failure
            request.error = exc
        finally:
            request.done.set()
            # The worker is the only thread that ever touches this session.
            # If reset or a timeout rotated the pair while this grab was in
            # flight, close the old native handle only after grab() returns.
            _close_retired_session_for_sct(request.sct)


def _ensure_grab_worker(
    work_queue: "queue.Queue[_GrabRequest | None]",
) -> None:
    global _grab_worker_started
    with _grab_state_lock:
        if _grab_worker_started or work_queue is not _grab_queue:
            return
        _grab_worker_started = True
    threading.Thread(
        target=_grab_worker_loop,
        args=(work_queue,),
        name="mss-grab",
        daemon=True,
    ).start()


def _retire_runtime_capture(
    sct: mss.mss,
    work_queue: "queue.Queue[_GrabRequest | None]",
) -> None:
    """Rotate the shared runtime capture after a native grab timeout.

    Discovery and coordinate tracking both use this channel. Leaving the old
    queue in place after one blocked grab makes every subsequent observer wait
    behind the same native call, which is exactly the tracking freeze seen in
    the log. The old daemon worker is left to finish on its private queue;
    future captures get a new session and worker immediately.
    """
    global _sct, _capture_lock, _grab_queue, _grab_worker_started
    with _capture_state_lock:
        if _sct is not sct or _grab_queue is not work_queue:
            return
        # The worker may still be inside ``grab``. Retain the native handle
        # until that same worker returns and calls
        # ``_close_retired_session_for_sct``; dropping this reference here
        # leaks one MSS/GDI session on every timed-out runtime capture.
        # Rotate the lock as well as the queue. Without this, a second timeout
        # would reuse the same lock identity and overwrite the first retired
        # session entry before its worker could close it.
        old_lock = _capture_lock
        _retired_sessions[id(old_lock)] = (old_lock, sct)
        _capture_lock = threading.Lock()
        _sct = None
        _grab_queue = queue.Queue(maxsize=16)
        with _grab_state_lock:
            _grab_worker_started = False
        try:
            work_queue.put_nowait(None)
        except queue.Full:
            pass


def _close_capture_session(sct: mss.mss | None) -> bool:
    """Close a native capture session and report whether cleanup succeeded."""
    if sct is None:
        return True
    try:
        sct.close()
    except Exception:
        return False
    return True


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


class _UiCaptureChannel:
    """Dedicated capture channel for the status-panel OCR (UI reads).

    The runtime pipeline (discovery / tracking / teleport scans) funnels every
    grab through one shared mss session and a single daemon worker. The status
    panel OCR used to share that same worker, so a slow or wedged UI capture
    could back up the runtime queue (and vice versa) — the multi-second
    discovery stalls seen in the wild. This channel owns its own mss session,
    lock, queue and worker, so UI reads can never starve the runtime pipeline.

    Resilience mirrors the runtime path: grabs run on one daemon worker with a
    bounded queue and a 2s caller timeout, and a broken session is dropped and
    recreated on ScreenShotError.

    The channel's mss session is created inside the worker thread, never in the
    caller. ``mss.MSS()`` itself can block just like ``grab()`` when the GDI
    desktop is wedged (observed after a seated danger teleport): creating it on
    the caller would leave the status read hanging with no bound, which is how
    a single loading transition used to produce repeated multi-second OCR
    timeouts. A stuck creation now only blocks this daemon worker; the caller
    always waits at most ``_CAPTURE_GRAB_TIMEOUT_S`` and fails closed until the
    retired worker actually exits. The next read starts a fresh worker only
    after that native operation and session cleanup complete.
    """

    def __init__(self) -> None:
        self._state_lock = threading.Lock()
        # Kept as a descriptive compatibility alias for tests/integrations
        # that inspect the channel's session serialization primitive.
        self._session_lock = self._state_lock
        self._queue: "queue.Queue[_GrabRequest | None]" = queue.Queue(maxsize=16)
        self._worker_started = False
        # A timed-out native grab cannot be cancelled. Retiring its queue and
        # immediately starting a replacement would only multiply the blocked
        # mss/OpenCV work after a loading transition. Keep queue identities here
        # until the old worker has returned from the native call and closed its
        # session; captures fail closed during that interval.
        self._retired_queues: set[int] = set()

    def _mark_retired_worker_finished(
        self,
        work_queue: "queue.Queue[_GrabRequest | None]",
    ) -> None:
        """Allow replacement captures after an abandoned worker fully exits."""
        with self._state_lock:
            self._retired_queues.discard(id(work_queue))

    def _ensure_worker(
        self, work_queue: "queue.Queue[_GrabRequest | None]"
    ) -> None:
        with self._state_lock:
            if self._worker_started or work_queue is not self._queue:
                return
            self._worker_started = True
        threading.Thread(
            target=self._grab_worker_loop,
            args=(work_queue,),
            name="ui-mss-grab",
            daemon=True,
        ).start()

    def _grab_worker_loop(
        self, work_queue: "queue.Queue[_GrabRequest | None]"
    ) -> None:
        # The worker owns the channel's mss session. It is created lazily on
        # the first request and closed when the worker exits via the ``None``
        # sentinel, so a wedged desktop can only block this daemon thread.
        sct: mss.mss | None = None
        try:
            while True:
                request = work_queue.get()
                if request is None:
                    return
                try:
                    if sct is None:
                        sct = mss.MSS()
                    request.result = sct.grab(request.monitor)
                except BaseException as exc:  # noqa: BLE001 - surface any native failure
                    request.error = exc
                finally:
                    request.done.set()
        finally:
            # Only this worker ever touched the session, so it is safe to
            # close here once the loop exits. A retired queue is cleared only
            # after the native call has returned and its session is closed;
            # if native cleanup fails, remain fail-closed rather than starting
            # a replacement session beside a possibly-live old handle.
            closed = _close_capture_session(sct)
            if closed:
                self._mark_retired_worker_finished(work_queue)

    def _retire_timed_out_grab(
        self,
        work_queue: "queue.Queue[_GrabRequest | None]",
        *,
        unresolved: bool = True,
    ) -> None:
        """Retire a wedged UI queue without starting a replacement native grab.

        Recreating only ``UiWorkQueue`` is insufficient: its replacement calls
        the same UI capture singleton, whose native worker and bounded queue
        remain blocked forever. Retire the channel's queue/worker pair, but
        keep this channel fail-closed until the old worker has returned from
        its native operation and closed the old session. Discovery/tracking
        channels have separate instances and are unaffected.
        """
        with self._state_lock:
            if work_queue is not self._queue:
                return
            if unresolved:
                self._retired_queues.add(id(work_queue))
            self._queue = queue.Queue(maxsize=16)
            self._worker_started = False
            # Discard requests queued behind the in-flight native operation.
            # Their callers already have bounded waits; allowing the retired
            # worker to process stale captures would delay its exit and could
            # leave the barrier stuck behind a full queue.
            while True:
                try:
                    queued = work_queue.get_nowait()
                except queue.Empty:
                    break
                if queued is not None:
                    queued.error = RuntimeError("retired UI capture request")
                    queued.done.set()
            # The queue is now guaranteed to have room for the exit sentinel.
            work_queue.put_nowait(None)

    def capture(self, x: int, y: int, width: int, height: int) -> np.ndarray | None:
        """Capture one screen rectangle on this channel's private worker."""
        if width <= 0 or height <= 0:
            raise ValueError("capture width and height must be positive")
        monitor = {"left": int(x), "top": int(y), "width": int(width), "height": int(height)}
        grab_failures = 0
        while grab_failures < 2:
            with self._state_lock:
                if self._retired_queues:
                    # The previous native operation is still unresolved. Do
                    # not create another mss worker/session that can compete
                    # with discovery and tracking; the old worker will clear
                    # this barrier after it returns and closes its handle.
                    return None
                work_queue = self._queue
                start_worker = not self._worker_started
            if start_worker:
                self._ensure_worker(work_queue)
            # The session is created inside the worker, so this caller thread
            # is never exposed to a blocking mss.MSS()/grab call: the wait
            # below is the only place it can pause, and it is bounded.
            # ``request.sct`` is deliberately unused here: the UI worker owns
            # its session and ignores that slot (the runtime worker needs it).
            request = _GrabRequest(monitor, None)
            # Recheck the queue identity while holding the state lock before
            # enqueueing. A timeout can retire this queue between the initial
            # snapshot and this request; never add new work to a retired queue.
            retire_full = False
            with self._state_lock:
                if (
                    work_queue is not self._queue
                    or self._retired_queues
                ):
                    return None
                try:
                    work_queue.put_nowait(request)
                except queue.Full:
                    retire_full = True
            if retire_full:
                # Rotate outside the state lock; retirement acquires that lock
                # to atomically record the queue identity.
                self._retire_timed_out_grab(work_queue)
                return None
            if not request.done.wait(_CAPTURE_GRAB_TIMEOUT_S):
                # Native grab or session creation hung (client/desktop wedge).
                # Retire this entire channel pair, not just the caller's OCR
                # task; otherwise every recreated UI worker queues behind the
                # same blocked call.
                self._retire_timed_out_grab(work_queue)
                return None
            if request.error is not None:
                if not isinstance(request.error, ScreenShotError):
                    # A request discarded while its queue was retired is an
                    # expected bounded capture failure for that caller, not a
                    # worker exception to propagate into the OCR feed.
                    if isinstance(request.error, RuntimeError) and str(
                        request.error
                    ) == "retired UI capture request":
                        return None
                    raise request.error
                # Broken session — rotate the channel so the next attempt
                # starts a fresh worker with a fresh mss session.
                grab_failures += 1
                # The native call has already returned with a normal
                # screenshot error, so this is not an unresolved grab. Rotate
                # the worker/session for the next attempt without applying the
                # timeout barrier used for a still-running native call.
                self._retire_timed_out_grab(work_queue, unresolved=False)
                continue
            shot = request.result
            frame = np.array(shot)
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        return None


_ui_capture_channel = _UiCaptureChannel()
# Discovery and local tracking are both observation pipelines, but they must
# not queue behind one another. Each observer owns an independent native grab
# worker/session; a wedged discovery frame therefore cannot starve tracking.
_observation_capture_channels: dict[str, _UiCaptureChannel] = {
    "discovery": _UiCaptureChannel(),
    "tracking": _UiCaptureChannel(),
}


def ui_capture_region(x: int, y: int, width: int, height: int) -> np.ndarray | None:
    """Capture a screen rectangle for UI OCR on its isolated channel."""
    return _ui_capture_channel.capture(x, y, width, height)


def observation_capture_region(
    x: int,
    y: int,
    width: int,
    height: int,
    *,
    observer: str,
) -> np.ndarray | None:
    """Capture for one runtime observer without sharing another observer's grab.

    ``observer`` is deliberately explicit: discovery and tracking have
    different freshness requirements and must never serialize behind the same
    native mss request. Unknown names fail closed instead of silently falling
    back to the shared legacy channel.
    """
    channel = _observation_capture_channels.get(observer)
    if channel is None:
        raise ValueError(f"unknown observation capture channel: {observer!r}")
    return channel.capture(x, y, width, height)


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
            work_queue = _grab_queue

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
            _ensure_grab_worker(work_queue)
            request = _GrabRequest(monitor, sct)
            try:
                work_queue.put_nowait(request)
            except queue.Full:
                # A wedged grab has backed the queue up. Fail the frame
                # instead of blocking; the backlog drains once grabs recover.
                return None
            if not request.done.wait(_CAPTURE_GRAB_TIMEOUT_S):
                # The native grab hung (client/desktop wedge). Do not freeze
                # the calling worker: fail this frame and rotate the shared
                # runtime channel so discovery and tracking can recover.
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
            if timed_out:
                _retire_runtime_capture(sct, work_queue)
            else:
                _close_retired_session(lock)
        # Replace the broken session only after this capture released its
        # lock: reset_capture_session() must acquire the lock to close the
        # session it rotates out.
        if reset_needed:
            reset_capture_session()
    return None
