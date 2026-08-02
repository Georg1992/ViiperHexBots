"""Small non-blocking worker queue used by the Tk UI."""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable


class UiWorkQueue:
    """Run at most one queued task at a time without blocking producers."""

    def __init__(self, *, name: str) -> None:
        self._queue: queue.Queue[Callable[[], None] | None] = queue.Queue(maxsize=1)
        self._closed = False
        self._pending_count = 0
        self._state_lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    def submit(self, task: Callable[[], None]) -> bool:
        """Queue *task* without blocking, replacing stale pending work."""
        with self._state_lock:
            if self._closed:
                return False
            try:
                self._queue.put_nowait(task)
                self._pending_count += 1
                return True
            except queue.Full:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    return False
                self._pending_count -= 1
                try:
                    self._queue.put_nowait(task)
                    self._pending_count += 1
                    return True
                except queue.Full:
                    return False

    @property
    def idle(self) -> bool:
        """True when accepted work is drained and shutdown worker has exited."""
        with self._state_lock:
            if self._pending_count != 0:
                return False
            return not self._closed or not self._thread.is_alive()

    def close(self) -> None:
        """Request worker exit without waiting on the UI thread.

        If a task is already pending, the worker drains it before exiting even
        when the one-slot queue cannot accept the sentinel immediately. This
        matters for configuration writes during application shutdown.
        """
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            # The worker will observe _closed after the pending task and exit
            # once the queue has drained.
            pass

    def _run(self) -> None:
        while True:
            task = self._queue.get()
            if task is None:
                return
            try:
                task()
            except Exception:
                # Callers own error reporting; a UI helper must not die on one
                # failed process read or config write.
                pass
            finally:
                with self._state_lock:
                    self._pending_count -= 1
                    closed = self._closed
            if closed and self._queue.empty():
                return
