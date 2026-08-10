"""Deterministic Tk-scheduled background-read task.

The process-memory feed uses one owned worker queue. A native operation is
never replaced or retried behind the original operation: a timeout is a
terminal feed fault that is reported explicitly. The status-panel feed owns
its own production reader loop and does not use this runner.
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable

from pybot.app.ui_work_queue import UiWorkQueue


class PeriodicTaskRunner:
    """Owns one periodic producer and its single worker lifecycle.

    Subclasses implement :meth:`should_submit`, :meth:`build_job`, and
    :meth:`apply_result`. A worker exception or a native timeout is reported
    through the failure hooks and permanently disables this feed; there is no
    hidden retry or replacement worker.
    """

    def __init__(
        self,
        *,
        root,
        name: str,
        timeout_s: float,
        default_delay_ms: int,
        post_to_tk: Callable[[Callable[[], None]], None],
        log: Callable[[str], None],
    ) -> None:
        self._root = root
        self._name = name
        self._timeout_s = timeout_s
        self._default_delay_ms = default_delay_ms
        self._post_to_tk = post_to_tk
        self._log = log
        self._after_id = None
        self._stopped = False
        self._pending = False
        self._pending_generation: int | None = None
        self._generation = 0
        self._started_at = 0.0
        self._terminal_fault = False
        self._results: queue.Queue = queue.Queue(maxsize=1)
        # Publication happens on worker threads while consumption happens on
        # Tk. Keep the single-slot queue from allowing an older result to
        # evict a newer one after reset/restart.
        self._result_lock = threading.Lock()
        self._work = UiWorkQueue(
            name=name,
            on_error=self._on_worker_error,
        )

    # ── Subclass hooks ──────────────────────────────────────────────

    def pending_delay(self) -> int:
        """Next poll delay while a read is still in flight."""
        return self._default_delay_ms

    def should_submit(self) -> int | None:
        """Return the next poll delay when a read may start, else ``None``."""
        raise NotImplementedError

    def build_job(self, generation: int) -> Callable[[], None] | None:
        """Return the worker callable for one read, or ``None`` to skip."""
        raise NotImplementedError

    def apply_result(self, result) -> None:
        """Apply one successful read result on the Tk thread."""
        raise NotImplementedError

    def on_failure(self, exc: Exception, generation: int) -> None:
        """Handle a worker exception for a still-current generation."""

    def on_terminal_failure(self, message: str) -> None:
        """Report a terminal producer failure; subclasses may add context."""
        self._log(f"[UI] {self._name} stopped: {message}")

    def _on_worker_error(self, exc: Exception) -> None:
        """Stop the producer when its worker catches an uncategorized error."""
        self._fail_terminal(
            f"worker exception {type(exc).__name__}: {exc}"
        )

    @property
    def faulted(self) -> bool:
        """True when this producer stopped because it cannot continue safely."""
        return self._terminal_fault

    @property
    def worker_alive(self) -> bool:
        """True while the owned queue worker is still executing or draining."""
        thread = getattr(self._work, "_thread", None)
        return bool(thread is not None and thread.is_alive())

    @property
    def shutdown_safe(self) -> bool:
        """True when the UI may close without accepting more feed callbacks.

        A faulted native call may still be alive on the daemon worker. That is
        deliberately reported by :attr:`worker_alive`; this property means
        only that the producer is permanently disabled and cannot touch Tk.
        """
        return self._terminal_fault or self.idle

    # ── Public API ──────────────────────────────────────────────────

    def start(self) -> None:
        """Begin the periodic poll loop."""
        if self._terminal_fault:
            self.on_terminal_failure("start requested after terminal failure")
            return
        self._stopped = False
        self._schedule_next()

    def stop(self) -> None:
        """Cancel polling and invalidate every result from the old lifecycle."""
        with self._result_lock:
            self._stopped = True
            self._generation += 1
            self._pending = False
            self._pending_generation = None
            self._started_at = 0.0
        if self._after_id is not None:
            try:
                self._root.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None

    def close(self) -> None:
        """Stop polling and release the feed's single worker at app shutdown."""
        self.stop()
        self._work.close()

    @property
    def idle(self) -> bool:
        """True when the worker queue has drained."""
        # A terminal fault does not make a live native worker idle. The caller
        # must see the real ownership state and keep shutdown ownership until
        # that worker exits; Python cannot honestly claim the resource is gone.
        return self._work.idle

    def reset(self) -> None:
        """Invalidate an in-flight read before a new configuration is used."""
        with self._result_lock:
            if self._terminal_fault:
                return
            self._generation += 1
            if not self._pending:
                self._pending_generation = None
                self._started_at = 0.0

    def request_now(self) -> None:
        """Run one immediate submit cycle (window/profile changed)."""
        self.request()

    def publish(self, generation: int, result) -> None:
        """Worker thread: deliver one successful read result.

        A reset can leave an older native read running while a newer request is
        already in flight. Never let that stale result replace the newer one in
        the bounded queue; doing so would leave the current request pending
        until the stall watchdog fires.
        """
        with self._result_lock:
            if generation != self._generation or self._stopped or self._terminal_fault:
                return
            try:
                self._results.put_nowait((generation, result))
            except queue.Full:
                try:
                    existing_generation, existing_result = self._results.get_nowait()
                except queue.Empty:
                    existing_generation = generation
                    existing_result = None
                if existing_generation > generation:
                    try:
                        self._results.put_nowait(
                            (existing_generation, existing_result)
                        )
                    except queue.Full:
                        pass
                    return
                try:
                    self._results.put_nowait((generation, result))
                except queue.Full:
                    pass
        self._post_to_tk(self.consume_results)

    def fail(self, generation: int, exc: Exception) -> None:
        """Worker thread: deliver one read failure."""
        self._post_to_tk(lambda: self._handle_failure(generation, exc))

    # ── Internals ───────────────────────────────────────────────────

    def request(self) -> int:
        """One poll cycle; returns the next poll delay in milliseconds.

        A completed result is drained first so the same tick can start the
        next read (the feeds' original consume-then-submit contract).
        """
        self.consume_results()
        if self._terminal_fault:
            return self._default_delay_ms
        if self._pending:
            # A native call cannot be cancelled safely from Python. Do not
            # clear the flag and retry: that would hide the real fault and
            # create competing work. Stop this producer permanently and report
            # the exact terminal condition.
            if (
                self._started_at
                and time.monotonic() - self._started_at >= self._timeout_s
            ):
                self._fail_terminal(
                    f"native read exceeded {self._timeout_s:.1f}s"
                )
            return self.pending_delay()
        delay = self.should_submit()
        if delay is None:
            # The feed declined a read this tick (no window/profile/panel).
            return self._default_delay_ms
        try:
            job = self.build_job(self._generation + 1)
        except Exception as exc:
            self._fail_terminal(
                f"read job construction failed with {type(exc).__name__}: {exc}"
            )
            return delay
        if job is None:
            return delay
        with self._result_lock:
            self._pending = True
            self._generation += 1
            generation = self._generation
            self._pending_generation = generation
            self._started_at = time.monotonic()

        def run_job() -> None:
            try:
                job()
            finally:
                self._worker_finished(generation)

        if not self._work.submit(run_job):
            self._fail_terminal("worker queue rejected the read")
        return delay

    def consume_results(self) -> None:
        """Apply any completed result on the Tk thread."""
        with self._result_lock:
            try:
                generation, result = self._results.get_nowait()
            except queue.Empty:
                return
        if (
            generation != self._generation
            or self._stopped
            or self._terminal_fault
        ):
            # A newer request owns the pending flag now. Drop the stale
            # result without releasing or overwriting the newer request. An
            # inactive feed also drops a completion from before pause/stop.
            return
        self._pending = False
        self._pending_generation = None
        self._started_at = 0.0
        self.apply_result(result)

    def _handle_failure(self, generation: int, exc: Exception) -> None:
        if generation != self._generation:
            return
        self._pending = False
        self._pending_generation = None
        self._started_at = 0.0
        try:
            self.on_failure(exc, generation)
        finally:
            self._fail_terminal(
                f"read failed with {type(exc).__name__}: {exc}"
            )

    def _fail_terminal(self, message: str) -> None:
        """Stop this producer after an unrecoverable operation failure.

        The current worker is never replaced. Its late completion is rejected
        by the generation token, while the feed remains explicitly faulted and
        never schedules another native operation.
        """
        with self._result_lock:
            if self._terminal_fault:
                return
            self._terminal_fault = True
            self._stopped = True
            self._generation += 1
            self._pending = False
            self._pending_generation = None
            self._started_at = 0.0
        self._work.discard_pending()
        self.on_terminal_failure(message)

    def _worker_finished(self, generation: int) -> None:
        """Release a reset generation only after its queued native work exits."""
        with self._result_lock:
            if (
                self._pending
                and self._pending_generation == generation
                and generation != self._generation
            ):
                self._pending = False
                self._pending_generation = None
                self._started_at = 0.0

    def _schedule_next(self) -> None:
        if self._stopped:
            return

        def _tick() -> None:
            self._after_id = None
            delay = self._default_delay_ms
            try:
                delay = self.request()
            finally:
                if not self._stopped:
                    try:
                        self._after_id = self._root.after(
                            max(50, int(delay)), _tick
                        )
                    except Exception:
                        self._after_id = None

        self._after_id = self._root.after(self._default_delay_ms, _tick)
