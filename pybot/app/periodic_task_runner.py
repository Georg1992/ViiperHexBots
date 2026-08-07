"""Generic periodic background-read task with bounded queue + stall recovery.

MainWindow runs two observation feeds (process-memory reads and status-panel
OCR). Both share the same request/pending/generation/result-queue/stall-
watchdog machinery. :class:`PeriodicTaskRunner` owns that machinery once;
each feed subclasses it and supplies only its own read condition, worker
job, and result handling.

Thread contract
---------------
Polling and result application run on the Tk thread (``root.after``).
Worker jobs run on an internal :class:`UiWorkQueue` thread and hand results
back through :meth:`publish` / :meth:`fail`, which schedule the Tk-side
:meth:`consume_results` via the injected ``post_to_tk`` callback — never
``root.after`` from a worker thread.
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable

from pybot.app.ui_work_queue import UiWorkQueue


class PeriodicTaskRunner:
    """Owns the poll / request / result / stall-recovery cycle for one feed.

    Subclasses implement :meth:`should_submit`, :meth:`build_job`, and
    :meth:`apply_result`; :meth:`on_failure`, :meth:`on_recover`, and
    :meth:`pending_delay` may be overridden to specialise diagnostics.
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
        self._started = False
        self._active = True
        self._pending = False
        self._generation = 0
        self._started_at = 0.0
        self._stall_count = 0
        self._results: queue.Queue = queue.Queue(maxsize=1)
        # Publication happens on worker threads while consumption happens on
        # Tk. Keep the single-slot queue from allowing an older result to
        # evict a newer one after reset/restart.
        self._result_lock = threading.Lock()
        self._work = UiWorkQueue(name=name)

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

    def on_recover(self, stall_count: int) -> None:
        """Diagnose a stalled read that was abandoned and restarted."""
        self._log(
            f"[UI] {self._name} read stalled — restarted reader "
            f"(stall #{stall_count})"
        )

    # ── Public API ──────────────────────────────────────────────────

    def start(self) -> None:
        """Begin the periodic poll loop."""
        self._stopped = False
        self._started = True
        self._schedule_next()

    def stop(self) -> None:
        """Cancel the periodic poll loop."""
        self._stopped = True
        self._started = False
        if self._after_id is not None:
            try:
                self._root.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None

    def close(self) -> None:
        """Stop polling and release the worker thread at app shutdown."""
        self.stop()
        self._work.close()

    @property
    def idle(self) -> bool:
        """True when the worker queue has drained."""
        return self._work.idle

    def reset(self) -> None:
        """Drop any in-flight read so the next submit is a fresh one."""
        with self._result_lock:
            self._generation += 1
            self._pending = False
            self._started_at = 0.0

    def set_active(self, active: bool) -> None:
        """Enable reads only for the lifecycle states that own this feed.

        Disabling invalidates the current generation, so a read that was
        already in native code cannot publish values after pause/stop. The
        periodic Tk tick remains installed; it simply declines submissions
        until the feed is enabled again.
        """
        active = bool(active)
        if active == self._active:
            return
        self._active = active
        self.reset()
        if active and self._started:
            self.request_now()

    @property
    def active(self) -> bool:
        """Whether this feed is currently allowed to start/apply reads."""
        return self._active

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
            if generation != self._generation:
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
        if not self._active:
            return self._default_delay_ms
        if self._pending:
            # A wedged read must not pin the pending flag forever: after the
            # stall timeout the worker is abandoned and recreated so the next
            # tick retries on a fresh thread.
            if (
                self._started_at
                and time.monotonic() - self._started_at >= self._timeout_s
            ):
                self._recover_stall()
            return self.pending_delay()
        delay = self.should_submit()
        if delay is None:
            # The feed declined a read this tick (no window/profile/panel).
            return self._default_delay_ms
        job = self.build_job(self._generation + 1)
        if job is None:
            return delay
        with self._result_lock:
            self._pending = True
            self._generation += 1
            self._started_at = time.monotonic()
        if not self._work.submit(job):
            self._pending = False
            self._started_at = 0.0
        return delay

    def consume_results(self) -> None:
        """Apply any completed result on the Tk thread."""
        with self._result_lock:
            try:
                generation, result = self._results.get_nowait()
            except queue.Empty:
                return
        if generation != self._generation or not self._active:
            # A newer request owns the pending flag now. Drop the stale
            # result without releasing or overwriting the newer request. An
            # inactive feed also drops a completion from before pause/stop.
            return
        self._pending = False
        self._stall_count = 0
        self.apply_result(result)

    def _handle_failure(self, generation: int, exc: Exception) -> None:
        if generation != self._generation:
            return
        self._pending = False
        self.on_failure(exc, generation)

    def _recover_stall(self) -> None:
        """Abandon a stalled read and restart on a fresh worker thread."""
        # Invalidate the abandoned worker before replacing its queue. Native
        # reads may still return later; they must never publish as the retry.
        with self._result_lock:
            self._generation += 1
            self._pending = False
            self._started_at = 0.0
        self._work.close()
        self._work = UiWorkQueue(name=self._name)
        self._stall_count += 1
        self.on_recover(self._stall_count)

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
