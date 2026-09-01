"""Process-memory observation feed for the main window.

A dedicated thread continuously reads name/SP/weight from the client process
and publishes fresh vitals directly to :class:`PlayerVitals`. Tk callbacks are
best-effort presentation only and never participate in memory production.

This is the same producer/projection split as :class:`StatusPanelFeed`; only
the data source differs. Native reads are cheap, so the value cadence is
faster than status-panel OCR.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from pybot.app.status_display import format_pair
from pybot.app.win32_util import window_exists
from pybot.config.clients import load_client_profile
from pybot.game_state import GameMemoryPoller, MemorySnapshot

# Faster than status-panel OCR (200ms): a native dword read is cheap and
# combat needs a post-skill sample inside the 200ms minimum skill delay.
MEMORY_POLL_MS = 100
# When this producer has no window/profile work, match the OCR search cadence
# so the thread is not a busy loop.
MEMORY_IDLE_MS = 1000
# Avoid flooding the application log when a native read is transiently bad.
MEMORY_LOG_INTERVAL_S = 5.0


@dataclass(frozen=True)
class MemoryReadResult:
    hwnd: int
    state: str
    snap: MemorySnapshot | None = None


class MemoryStatsFeed:
    """Session-owned process-memory producer and best-effort UI projection."""

    def __init__(
        self,
        *,
        config,
        vitals,
        log: Callable[[str], None],
        post_to_tk: Callable[[Callable[[], None]], None],
        on_name: Callable[[str], None],
        on_sp: Callable[[str], None],
        on_weight: Callable[[str], None],
    ) -> None:
        self._post_to_tk = post_to_tk
        self._log = log
        self._config = config
        self._vitals = vitals
        self._on_name = on_name
        self._on_sp = on_sp
        self._on_weight = on_weight
        self._poller = GameMemoryPoller()
        self._last_error_log_at = 0.0
        self._feed_state_lock = threading.RLock()
        self._stopped = False
        self._terminal_fault = False
        self._reader_stop = threading.Event()
        self._reader_thread: threading.Thread | None = None
        self._lifecycle_epoch = 0
        self._ui_result_lock = threading.Lock()
        self._ui_result_pending = False
        self._latest_ui_result: tuple[
            int,
            int | None,
            MemoryReadResult,
        ] | None = None
        self._last_observation_epoch: int | None = getattr(
            vitals, "observation_epoch", None
        )

    # ── Autonomous reader lifecycle ─────────────────────────────────

    def start(self) -> None:
        """Start the one long-lived memory reader, independent of Tk scheduling."""
        with self._feed_state_lock:
            if self._terminal_fault:
                self._log("[UI] Memory reader start ignored after terminal failure")
                return
            self._stopped = False
            self._reader_stop.clear()
            thread = self._reader_thread
            if thread is None or not thread.is_alive():
                thread = threading.Thread(
                    target=self._reader_loop,
                    name="ui-memory-reader",
                    daemon=True,
                )
                self._reader_thread = thread
                thread.start()

    def stop(self) -> None:
        """Stop the reader and invalidate results already queued for Tk."""
        with self._feed_state_lock:
            self._stopped = True
            self._lifecycle_epoch += 1
            self._reader_stop.set()
        with self._ui_result_lock:
            self._latest_ui_result = None
            self._ui_result_pending = False

    def close(self) -> None:
        self.stop()

    @property
    def idle(self) -> bool:
        """True only after the dedicated reader thread has exited."""
        thread = self._reader_thread
        return thread is None or not thread.is_alive()

    @property
    def faulted(self) -> bool:
        """True when a reader/producer failure permanently disabled the feed."""
        with self._feed_state_lock:
            return self._terminal_fault

    def _fail_terminal(self, message: str) -> None:
        """Permanently stop the producer after an unsafe reader failure."""
        with self._feed_state_lock:
            if self._terminal_fault:
                return
            self._terminal_fault = True
            self._stopped = True
            self._lifecycle_epoch += 1
            self._reader_stop.set()
        with self._ui_result_lock:
            self._latest_ui_result = None
            self._ui_result_pending = False
        self._log(f"[UI] Memory reader stopped: {message}")

    def _reader_loop(self) -> None:
        """Continuously read process memory and publish it to ``PlayerVitals``."""
        while not self._reader_stop.is_set():
            with self._feed_state_lock:
                epoch = self._lifecycle_epoch
            observation_epoch = getattr(self._vitals, "observation_epoch", None)
            self._sync_observation_epoch(observation_epoch)
            try:
                result = self._read_snapshot()
            except Exception as exc:
                self._fail_terminal(
                    f"{type(exc).__name__}: {exc}"
                )
                break
            if self._reader_stop.is_set():
                break
            if result is not None:
                # This is the producer boundary. Vitals are committed here,
                # before any optional Tk projection, so consumers never ask
                # the memory thread for a value and a blocked UI cannot stop it.
                try:
                    self._record_reader_result(
                        result,
                        epoch,
                        observation_epoch=observation_epoch,
                    )
                except Exception as exc:
                    self._fail_terminal(
                        f"commit {type(exc).__name__}: {exc}"
                    )
                    break
                current_observation_epoch = getattr(
                    self._vitals, "observation_epoch", None
                )
                if (
                    observation_epoch is not None
                    and current_observation_epoch != observation_epoch
                ):
                    continue
                try:
                    self._queue_ui_result(
                        epoch,
                        result,
                        observation_epoch=observation_epoch,
                    )
                except Exception as exc:
                    self._log(
                        f"[UI] Memory projection disabled: "
                        f"{type(exc).__name__}: {exc}"
                    )
            delay_ms = (
                MEMORY_IDLE_MS
                if result is None or result.state == "inactive"
                else MEMORY_POLL_MS
            )
            self._reader_stop.wait(max(0.05, delay_ms / 1000.0))

    def _sync_observation_epoch(self, observation_epoch: int | None) -> None:
        """Notice a teleport epoch without dropping the cached module base."""
        with self._feed_state_lock:
            if observation_epoch == self._last_observation_epoch:
                return
            self._last_observation_epoch = observation_epoch

    def _read_snapshot(self) -> MemoryReadResult:
        with self._feed_state_lock:
            hwnd = self._config.window_id
            use_memory = bool(self._config.use_memory_reading)
            profile_name = self._config.client_profile
        if not use_memory:
            return MemoryReadResult(hwnd=hwnd or 0, state="inactive")
        profile = load_client_profile(profile_name)
        if profile is None or not profile.memory.has_any:
            return MemoryReadResult(hwnd=hwnd or 0, state="inactive")
        if not hwnd or not window_exists(hwnd):
            return MemoryReadResult(hwnd=hwnd or 0, state="inactive")
        snap = self._poller.read(hwnd, profile.memory)
        if not snap.ok:
            return MemoryReadResult(hwnd=hwnd, state="failed", snap=snap)
        return MemoryReadResult(hwnd=hwnd, state="values", snap=snap)

    def _record_reader_result(
        self,
        result: MemoryReadResult,
        epoch: int,
        *,
        observation_epoch: int | None = None,
    ) -> None:
        """Commit control vitals without touching Tk."""
        with self._feed_state_lock:
            if (
                epoch != self._lifecycle_epoch
                or result.hwnd != self._config.window_id
                or self._stopped
            ):
                return
            if (
                observation_epoch is not None
                and getattr(self._vitals, "observation_epoch", None)
                != observation_epoch
            ):
                return
            if (
                observation_epoch is not None
                and observation_epoch != self._last_observation_epoch
            ):
                return
            if result.state == "values":
                snap = result.snap
                if snap is None:
                    return
                publish_sp = getattr(self._vitals, "publish_sp_if_current", None)
                publish_weight = getattr(
                    self._vitals, "publish_weight_if_current", None
                )
                if observation_epoch is not None and callable(publish_sp):
                    if not publish_sp(snap.sp, snap.sp_max, observation_epoch):
                        return
                else:
                    self._vitals.publish_sp(snap.sp, snap.sp_max)
                if observation_epoch is not None and callable(publish_weight):
                    publish_weight(
                        snap.weight, snap.weight_max, observation_epoch
                    )
                else:
                    self._vitals.publish_weight(snap.weight, snap.weight_max)
                return
            if result.state in {"inactive", "failed"}:
                # Failed or idle frames are not new values. Keep the last
                # successful snapshot intact and retry, matching OCR misses.
                return

    def _queue_ui_result(
        self,
        epoch: int,
        result: MemoryReadResult,
        *,
        observation_epoch: int | None = None,
    ) -> None:
        """Coalesce UI projections so a slow Tk loop never backlogs frames."""
        with self._ui_result_lock:
            self._latest_ui_result = (epoch, observation_epoch, result)
            if self._ui_result_pending:
                return
            self._ui_result_pending = True
        self._post_to_tk(self._consume_ui_result)

    def _consume_ui_result(self) -> None:
        with self._ui_result_lock:
            queued = self._latest_ui_result
            self._latest_ui_result = None
            self._ui_result_pending = False
        if queued is not None:
            epoch, observation_epoch, result = queued
            with self._feed_state_lock:
                current = epoch == self._lifecycle_epoch and not self._stopped
            if (
                current
                and (
                    observation_epoch is None
                    or getattr(self._vitals, "observation_epoch", None)
                    == observation_epoch
                )
            ):
                self._project_result(result)
        with self._ui_result_lock:
            should_repost = (
                self._latest_ui_result is not None
                and not self._ui_result_pending
                and not self._stopped
            )
            if should_repost:
                self._ui_result_pending = True
        if should_repost:
            self._post_to_tk(self._consume_ui_result)

    # ── UI projection ───────────────────────────────────────────────

    def _project_result(self, result: MemoryReadResult) -> None:
        """Render a reader snapshot on Tk without mutating reader state."""
        if result.state == "failed":
            self._log_unsuccessful(
                getattr(result.snap, "error", None) if result.snap else None
            )
            return
        if result.state == "inactive":
            self._on_name("—")
            if self._memory_owns_sp_weight():
                self._on_sp("—")
                self._on_weight("—")
            return
        snap = result.snap
        if snap is None:
            return
        self._on_name(snap.char_name or "—")
        self._on_sp(format_pair(snap.sp, snap.sp_max))
        self._on_weight(format_pair(snap.weight, snap.weight_max))

    def _memory_owns_sp_weight(self) -> bool:
        """True when SP/Weight come from process memory (server profiles)."""
        return bool(self._config.use_memory_reading)

    def _log_unsuccessful(self, error: str | None) -> None:
        now = time.monotonic()
        if now - self._last_error_log_at < MEMORY_LOG_INTERVAL_S:
            return
        self._last_error_log_at = now
        detail = f": {error}" if error else ""
        self._log(
            "[UI] Memory read unsuccessful — retaining last memory state"
            + detail
        )
