"""Session-owned status-panel OCR producer for the main window.

A dedicated thread continuously captures/parses the static Basic Info panel
and publishes fresh vitals directly to :class:`PlayerVitals`. Tk callbacks are
best-effort presentation only and never participate in OCR production.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

from pybot.app.status_display import format_pair
from pybot.app.status_panel_worker import (
    STATUS_PANEL_READ_TIMEOUT_S,
    StatusPanelReadResult,
    read_status_panel_snapshot,
)
from pybot.recognition.ui.status_panel import StatusPanelValues

# Searching for the Basic Info header.
STATUS_PANEL_SEARCH_MS = 1000
# Panel locked — read current SP / Weight only.
STATUS_PANEL_VALUE_MS = 200
# Re-parse max SP / Weight this often while the panel stays locked.
STATUS_PANEL_MAX_REFRESH_S = 1.0
# Avoid flooding the application log when SP/weight OCR is transiently bad.
STATUS_PANEL_HP_ONLY_LOG_S = 5.0
_STATUS_PANEL_WATCHDOG_GRACE_S = 1.0


class StatusPanelFeed:
    """Session-owned status OCR producer and best-effort UI projection."""

    def __init__(
        self,
        *,
        root,
        config,
        vitals,
        overlay,
        log: Callable[[str], None],
        post_to_tk: Callable[[Callable[[], None]], None],
        on_hp: Callable[[str], None],
        on_sp: Callable[[str], None],
        on_weight: Callable[[str], None],
    ) -> None:
        self._post_to_tk = post_to_tk
        self._log = log
        self._config = config
        self._vitals = vitals
        self._status_panel_overlay = overlay
        self._on_hp = on_hp
        self._on_sp = on_sp
        self._on_weight = on_weight
        self._status_panel_confirmed: StatusPanelValues | None = None
        self._status_panel_max_read_at = 0.0
        self._last_hp_only_log_at = 0.0
        # Monotonic time the Basic Info panel was last seen missing, or None.
        self._status_panel_missing_since: float | None = None
        self._last_panel_missing_log_at = 0.0
        # Throttle for the panel-open-but-digits-unreadable diagnostic so a
        # persistently blind feed is visible in the log instead of silent.
        self._last_digits_missing_log_at = 0.0
        self._last_read_timeout_log_at = 0.0
        # Last known Basic Info origin — anchors the open-panel prompt.
        self._status_panel_anchor: tuple[int, int] = (0, 0)
        # Cache client geometry after the first successful read. Re-querying
        # GetClientRect/ClientToScreen on every 200 ms poll is unnecessary and
        # is the native call that can wedge during a teleport transition.
        self._status_panel_client_hint: tuple[int, int, int, int] | None = None
        self._feed_state_lock = threading.RLock()
        self._stopped = False
        self._started = False
        self._ocr_stop = threading.Event()
        self._ocr_thread: threading.Thread | None = None
        self._lifecycle_epoch = 0
        self._ui_result_lock = threading.Lock()
        self._ui_result_pending = False
        self._latest_ui_result: tuple[
            int,
            int | None,
            StatusPanelReadResult,
        ] | None = None
        # PlayerVitals epochs are advanced by teleport transitions to reject
        # stale in-flight publications. The Basic Info panel itself is static
        # for the whole session, so its confirmed anchor and client geometry
        # must survive a danger teleport; changing gameplay state must never
        # force OCR back into an expensive full-panel search.
        self._last_observation_epoch: int | None = getattr(
            vitals, "observation_epoch", None
        )

    # ── Autonomous OCR lifecycle ────────────────────────────────────

    def start(self) -> None:
        """Start the one long-lived OCR reader, independent of Tk scheduling."""
        with self._feed_state_lock:
            self._stopped = False
            self._started = True
            self._ocr_stop.clear()
            thread = self._ocr_thread
            if thread is None or not thread.is_alive():
                thread = threading.Thread(
                    target=self._ocr_loop,
                    name="ui-status-reader",
                    daemon=True,
                )
                self._ocr_thread = thread
                thread.start()

    def stop(self) -> None:
        """Stop the reader and invalidate results already queued for Tk."""
        with self._feed_state_lock:
            self._stopped = True
            self._started = False
            self._lifecycle_epoch += 1
            self._ocr_stop.set()
        with self._ui_result_lock:
            self._latest_ui_result = None
            self._ui_result_pending = False

    def close(self) -> None:
        self.stop()

    @property
    def idle(self) -> bool:
        """True only after the dedicated reader thread has exited."""
        thread = self._ocr_thread
        return thread is None or not thread.is_alive()

    def set_active(self, active: bool) -> None:
        """Compatibility no-op: status OCR is session-owned, never gated.

        The hunt lifecycle may pause combat, sitting, storage, and timers, but
        it must not pause the observation producer. Consumers always read the
        last published values from ``PlayerVitals``.
        """
        return

    def request_now(self) -> None:
        """Compatibility no-op; consumers cannot wake or reconfigure OCR."""
        return

    def _ocr_loop(self) -> None:
        """Continuously read status pixels and publish them to ``PlayerVitals``."""
        while not self._ocr_stop.is_set():
            with self._feed_state_lock:
                epoch = self._lifecycle_epoch
            observation_epoch = getattr(self._vitals, "observation_epoch", None)
            self._sync_observation_epoch(observation_epoch)
            try:
                result = self._read_snapshot()
            except Exception as exc:
                self._log(f"[UI] Status-panel read failed: {exc}")
                result = None
            if self._ocr_stop.is_set():
                break
            if result is not None:
                # This is the producer boundary. Vitals are committed here,
                # before any optional Tk projection, so consumers never ask
                # the OCR thread for a value and a blocked UI cannot stop it.
                try:
                    self._record_reader_result(
                        result,
                        epoch,
                        observation_epoch=observation_epoch,
                    )
                except Exception as exc:
                    self._log(f"[UI] Status-panel result commit failed: {exc}")
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
                    self._log(f"[UI] Status-panel projection queue failed: {exc}")
            with self._feed_state_lock:
                confirmed = self._status_panel_confirmed
            delay_ms = STATUS_PANEL_VALUE_MS if confirmed else STATUS_PANEL_SEARCH_MS
            self._ocr_stop.wait(max(0.05, delay_ms / 1000.0))

    def _sync_observation_epoch(self, observation_epoch: int | None) -> None:
        """Notice a teleport epoch without reconfiguring the static OCR ROI."""
        with self._feed_state_lock:
            if observation_epoch == self._last_observation_epoch:
                return
            self._last_observation_epoch = observation_epoch
            # The panel layout is session-static. Reset only the max refresh
            # clock so the next fixed-ROI read refreshes the maxima; retain the
            # confirmed anchor and client hint. Epoch publication guards reject
            # stale transition frames without making OCR rediscover an unchanged
            # panel.
            self._status_panel_max_read_at = 0.0

    def _read_snapshot(self) -> StatusPanelReadResult:
        with self._feed_state_lock:
            hwnd = self._config.window_id
            confirmed = self._status_panel_confirmed
            now = time.monotonic()
            refresh_max = (
                confirmed is None
                or now - self._status_panel_max_read_at >= STATUS_PANEL_MAX_REFRESH_S
            )
            client_hint = self._status_panel_client_hint
        # This is already the one permanent status-reader thread. Do not add
        # another daemon thread or a process-wide single-flight lock here. The
        # live producer performs one fixed full-value parse and publishes only
        # complete fresh snapshots; no fallback cascade can change its timing.
        try:
            return read_status_panel_snapshot(
                hwnd,
                confirmed,
                refresh_max=refresh_max,
                timeout_s=STATUS_PANEL_READ_TIMEOUT_S,
                client_hint=client_hint,
                refresh_client=False,
                reanchor=False,
                allow_partial=False,
            )
        except Exception as exc:
            # A parser/capture exception must become one ordinary result, not
            # terminate the permanent producer. The next loop simply retries
            # the same static layout; no consumer-driven recovery is needed.
            return StatusPanelReadResult(
                hwnd=hwnd,
                state="read_failed",
                error=f"{type(exc).__name__}: {exc}",
            )

    def _record_reader_result(
        self,
        result: StatusPanelReadResult,
        epoch: int,
        *,
        observation_epoch: int | None = None,
    ) -> None:
        """Commit parser state and control vitals without touching Tk."""
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
            # A new teleport epoch invalidates the confirmed anchor and every
            # cached value. Never commit a result from a read associated with a
            # different producer epoch, even if the caller supplied an object
            # that happens to have the same window handle. Untagged legacy
            # calls remain compatible with the pre-epoch test adapter.
            if (
                observation_epoch is not None
                and observation_epoch != self._last_observation_epoch
            ):
                return
            if result.state in {"values", "hp_only", "sp_only", "hp_sp_only"}:
                self._status_panel_missing_since = None
            if result.state in {
                "values",
                "hp_only",
                "sp_only",
                "hp_sp_only",
            }:
                if result.values is not None:
                    # Do not update the anchor/client cache until the
                    # epoch-guarded vitals publication succeeds. During the
                    # teleport quarantine a transition-frame result can be
                    # rejected by PlayerVitals; caching it here would make the
                    # rejected frame the next fixed-ROI anchor and could
                    # resurrect the previous area's SP on the next sit.
                    publish_snapshot = getattr(
                        self._vitals, "publish_snapshot_if_current", None
                    )
                    if (
                        observation_epoch is not None
                        and callable(publish_snapshot)
                        and self._panel_owns_sp_weight()
                    ):
                        if not publish_snapshot(
                            result.values.hp,
                            result.values.hp_max,
                            result.values.sp,
                            result.values.sp_max,
                            result.values.weight,
                            result.values.weight_max,
                            observation_epoch,
                        ):
                            return
                    else:
                        publish_hp = getattr(
                            self._vitals, "publish_hp_if_current", None
                        )
                        if (
                            observation_epoch is not None
                            and callable(publish_hp)
                        ):
                            if not publish_hp(
                                result.values.hp,
                                result.values.hp_max,
                                observation_epoch,
                            ):
                                return
                        else:
                            self._vitals.publish_hp(
                                result.values.hp, result.values.hp_max
                            )
                        if self._panel_owns_sp_weight():
                            self._vitals.publish_sp(
                                result.values.sp, result.values.sp_max
                            )
                            self._vitals.publish_weight(
                                result.values.weight, result.values.weight_max
                            )
                    # The sample is now accepted. Only accepted frames may
                    # establish the next OCR anchor or refresh geometry hints.
                    self._status_panel_confirmed = result.values
                    self._status_panel_anchor = result.values.panel_origin
                    result_width = getattr(result, "client_width", 0)
                    result_height = getattr(result, "client_height", 0)
                    if result_width > 0 and result_height > 0:
                        self._status_panel_client_hint = (
                            result.client_left,
                            result.client_top,
                            result_width,
                            result_height,
                        )
                    if getattr(result, "full_refresh", False):
                        self._status_panel_max_read_at = time.monotonic()
                hp = getattr(result, "hp", None)
                sp = getattr(result, "sp", None)
                if hp is not None:
                    publish_hp = getattr(
                        self._vitals, "publish_hp_if_current", None
                    )
                    if observation_epoch is not None and callable(publish_hp):
                        if not publish_hp(*hp, observation_epoch):
                            return
                    else:
                        self._vitals.publish_hp(*hp)
                if sp is not None and self._panel_owns_sp_weight():
                    publish_sp = getattr(
                        self._vitals, "publish_sp_if_current", None
                    )
                    if observation_epoch is not None and callable(publish_sp):
                        if not publish_sp(*sp, observation_epoch):
                            return
                    else:
                        self._vitals.publish_sp(*sp)
                return
            if result.state in {
                "inactive",
                "read_failed",
                "panel_missing",
                "roi_missing",
                "read_timeout",
                "panel_open_digits_missing",
                "client_missing",
            }:
                # Failed frames are not new values. Keep the last successful
                # snapshot intact and retry the same session-static layout.
                # No miss streak can trigger a fallback or reconfiguration.
                return

    def _queue_ui_result(
        self,
        epoch: int,
        result: StatusPanelReadResult,
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

    def _project_result(self, result: StatusPanelReadResult) -> None:
        """Render a reader snapshot on Tk without mutating reader state."""
        if result.state == "read_timeout":
            now = time.monotonic()
            if now - self._last_read_timeout_log_at >= STATUS_PANEL_HP_ONLY_LOG_S:
                self._last_read_timeout_log_at = now
                detail = getattr(result, "error", None)
                self._log(
                    "[UI] Status-panel read timed out — retaining last OCR state"
                    + (f": {detail}" if detail else "")
                )
            return
        if result.state in {"inactive", "read_failed", "client_missing"}:
            self._status_panel_overlay.hide()
            self._on_hp("—")
            if self._panel_owns_sp_weight():
                self._on_sp("—")
                self._on_weight("—")
            if result.state == "read_failed":
                detail = getattr(result, "error", None)
                self._log(
                    "[UI] Status-panel read failed"
                    + (f": {detail}" if detail else "")
                )
            return
        if result.state == "panel_missing":
            self._on_hp("—")
            if self._panel_owns_sp_weight():
                self._on_sp("—")
                self._on_weight("—")
            self._status_panel_overlay.show_panel_missing(
                client_left=result.client_left,
                client_top=result.client_top,
                panel_origin=self._status_panel_anchor,
            )
            return
        if result.state in {"hp_only", "sp_only", "hp_sp_only"}:
            hp = getattr(result, "hp", None)
            sp = getattr(result, "sp", None)
            if result.values is not None:
                values = result.values
                self._on_hp(format_pair(values.hp, values.hp_max))
                if self._panel_owns_sp_weight():
                    self._on_sp(format_pair(values.sp, values.sp_max))
                    self._on_weight(format_pair(values.weight, values.weight_max))
            if hp is not None:
                self._on_hp(format_pair(*hp))
            if sp is not None and self._panel_owns_sp_weight():
                self._on_sp(format_pair(*sp))
            return
        if result.state == "panel_open_digits_missing":
            self._log_digits_missing()
            return
        if result.values is not None:
            values = result.values
            self._status_panel_overlay.update(
                values,
                client_left=result.client_left,
                client_top=result.client_top,
            )
            self._on_hp(format_pair(values.hp, values.hp_max))
            if self._panel_owns_sp_weight():
                self._on_sp(format_pair(values.sp, values.sp_max))
                self._on_weight(format_pair(values.weight, values.weight_max))

    def apply_result(self, result: StatusPanelReadResult) -> None:
        """Compatibility API that uses the same passive producer commit path."""
        with self._feed_state_lock:
            epoch = self._lifecycle_epoch
        # Compatibility callers have no capture token. Before any teleport,
        # epoch zero is the original session and remains safe for old tests and
        # adapters. Once a teleport advanced the observation epoch, an untagged
        # completion is ambiguous and must be dropped rather than treated as a
        # fresh SP sample. Production reads use _ocr_loop and carry the token
        # captured before the native read.
        observation_epoch = getattr(self._vitals, "observation_epoch", 0)
        if observation_epoch:
            return
        self._record_reader_result(result, epoch)
        self._project_result(result)

    # ── Panel state helpers ─────────────────────────────────────────

    def reset(self) -> None:
        """Compatibility no-op; the status producer is session-owned."""
        return

    def _panel_owns_sp_weight(self) -> bool:
        """True when SP/Weight come from Basic Info OCR (Generic / no memory)."""
        return not self._config.use_memory_reading

    def _log_digits_missing(self) -> None:
        """Throttled diagnostic for a panel that is open but unreadable.

        The panel header is visible but HP/SP/Weight digits fail to parse — a
        state that publishes nothing and logs nothing. While it persists, the
        danger detector goes blind, so surface it instead of failing silently.
        """
        now = time.monotonic()
        if now - self._last_digits_missing_log_at < STATUS_PANEL_HP_ONLY_LOG_S:
            return
        self._last_digits_missing_log_at = now
        since = ""
        if self._status_panel_missing_since is not None:
            since = (
                f" (panel missing {int(now - self._status_panel_missing_since)}s)"
            )
        self._log(
            "[UI] Status panel open but HP/SP/weight digits unreadable"
            f"{since} — feed paused until OCR recovers"
        )

    def _apply_sp_only_result(self, sp: tuple[int, int] | None) -> None:
        """Publish SP even when HP/Weight OCR failed in the same frame."""
        if sp is None:
            return
        if self._panel_owns_sp_weight():
            self._vitals.publish_sp(*sp)
            self._on_sp(format_pair(*sp))

    def _apply_hp_only_result(self, hp: tuple[int, int] | None) -> None:
        """Publish HP even when another status-panel row failed OCR."""
        if hp is None:
            return
        # An hp_only read means the panel header/origin was found, so a live
        # HP OCR is proof the panel feed is back — end a missing spell even
        # when the SP/weight rows are still unreadable.
        if self._status_panel_missing_since is not None:
            duration = time.monotonic() - self._status_panel_missing_since
            self._status_panel_missing_since = None
            self._log(
                f"[UI] Status panel recovered (HP only) after {int(duration)}s"
            )
        self._vitals.publish_hp(*hp)
        self._on_hp(format_pair(*hp))
        now = time.monotonic()
        if now - self._last_hp_only_log_at >= STATUS_PANEL_HP_ONLY_LOG_S:
            self._last_hp_only_log_at = now
            self._log(
                f"[UI] Status panel HP-only read hp={hp[0]}/{hp[1]}; "
                "SP/weight OCR unavailable"
            )

    def _commit_status_panel(
        self,
        values: StatusPanelValues,
        *,
        client_left: int,
        client_top: int,
        publish_vitals: bool = True,
    ) -> None:
        """Store a successful read; UI stats update only when numbers change."""
        if self._status_panel_missing_since is not None:
            duration = time.monotonic() - self._status_panel_missing_since
            self._status_panel_missing_since = None
            self._log(
                f"[UI] Status panel recovered after {int(duration)}s"
            )
        self._status_panel_confirmed = values
        self._status_panel_anchor = values.panel_origin
        self._status_panel_overlay.update(
            values, client_left=client_left, client_top=client_top
        )
        # HP is vision-only — always mirror into the bot UI from panel OCR.
        self._on_hp(format_pair(values.hp, values.hp_max))
        # Publish HP and SP to vitals every successful OCR tick.
        # HP goes to vitals unconditionally (hunt workers need it for danger).
        if publish_vitals:
            self._vitals.publish_hp(values.hp, values.hp_max)
            if self._panel_owns_sp_weight():
                self._vitals.publish_sp(values.sp, values.sp_max)
                self._vitals.publish_weight(values.weight, values.weight_max)
        # The reader already committed ``previous`` before this UI projection.
        # Projection must still refresh the labels on the first frame; avoid
        # using the reader's state cache as a UI-change detector.
        if self._panel_owns_sp_weight():
            self._apply_status_panel_stats(values)

    def _apply_status_panel_stats(self, values: StatusPanelValues) -> None:
        """Apply vision SP/Weight when memory is off (HP set in commit)."""
        if not self._panel_owns_sp_weight():
            return
        self._on_sp(format_pair(values.sp, values.sp_max))
        self._on_weight(format_pair(values.weight, values.weight_max))
