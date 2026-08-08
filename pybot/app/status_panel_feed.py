"""Status-panel OCR observation feed for the main window.

Periodically OCRs the client's Basic Info panel and publishes HP/SP/weight
into the shared :class:`PlayerVitals` plus the bot UI labels. The capture/OCR
runs on the internal worker thread; all label/overlay/vitals updates happen
on the Tk thread via the runner's result callback.

The result state machine (panel missing, HP-only, digits unreadable, full
refresh, ...) lives here so MainWindow only wires the feed.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

from pybot.app.periodic_task_runner import PeriodicTaskRunner
from pybot.app.status_display import format_pair
from pybot.app.status_panel_worker import (
    STATUS_PANEL_READ_TIMEOUT_S,
    StatusPanelReadResult,
    read_status_panel_snapshot,
    read_status_panel_snapshot_bounded,
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


class StatusPanelFeed(PeriodicTaskRunner):
    """One observation feed: periodic Basic Info OCR reads."""

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
        super().__init__(
            root=root,
            name="ui-status-reader",
            timeout_s=STATUS_PANEL_READ_TIMEOUT_S + _STATUS_PANEL_WATCHDOG_GRACE_S,
            default_delay_ms=STATUS_PANEL_SEARCH_MS,
            post_to_tk=post_to_tk,
            log=log,
            use_work_queue=False,
        )
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
        self._status_panel_geometry_refresh = False
        self._status_panel_reanchor = False
        self._status_panel_miss_count = 0
        self._feed_state_lock = threading.RLock()
        self._ocr_stop = threading.Event()
        self._ocr_wake = threading.Event()
        self._ocr_thread: threading.Thread | None = None
        self._lifecycle_epoch = 0
        self._ui_result_lock = threading.Lock()
        self._ui_result_pending = False
        self._latest_ui_result: tuple[int, StatusPanelReadResult] | None = None

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
        self._ocr_wake.set()

    def stop(self) -> None:
        """Stop the reader and invalidate results already queued for Tk."""
        with self._feed_state_lock:
            self._stopped = True
            self._started = False
            self._lifecycle_epoch += 1
            self._ocr_stop.set()
        with self._result_lock:
            self._generation += 1
            self._pending = False
            self._started_at = 0.0
        with self._ui_result_lock:
            self._latest_ui_result = None
            self._ui_result_pending = False
        self._ocr_wake.set()

    def close(self) -> None:
        self.stop()

    @property
    def idle(self) -> bool:
        """True only after the dedicated reader thread has exited."""
        thread = self._ocr_thread
        return thread is None or not thread.is_alive()

    def set_active(self, active: bool) -> None:
        active = bool(active)
        with self._feed_state_lock:
            if active == self._active:
                return
            self._active = active
        self.reset()
        self._ocr_wake.set()

    def request_now(self) -> None:
        """Wake the reader after a window/profile change."""
        self._ocr_wake.set()

    def _ocr_loop(self) -> None:
        """Read and publish vitals continuously; Tk only receives presentation."""
        while not self._ocr_stop.is_set():
            # Consume a wake at the start of a cycle. A reset/window change
            # arriving while the native read is running remains set until the
            # post-read check below and therefore cannot be lost.
            self._ocr_wake.clear()
            with self._feed_state_lock:
                active = self._active
                epoch = self._lifecycle_epoch
            if not active:
                self._ocr_stop.wait(0.1)
                continue
            with self._result_lock:
                generation = self._generation + 1
                self._generation = generation
                self._pending = True
                self._started_at = time.monotonic()
            try:
                result = self._read_snapshot()
            except Exception as exc:
                self._log(f"[UI] Status-panel read failed: {exc}")
                result = None
            with self._result_lock:
                current = generation == self._generation
                self._pending = False
                self._started_at = 0.0
            with self._feed_state_lock:
                current = (
                    current
                    and epoch == self._lifecycle_epoch
                    and self._active
                    and not self._stopped
                )
            if current and result is not None:
                # Parser state and vitals are committed on this reader thread.
                # The UI projection is coalesced separately and cannot stop OCR
                # or allow a blocked Tk queue to build one callback per frame.
                self._record_reader_result(result, epoch)
                self._queue_ui_result(epoch, result)
            with self._feed_state_lock:
                confirmed = self._status_panel_confirmed
            delay_ms = self.pending_delay() if confirmed else STATUS_PANEL_SEARCH_MS
            # Do not sleep after a lifecycle/window wake that arrived while
            # the read was in progress; the next cycle must consume it now.
            if self._ocr_wake.is_set():
                continue
            self._ocr_wake.wait(max(0.05, delay_ms / 1000.0))

    def _read_snapshot(self) -> StatusPanelReadResult:
        with self._feed_state_lock:
            hwnd = self._config.window_id
            confirmed = self._status_panel_confirmed
            now = time.monotonic()
            refresh_max = (
                confirmed is None
                or now - self._status_panel_max_read_at >= STATUS_PANEL_MAX_REFRESH_S
            )
            refresh_client = self._status_panel_geometry_refresh
            reanchor = self._status_panel_reanchor
            client_hint = self._status_panel_client_hint
            self._status_panel_geometry_refresh = False
            self._status_panel_reanchor = False
        # Keep the native parser boundary bounded. The service scheduler stays
        # alive even if one Win32/OpenCV operation wedges; the existing
        # single-flight guard prevents overlapping native reads.
        return read_status_panel_snapshot_bounded(
            hwnd,
            confirmed,
            refresh_max=refresh_max,
            timeout_s=STATUS_PANEL_READ_TIMEOUT_S,
            client_hint=client_hint,
            refresh_client=refresh_client,
            reanchor=reanchor,
        )

    def _record_reader_result(
        self,
        result: StatusPanelReadResult,
        epoch: int,
    ) -> None:
        """Commit parser state and control vitals without touching Tk."""
        with self._feed_state_lock:
            if (
                epoch != self._lifecycle_epoch
                or result.hwnd != self._config.window_id
                or not self._active
                or self._stopped
            ):
                return
            if result.state in {"values", "hp_only", "sp_only", "hp_sp_only"}:
                self._status_panel_miss_count = 0
                self._status_panel_missing_since = None
            if result.state in {
                "values",
                "hp_only",
                "sp_only",
                "hp_sp_only",
            }:
                if result.values is not None:
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
                    if result.full_refresh:
                        self._status_panel_max_read_at = time.monotonic()
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
                hp = getattr(result, "hp", None)
                sp = getattr(result, "sp", None)
                if hp is not None:
                    self._vitals.publish_hp(*hp)
                if sp is not None and self._panel_owns_sp_weight():
                    self._vitals.publish_sp(*sp)
                return
            if result.state in {"inactive", "read_failed", "panel_missing"}:
                if result.state == "panel_missing":
                    self._status_panel_miss_count += 1
                    if self._status_panel_missing_since is None:
                        self._status_panel_missing_since = time.monotonic()
                        self._log(
                            "[UI] Status panel missing — HP/SP/weight feed paused"
                        )
                    if self._status_panel_miss_count >= 3:
                        self._status_panel_geometry_refresh = True
                        self._status_panel_reanchor = True
                        self._status_panel_miss_count = 0
                self._clear_status_panel_ui_values()
            elif result.state == "roi_missing":
                self._status_panel_miss_count += 1
                if self._status_panel_miss_count >= 3:
                    self._status_panel_geometry_refresh = True
                    self._status_panel_reanchor = True
                    self._status_panel_miss_count = 0
            elif result.state == "client_missing":
                self._status_panel_client_hint = None
                self._reset_status_panel_tracking()
                self._clear_status_panel_ui_values()

    def _clear_status_panel_ui_values(self) -> None:
        """Clear vision vitals from the reader thread, without widgets."""
        self._vitals.publish_hp(None, None)
        if self._panel_owns_sp_weight():
            self._vitals.clear_sp()
            self._vitals.publish_weight(None, None)

    def _queue_ui_result(
        self,
        epoch: int,
        result: StatusPanelReadResult,
    ) -> None:
        """Coalesce UI projections so a slow Tk loop never backlogs frames."""
        with self._ui_result_lock:
            self._latest_ui_result = (epoch, result)
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
            epoch, result = queued
            with self._feed_state_lock:
                current = (
                    epoch == self._lifecycle_epoch
                    and self._active
                    and not self._stopped
                )
            if current:
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

    # ── Submit / job ────────────────────────────────────────────────

    def pending_delay(self) -> int:
        # A locked panel is polled fast so HP/SP updates and the danger feed
        # keep flowing even while a read is in flight.
        return STATUS_PANEL_VALUE_MS

    def should_submit(self) -> int | None:
        """Schedule reads only while the application lifecycle permits them."""
        if not self.active:
            return None
        confirmed = self._status_panel_confirmed
        return (
            STATUS_PANEL_VALUE_MS
            if confirmed is not None
            else STATUS_PANEL_SEARCH_MS
        )

    def build_job(self, generation: int) -> Callable[[], None] | None:
        """Compatibility adapter for callers that still use request()."""
        def _read() -> None:
            try:
                self.publish(
                    generation,
                    read_status_panel_snapshot_bounded(
                        self._config.window_id,
                        self._status_panel_confirmed,
                        refresh_max=True,
                        timeout_s=STATUS_PANEL_READ_TIMEOUT_S,
                    ),
                )
            except Exception as exc:
                self.fail(generation, exc)

        return _read

    # ── Result handling ─────────────────────────────────────────────

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
        """Compatibility API: render and publish one result immediately."""
        with self._feed_state_lock:
            self._apply_result(result, publish_vitals=True)

    def _apply_result(
        self,
        result: StatusPanelReadResult,
        *,
        publish_vitals: bool,
    ) -> None:
        if not self.active or result.hwnd != self._config.window_id:
            return
        result_width = getattr(result, "client_width", 0)
        result_height = getattr(result, "client_height", 0)
        if (
            result_width > 0
            and result_height > 0
            and result.state in {
                "values",
                "hp_only",
                "sp_only",
                "hp_sp_only",
                "panel_open_digits_missing",
            }
        ):
            # Any of these states proves the panel/header was found in the
            # captured pixels. Keep its rectangle as a trusted fallback so a
            # transient post-teleport frame never sends the next poll through
            # synchronous Win32 geometry again.
            self._status_panel_client_hint = (
                result.client_left,
                result.client_top,
                result_width,
                result_height,
            )
        if result.state == "read_timeout":
            # A transport timeout does not prove that the panel disappeared.
            # Keep the last presentation while the dedicated reader retries.
            now = time.monotonic()
            if now - self._last_read_timeout_log_at >= STATUS_PANEL_HP_ONLY_LOG_S:
                self._last_read_timeout_log_at = now
                detail = getattr(result, "error", None)
                self._log(
                    "[UI] Status-panel read timed out — retaining last OCR state"
                    + (f": {detail}" if detail else "")
                )
            return
        if result.state in ("inactive", "read_failed"):
            self._status_panel_overlay.hide()
            if result.state == "read_failed":
                detail = getattr(result, "error", None)
                self._log(
                    "[UI] Status-panel read failed"
                    + (f": {detail}" if detail else "")
                )
            return
        if result.state == "client_missing":
            self._status_panel_client_hint = None
            self._reset_status_panel_tracking()
            self._status_panel_overlay.hide()
            self._clear_status_panel_ui()
            return
        if result.state == "roi_missing":
            self._status_panel_miss_count += 1
            if self._status_panel_miss_count >= 3:
                self._status_panel_geometry_refresh = True
                self._status_panel_reanchor = True
                self._status_panel_miss_count = 0
            return
        if result.state == "panel_missing":
            # Keep the last trusted rectangle across transient unreadable
            # frames. After several fixed-ROI misses, refresh geometry and
            # re-find the panel once; the normal hot path never searches.
            self._status_panel_miss_count += 1
            if self._status_panel_miss_count >= 3:
                self._status_panel_geometry_refresh = True
                self._status_panel_reanchor = True
                self._status_panel_miss_count = 0
            self._show_panel_missing(
                client_left=result.client_left,
                client_top=result.client_top,
            )
            return
        if result.state in ("hp_only", "sp_only", "hp_sp_only"):
            self._status_panel_miss_count = 0
            # HP and SP are independent control signals. During SIT, SP must
            # still publish even when HP or Weight is unreadable; otherwise
            # recovery waits forever on a stale/empty vitals value.
            hp = getattr(result, "hp", None)
            sp = getattr(result, "sp", None)
            if hp is not None:
                if publish_vitals:
                    self._apply_hp_only_result(hp)
                else:
                    self._on_hp(format_pair(*hp))
            if sp is not None:
                if publish_vitals:
                    self._apply_sp_only_result(sp)
                else:
                    self._on_sp(format_pair(*sp))
            return
        if result.state == "panel_open_digits_missing":
            if self._status_panel_confirmed is None:
                self._show_panel_missing(
                    client_left=result.client_left,
                    client_top=result.client_top,
                )
            else:
                # Panel header found but HP/SP/Weight digits unreadable. This
                # is a silent state (no publish, no error); surface it at a
                # throttled cadence so a persistently blind feed is visible.
                self._log_digits_missing()
            return
        if result.values is not None:
            self._status_panel_miss_count = 0
            if result.full_refresh:
                self._status_panel_max_read_at = time.monotonic()
            self._commit_status_panel(
                result.values,
                client_left=result.client_left,
                client_top=result.client_top,
                publish_vitals=publish_vitals,
            )

    def on_failure(self, exc: Exception, generation: int) -> None:
        self._log(f"[UI] Status-panel read failed: {exc}")

    def on_recover(self, stall_count: int) -> None:
        self._log(
            "[UI] Status-panel read stalled — restarted OCR worker "
            f"(stall #{stall_count})"
        )

    # ── Panel state helpers ─────────────────────────────────────────

    def reset(self) -> None:
        """Drop in-flight reads and forget the confirmed panel layout.

        A window/client change invalidates the previous window's panel origin;
        the next read must re-search for the Basic Info header instead of
        reusing stale coordinates on the new window. The generation and epoch
        advance under the same lifecycle boundary so an old read cannot commit
        in the small interval between those two invalidations.
        """
        with self._feed_state_lock:
            self._lifecycle_epoch += 1
            with self._result_lock:
                self._generation += 1
                self._pending = False
                self._started_at = 0.0
            self._reset_status_panel_tracking()
            self._status_panel_client_hint = None
            self._status_panel_geometry_refresh = False
            self._status_panel_reanchor = False
            self._status_panel_miss_count = 0
        with self._ui_result_lock:
            self._latest_ui_result = None
            self._ui_result_pending = False

    def _panel_owns_sp_weight(self) -> bool:
        """True when SP/Weight come from Basic Info OCR (Generic / no memory)."""
        return not self._config.use_memory_reading

    def _clear_vision_stats(self, placeholder: str = "—") -> None:
        """Clear vision-backed labels (HP always; SP/Weight when panel owns them)."""
        self._on_hp(placeholder)
        if self._panel_owns_sp_weight():
            self._on_sp(placeholder)
            self._on_weight(placeholder)
            self._vitals.clear_sp()

    def _clear_status_panel_ui(self) -> None:
        """Panel missing/unreadable — drop HP; drop SP/Weight only if vision owns them."""
        self._clear_vision_stats()

    def _reset_status_panel_tracking(self) -> None:
        self._status_panel_confirmed = None
        self._status_panel_max_read_at = 0.0
        self._status_panel_missing_since = None

    def _show_panel_missing(
        self,
        *,
        client_left: int,
        client_top: int,
    ) -> None:
        """Basic Info not open — clear reads and prompt to open it."""
        now = time.monotonic()
        if self._status_panel_missing_since is None:
            self._status_panel_missing_since = now
            self._log("[UI] Status panel missing — HP/SP/weight feed paused")
        elif now - self._last_panel_missing_log_at >= STATUS_PANEL_HP_ONLY_LOG_S:
            self._last_panel_missing_log_at = now
            self._log(
                "[UI] Status panel still missing "
                f"({int(now - self._status_panel_missing_since)}s)"
            )
        self._reset_status_panel_tracking()
        self._clear_status_panel_ui()
        self._status_panel_overlay.show_panel_missing(
            client_left=client_left,
            client_top=client_top,
            panel_origin=self._status_panel_anchor,
        )

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
        previous = self._status_panel_confirmed
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
