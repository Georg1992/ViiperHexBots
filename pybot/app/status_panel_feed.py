"""Status-panel OCR observation feed for the main window.

Periodically OCRs the client's Basic Info panel and publishes HP/SP/weight
into the shared :class:`PlayerVitals` plus the bot UI labels. The capture/OCR
runs on the internal worker thread; all label/overlay/vitals updates happen
on the Tk thread via the runner's result callback.

The result state machine (panel missing, HP-only, digits unreadable, full
refresh, ...) lives here so MainWindow only wires the feed.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from pybot.app.periodic_task_runner import PeriodicTaskRunner
from pybot.app.status_display import format_pair, status_panel_numbers
from pybot.app.status_panel_worker import (
    StatusPanelReadResult,
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
# A status-panel read must never pin its pending flag forever: a wedged
# capture/OCR would otherwise silently kill the HP/SP feed (and with it
# danger protection) with no error and no log line. After this long the
# worker is abandoned and recreated.
STATUS_PANEL_READ_TIMEOUT_S = 6.0


class StatusPanelFeed(PeriodicTaskRunner):
    """One observation feed: periodic Basic Info OCR reads."""

    def __init__(
        self,
        *,
        root,
        config,
        vitals,
        overlay,
        panel_active: Callable[[], bool],
        log: Callable[[str], None],
        post_to_tk: Callable[[Callable[[], None]], None],
        on_hp: Callable[[str], None],
        on_sp: Callable[[str], None],
        on_weight: Callable[[str], None],
    ) -> None:
        super().__init__(
            root=root,
            name="ui-status-reader",
            timeout_s=STATUS_PANEL_READ_TIMEOUT_S,
            default_delay_ms=STATUS_PANEL_SEARCH_MS,
            post_to_tk=post_to_tk,
            log=log,
        )
        self._config = config
        self._vitals = vitals
        self._status_panel_overlay = overlay
        self._panel_active = panel_active
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
        # Last known Basic Info origin — anchors the open-panel prompt.
        self._status_panel_anchor: tuple[int, int] = (0, 0)

    # ── Submit / job ────────────────────────────────────────────────

    def pending_delay(self) -> int:
        # A locked panel is polled fast so HP/SP updates and the danger feed
        # keep flowing even while a read is in flight.
        return STATUS_PANEL_VALUE_MS

    def should_submit(self) -> int | None:
        if not self._panel_active():
            # No OCR while off/paused — cheaply re-check later. Also drop any
            # stale missing/recovery bookkeeping so a panel gap from a previous
            # run is not attributed to the next one, and hide the overlay so
            # the "open Basic Info" prompt cannot linger over the game.
            self._status_panel_missing_since = None
            self._status_panel_overlay.hide()
            return None
        confirmed = self._status_panel_confirmed
        return (
            STATUS_PANEL_VALUE_MS
            if confirmed is not None
            else STATUS_PANEL_SEARCH_MS
        )

    def build_job(self, generation: int) -> Callable[[], None] | None:
        hwnd = self._config.window_id
        confirmed = self._status_panel_confirmed
        now = time.monotonic()
        refresh_max = (
            confirmed is None
            or now - self._status_panel_max_read_at >= STATUS_PANEL_MAX_REFRESH_S
        )

        def _read() -> None:
            try:
                result = read_status_panel_snapshot_bounded(
                    hwnd, confirmed, refresh_max=refresh_max
                )
            except Exception as exc:
                self.fail(generation, exc)
                return
            self.publish(generation, result)

        return _read

    # ── Result handling ─────────────────────────────────────────────

    def apply_result(self, result: StatusPanelReadResult) -> None:
        if result.hwnd != self._config.window_id:
            return
        if result.state in ("inactive", "read_timeout", "read_failed"):
            self._status_panel_overlay.hide()
            if result.state == "read_timeout":
                self._log(
                    "[UI] Status-panel read timed out — discarded blocked OCR read"
                )
            elif result.state == "read_failed":
                self._log("[UI] Status-panel read failed")
            return
        if result.state == "client_missing":
            self._reset_status_panel_tracking()
            self._status_panel_overlay.hide()
            self._clear_status_panel_ui()
            return
        if result.state == "panel_missing":
            self._show_panel_missing(
                client_left=result.client_left,
                client_top=result.client_top,
            )
            return
        if result.state == "hp_only":
            # HP damage detection must not depend on SP/weight OCR. A malformed
            # SP or weight row is common during bar animation; retain the last
            # confirmed panel for those values but publish the independently
            # parsed HP pair immediately so DangerDetector can see damage.
            self._apply_hp_only_result(result.hp)
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
            if result.full_refresh:
                self._status_panel_max_read_at = time.monotonic()
            self._commit_status_panel(
                result.values,
                client_left=result.client_left,
                client_top=result.client_top,
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
        reusing stale coordinates on the new window.
        """
        super().reset()
        self._reset_status_panel_tracking()

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
        self._vitals.publish_hp(values.hp, values.hp_max)
        if self._panel_owns_sp_weight():
            self._vitals.publish_sp(values.sp, values.sp_max)
            self._vitals.publish_weight(values.weight, values.weight_max)
        if previous is not None and status_panel_numbers(
            previous
        ) == status_panel_numbers(values):
            return
        if self._panel_owns_sp_weight():
            self._apply_status_panel_stats(values)

    def _apply_status_panel_stats(self, values: StatusPanelValues) -> None:
        """Apply vision SP/Weight when memory is off (HP set in commit)."""
        if not self._panel_owns_sp_weight():
            return
        self._on_sp(format_pair(values.sp, values.sp_max))
        self._on_weight(format_pair(values.weight, values.weight_max))
