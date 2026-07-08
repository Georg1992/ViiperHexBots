"""Bot lifecycle manager — orchestrates start/stop/pause/resume.

Owns the BotController instance, the focus-polling loop, and the VIIPER
initialisation sequence.  Emits callbacks so the UI layer can react to
state transitions without being coupled to the lifecycle logic.
"""

from __future__ import annotations

import threading
import time
import tkinter as tk
from collections.abc import Callable
from enum import Enum, auto
from tkinter import messagebox

from pybot.app.bot_controller import BotController, DEFAULT_STOP_JOIN_TIMEOUT_S
from pybot.app.config_store import AppConfig
from pybot.app.overlay import Win32HuntOverlay
from pybot.mobs.catalog import MobEntry, mob_folder_by_index
from pybot.app.session_log import AppSessionLog
from pybot.app.viiper_manager import ViiperManager
from pybot.app.win32_util import is_window_active
from pybot.runtime.overlay_ports import NullOverlay


class BotState(Enum):
    """Bot lifecycle states visible to the UI layer."""

    OFF = auto()
    RUNNING = auto()
    PAUSED = auto()


class BotLifecycleManager:
    """Manages the bot runtime from VIIPER init through hunt thread lifecycle.

    Async initialisation
        ``init_viiper()`` runs on a background thread.  On success it
        sets *input_ready* to ``True`` and fires *on_input_ready*.

    Bot thread lifecycle
        ``start(config, session_id)`` / ``stop()`` /
        ``pause()`` / ``resume()`` manage the :class:`BotController`
        thread and emit *on_state_change* after each transition.

    Focus polling
        While running, a 300 ms timer checks whether the game window
        is still active.  When focus is lost, ``pause()`` is called
        automatically.
    """

    def __init__(
        self,
        root: tk.Tk,
        config: AppConfig,
        mob_catalog: list[MobEntry],
        session: AppSessionLog,
        viiper: ViiperManager,
        *,
        hunt_overlay: Win32HuntOverlay | None = None,
        on_state_change: Callable[[BotState], None] | None = None,
        on_log: Callable[[str], None] | None = None,
        on_input_ready: Callable[[], None] | None = None,
        on_exit_requested: Callable[[], None] | None = None,
    ) -> None:
        self._root = root
        self._config = config
        self._mob_catalog = mob_catalog
        self._session = session
        self._viiper = viiper
        self._hunt_overlay = hunt_overlay or Win32HuntOverlay()
        self._on_state_change = on_state_change
        self._on_log = on_log or (lambda _: None)
        self._on_input_ready_call = on_input_ready
        self._on_exit_requested_call = on_exit_requested

        self._bot: BotController | None = None
        self._state = BotState.OFF
        self._input_ready = False
        self._focus_grace_until = 0.0
        self._stop_joiner: threading.Thread | None = None

    # ── Properties ──────────────────────────────────────────────────

    @property
    def state(self) -> BotState:
        """Current bot state (OFF / RUNNING / PAUSED)."""
        return self._state

    @property
    def input_ready(self) -> bool:
        """``True`` once VIIPER has been started successfully."""
        return self._input_ready

    @property
    def window_id(self) -> int:
        """Convenience access to the configured game window HWND."""
        return self._config.window_id

    # ── VIIPER initialisation (call on a background thread) ─────────

    def init_viiper(self) -> None:
        """Start the VIIPER server.

        Call this on a background thread (e.g. ``threading.Thread(
        target=lifecycle.init_viiper, daemon=True).start()``).
        On success, *input_ready* becomes ``True`` and the
        *on_input_ready* callback fires on the main thread.
        """
        try:
            self._viiper.start()
        except (FileNotFoundError, RuntimeError) as exc:
            self._root.after(
                0,
                lambda: messagebox.showerror("ViiperHexBots", str(exc)),
            )
            if self._on_exit_requested_call:
                self._root.after(0, self._on_exit_requested_call)
            return
        self._input_ready = True
        if self._on_input_ready_call:
            self._root.after(0, self._on_input_ready_call)

    # ── Bot lifecycle ───────────────────────────────────────────────

    def start(
        self,
        config_snapshot: AppConfig,
        session_id: str,
    ) -> None:
        """Start the hunt runtime on a daemon thread.

        Each call generates a fresh ``hunt_session_id`` so the runtime
        logs go into a new directory — not the app-level session dir.

        Args:
            config_snapshot: Fully synced AppConfig with current UI values.
            session_id: App-level session identifier (used as prefix).
        """
        if self._state != BotState.OFF:
            return

        self._await_prior_stop_joiner()

        if self._bot is not None:
            self._bot.request_stop()
            self._bot = None

        mob_name = mob_folder_by_index(
            self._mob_catalog, config_snapshot.selected_monster
        )

        runtime_overlay = (
            self._hunt_overlay
            if config_snapshot.hunt_log_overlay
            else NullOverlay()
        )

        self._bot = BotController(
            app_config=config_snapshot,
            session_id=session_id,
            on_log=self._on_log,
            overlay=runtime_overlay,
        )
        self._bot.start(mob_name=mob_name)
        self._state = BotState.RUNNING
        self._arm_focus_grace()
        # Start overlay upkeep timer (reposition + repaint every 100ms)
        self._root.after(100, self._schedule_overlay_tick)
        self._session.write_block(
            "bot start",
            f"hwnd={config_snapshot.window_id}\n"
            f"mobIndex={config_snapshot.selected_monster}\n"
            f"huntSession={session_id}",
        )
        if self._on_state_change:
            self._on_state_change(BotState.RUNNING)
        self._root.after(300, self._poll_focus)

    def _schedule_overlay_tick(self) -> None:
        """Periodic overlay upkeep while the bot is running."""
        if self._state != BotState.OFF:
            self._hunt_overlay.tick()
            self._root.after(100, self._schedule_overlay_tick)

    def set_search_range_cells(self, cells: int) -> None:
        """Keep hunt capture and overlay search boxes aligned with the GUI."""
        self._hunt_overlay.set_search_range_cells(cells)
        if self._bot is not None:
            self._bot.set_search_range_cells(cells)

    def stop(self) -> None:
        """Stop the hunt runtime and destroy the overlay."""
        if self._state == BotState.OFF and self._bot is None:
            return

        bot = self._bot
        if bot is not None:
            bot.request_stop()
        self._bot = None
        self._state = BotState.OFF
        self._hunt_overlay.reset_stats()
        self._hunt_overlay.destroy()
        if self._on_state_change:
            self._on_state_change(BotState.OFF)
        if bot is not None:
            self._start_stop_joiner(bot)

    def _start_stop_joiner(self, bot: BotController) -> None:
        self._await_prior_stop_joiner()

        def _join() -> None:
            bot.stop(join_timeout=DEFAULT_STOP_JOIN_TIMEOUT_S)

        self._stop_joiner = threading.Thread(
            target=_join,
            name="bot-stop-joiner",
            daemon=True,
        )
        self._stop_joiner.start()

    def _await_prior_stop_joiner(self) -> None:
        if self._stop_joiner is not None and self._stop_joiner.is_alive():
            self._stop_joiner.join(timeout=DEFAULT_STOP_JOIN_TIMEOUT_S)
        self._stop_joiner = None

    def pause(self) -> None:
        """Pause the hunt runtime because game focus was lost."""
        if self._state != BotState.RUNNING:
            return
        if self._bot is not None:
            self._bot.pause()
        self._state = BotState.PAUSED
        self._on_log("[STATE] Bot paused")
        self._session.write_focus_change("paused (focus lost)")
        if self._on_state_change:
            self._on_state_change(BotState.PAUSED)

    def resume(self) -> None:
        """Resume the hunt runtime after focus is regained."""
        if self._state != BotState.PAUSED:
            return
        if self._bot is not None:
            self._bot.resume()
        self._state = BotState.RUNNING
        self._arm_focus_grace()
        self._on_log("[STATE] Bot resumed")
        self._session.write_focus_change("resumed")
        if self._on_state_change:
            self._on_state_change(BotState.RUNNING)
        self._root.after(300, self._poll_focus)

    def _arm_focus_grace(self, seconds: float = 2.0) -> None:
        """Ignore focus-loss auto-pause briefly after start/resume."""
        self._focus_grace_until = time.monotonic() + seconds

    # ── Focus polling ───────────────────────────────────────────────

    def _poll_focus(self) -> None:
        if (
            self._state == BotState.RUNNING
            and self._config.window_id
            and time.monotonic() >= self._focus_grace_until
            and not is_window_active(self._config.window_id)
        ):
            self.pause()
        if self._state in (BotState.RUNNING, BotState.PAUSED):
            self._root.after(300, self._poll_focus)
