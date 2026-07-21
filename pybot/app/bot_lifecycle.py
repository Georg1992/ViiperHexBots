"""Bot lifecycle manager — orchestrates start/stop/pause/resume.

Heavy hunt startup runs on a background thread.  Cross-thread UI work is
queued and drained on the Tk main thread (never ``root.after`` from workers).
"""

from __future__ import annotations

import queue
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
from pybot.app.win32_util import is_window_active, restore_and_activate
from pybot.runtime.overlay_ports import NullOverlay

_MAIN_DISPATCH_MS = 50
_MAX_DISPATCH_PER_TICK = 20


class BotState(Enum):
    """Bot lifecycle states visible to the UI layer."""

    OFF = auto()
    STARTING = auto()
    RUNNING = auto()
    PAUSED = auto()


class BotLifecycleManager:
    """Manages the bot runtime from VIIPER init through hunt thread lifecycle."""

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
        self._start_thread: threading.Thread | None = None
        self._start_cancelled = False
        self._start_generation = 0
        self._main_queue: queue.Queue[Callable[[], None]] = queue.Queue()
        self._root.after(_MAIN_DISPATCH_MS, self._drain_main_queue)

    @property
    def state(self) -> BotState:
        return self._state

    @property
    def input_ready(self) -> bool:
        return self._input_ready

    @property
    def window_id(self) -> int:
        return self._config.window_id

    def _post_to_main(self, callback: Callable[[], None]) -> None:
        self._main_queue.put_nowait(callback)

    def _drain_main_queue(self) -> None:
        processed = 0
        try:
            while processed < _MAX_DISPATCH_PER_TICK:
                callback = self._main_queue.get_nowait()
                try:
                    callback()
                except Exception as exc:
                    self._on_log(f"[STATE] UI callback error: {exc}")
                processed += 1
        except queue.Empty:
            pass
        finally:
            self._root.after(_MAIN_DISPATCH_MS, self._drain_main_queue)

    def init_viiper(self) -> None:
        try:
            self._viiper.start()
        except (FileNotFoundError, RuntimeError) as exc:
            self._post_to_main(
                lambda: messagebox.showerror("ViiperHexBots", str(exc)),
            )
            if self._on_exit_requested_call:
                self._post_to_main(self._on_exit_requested_call)
            return

        def _mark_input_ready() -> None:
            self._input_ready = True
            if self._on_input_ready_call:
                self._on_input_ready_call()

        self._post_to_main(_mark_input_ready)

    def await_shutdown(self, timeout: float = DEFAULT_STOP_JOIN_TIMEOUT_S + 1.0) -> None:
        """Block until async start/stop threads finish (for app exit)."""
        self._start_cancelled = True
        if self._start_thread is not None and self._start_thread.is_alive():
            self._start_thread.join(timeout=timeout)
        if self._stop_joiner is not None and self._stop_joiner.is_alive():
            self._stop_joiner.join(timeout=timeout)

    def start(
        self,
        config_snapshot: AppConfig,
        session_id: str,
    ) -> bool:
        """Begin async hunt startup. Returns True when accepted."""
        if self._state not in (BotState.OFF,):
            return False
        if self._start_thread is not None and self._start_thread.is_alive():
            return False

        self._start_cancelled = False
        self._start_generation += 1
        generation = self._start_generation
        self._state = BotState.STARTING
        self._post_to_main(
            lambda: self._on_state_change(BotState.STARTING)
            if self._on_state_change
            else None
        )

        def _run_start() -> None:
            try:
                self._await_prior_stop_joiner()
                if self._start_cancelled:
                    return

                restore_and_activate(config_snapshot.window_id)
                self._session.open(session_id=session_id)
                if self._start_cancelled:
                    return

                mob_name = mob_folder_by_index(
                    self._mob_catalog, config_snapshot.selected_monster
                )
                runtime_overlay = (
                    self._hunt_overlay
                    if config_snapshot.hunt_log_overlay
                    else NullOverlay()
                )
                bot = BotController(
                    app_config=config_snapshot,
                    session_id=session_id,
                    on_log=self._on_log,
                    overlay=runtime_overlay,
                )
                if self._start_cancelled:
                    return

                bot.start(mob_name=mob_name)
                self._post_to_main(
                    lambda: self._finish_start(
                        bot,
                        config_snapshot=config_snapshot,
                        session_id=session_id,
                        generation=generation,
                    ),
                )
            except Exception as exc:
                self._post_to_main(
                    lambda err=exc: self._fail_start(err, generation=generation),
                )

        self._start_thread = threading.Thread(
            target=_run_start,
            name="bot-start",
            daemon=True,
        )
        self._start_thread.start()
        return True

    def _finish_start(
        self,
        bot: BotController,
        *,
        config_snapshot: AppConfig,
        session_id: str,
        generation: int,
    ) -> None:
        if generation != self._start_generation:
            if bot.running:
                bot.request_stop()
                self._start_stop_joiner(bot, destroy_overlay=False)
            return

        if self._start_cancelled or self._state != BotState.STARTING:
            if bot.running:
                bot.request_stop()
                self._start_stop_joiner(bot)
            if self._state == BotState.STARTING:
                self._state = BotState.OFF
                if self._on_state_change:
                    self._on_state_change(BotState.OFF)
            return

        if not bot.running:
            self._on_log("[STATE] Bot start failed — hunt thread did not start")
            self._state = BotState.OFF
            if self._on_state_change:
                self._on_state_change(BotState.OFF)
            return

        self._bot = bot
        self._state = BotState.RUNNING
        self._arm_focus_grace()

        if config_snapshot.hunt_log_overlay and config_snapshot.window_id:
            ok = self._hunt_overlay.create(
                config_snapshot.window_id,
                search_range_cells=config_snapshot.search_range,
            )
            if ok:
                self._on_log(f"[OVERLAY] created on hwnd={config_snapshot.window_id}")
            else:
                self._on_log(f"[OVERLAY] failed: {self._hunt_overlay.last_error()}")

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
        self._on_log("[STATE] Hunt runtime started")

    def _fail_start(self, exc: Exception, *, generation: int) -> None:
        if generation != self._start_generation:
            return
        if self._state != BotState.STARTING:
            return
        self._on_log(f"[STATE] Bot start failed: {exc}")
        self._state = BotState.OFF
        if self._on_state_change:
            self._on_state_change(BotState.OFF)

    def stop(self) -> None:
        if self._state == BotState.OFF and self._bot is None:
            return

        if self._state == BotState.STARTING:
            self._start_cancelled = True

        bot = self._bot
        if bot is not None:
            bot.request_stop()

        self._bot = None
        self._state = BotState.OFF
        self._hunt_overlay.reset_stats()
        if bot is None:
            self._hunt_overlay.destroy()
        if self._on_state_change:
            self._on_state_change(BotState.OFF)
        if bot is not None:
            self._start_stop_joiner(bot)

    def _destroy_hunt_overlay(self) -> None:
        self._hunt_overlay.destroy()

    def _start_stop_joiner(
        self,
        bot: BotController,
        *,
        destroy_overlay: bool = True,
    ) -> None:
        def _join() -> None:
            # Retry until the hunt thread exits — do not start another hunt over it.
            stopped = bot.stop(join_timeout=DEFAULT_STOP_JOIN_TIMEOUT_S)
            if not stopped:
                stopped = bot.stop(join_timeout=DEFAULT_STOP_JOIN_TIMEOUT_S)
            if not stopped:
                self._on_log(
                    "[STATE] Hunt thread still alive after stop join — "
                    "restart will wait for it"
                )
                bot.stop(join_timeout=DEFAULT_STOP_JOIN_TIMEOUT_S * 2)
            if destroy_overlay:
                self._post_to_main(self._destroy_hunt_overlay)

        self._stop_joiner = threading.Thread(
            target=_join,
            name="bot-stop-joiner",
            daemon=True,
        )
        self._stop_joiner.start()

    def _await_prior_stop_joiner(self) -> None:
        """Block until the previous hunt fully stopped (required before restart)."""
        if self._stop_joiner is not None and self._stop_joiner.is_alive():
            # Stop may need more than one join window if workers were busy.
            self._stop_joiner.join(timeout=DEFAULT_STOP_JOIN_TIMEOUT_S * 2)
        self._stop_joiner = None

    def pause(self) -> None:
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

    def set_search_range_cells(self, cells: int) -> None:
        self._hunt_overlay.set_search_range_cells(cells)
        if self._bot is not None:
            self._bot.set_search_range_cells(cells)

    def _arm_focus_grace(self, seconds: float = 2.0) -> None:
        self._focus_grace_until = time.monotonic() + seconds

    def _schedule_overlay_tick(self) -> None:
        if self._state != BotState.OFF:
            self._hunt_overlay.tick()
            self._root.after(100, self._schedule_overlay_tick)

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
