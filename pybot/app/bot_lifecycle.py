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
from pybot.game_state import PlayerVitals
from pybot.mobs.catalog import MobEntry, mob_folder_by_index
from pybot.app.session_log import AppSessionLog
from pybot.app.viiper_manager import ViiperManager
from pybot.app.win32_util import is_window_active, restore_and_activate
from pybot.runtime.overlay_ports import NullOverlay

_MAIN_DISPATCH_MS = 50
_MAX_DISPATCH_PER_TICK = 20
# A stop attempt must yield even when a worker is permanently non-cooperative.
# Ownership remains with this lifecycle manager so a later Stop can retry.
_STOP_RETRY_ATTEMPTS = 3


class BotState(Enum):
    """Bot lifecycle states visible to the UI layer."""

    OFF = auto()
    STARTING = auto()
    RUNNING = auto()
    STOPPING = auto()
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
        vitals: PlayerVitals | None = None,
        stream_store=None,
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
        self._hunt_overlay = (
            Win32HuntOverlay() if hunt_overlay is None else hunt_overlay
        )
        self._vitals = PlayerVitals() if vitals is None else vitals
        # Process-wide VIIPER stream store shared with ViiperManager.
        self._stream_store = stream_store
        self._on_state_change = on_state_change
        self._on_log = (lambda _: None) if on_log is None else on_log
        self._on_input_ready_call = on_input_ready
        self._on_exit_requested_call = on_exit_requested

        self._bot: BotController | None = None
        self._state = BotState.OFF
        self._input_ready = False
        self._focus_grace_until = 0.0
        self._stop_joiner: threading.Thread | None = None
        # Serializes ownership handoff between the Tk thread and stop joiner.
        self._ownership_lock = threading.RLock()
        self._stopping = False
        self._start_thread: threading.Thread | None = None
        self._start_cancelled = False
        self._start_generation = 0
        # A queued finish callback still owns a just-launched bot until Tk
        # consumes it. Prevent restart/exit from outrunning that handoff.
        self._pending_start_callbacks = 0
        # Failed startup sessions are closed by the bot's stop joiner. Keep the
        # ownership marker across bounded retry attempts so cleanup cannot be
        # lost when the first stop attempt yields.
        self._failed_start_session_owner: BotController | None = None
        # Stale terminal callbacks can hand a bot to an independent cleanup
        # joiner when a newer lifecycle owner already exists. Track those bots
        # so application exit cannot outrun their cleanup.
        self._orphan_cleanup_bots: set[BotController] = set()
        self._orphan_cleanup_threads: set[threading.Thread] = set()
        # Bumped when a new hunt owns the overlay so a late stop-joiner cannot
        # destroy the overlay of a newer start.
        self._overlay_epoch = 0
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

    @property
    def stopping(self) -> bool:
        """True while a prior hunt is still unwinding its worker threads."""
        return self._stopping

    def retry_shutdown_cleanup(self) -> None:
        """Retry stale startup cleanup without blocking the Tk thread."""
        with self._ownership_lock:
            retry_bots = [
                bot
                for bot in self._orphan_cleanup_bots
                if not any(
                    thread.is_alive()
                    for thread in self._orphan_cleanup_threads
                )
            ]
        for bot in retry_bots:
            with self._ownership_lock:
                self._orphan_cleanup_bots.discard(bot)
            self._start_orphan_stop_joiner(
                bot,
                end_session=self._failed_start_session_owner is bot,
            )

    @property
    def shutdown_ready(self) -> bool:
        """True when lifecycle workers no longer need a shutdown join."""
        with self._ownership_lock:
            return not (
                self._stopping
                or (
                    self._start_thread is not None
                    and self._start_thread.is_alive()
                )
                or (
                    self._stop_joiner is not None
                    and self._stop_joiner.is_alive()
                )
                or self._pending_start_callbacks > 0
                or self._failed_start_session_owner is not None
                or bool(self._orphan_cleanup_bots)
            )

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
            error_text = str(exc)
            self._post_to_main(
                lambda: messagebox.showerror("ViiperHexBots", error_text),
            )
            if self._on_exit_requested_call:
                self._post_to_main(self._on_exit_requested_call)
            return

        def _mark_input_ready() -> None:
            self._input_ready = True
            if self._on_input_ready_call:
                self._on_input_ready_call()

        self._post_to_main(_mark_input_ready)


    def _is_current_start(self, generation: int) -> bool:
        """True when *generation* is still the active, non-cancelled start."""
        return (
            generation == self._start_generation
            and not self._start_cancelled
            and self._state == BotState.STARTING
        )

    @staticmethod
    def _bot_shutdown_pending(bot: BotController) -> bool:
        """Read the optional controller ownership flag compatibly.

        Older/custom doubles may not define ``shutdown_pending``; MagicMock
        also fabricates arbitrary attributes. Only a real bool is authoritative
        so those doubles continue to use their existing ``running`` contract.
        """
        pending = getattr(bot, "shutdown_pending", None)
        if isinstance(pending, bool):
            return pending
        return bool(getattr(bot, "running", False))

    def _emit_state(self, state: BotState, *, generation: int | None = None) -> None:
        """Notify UI of *state*, ignoring stale STARTING events after cancel."""
        if generation is not None and generation != self._start_generation:
            return
        if state == BotState.STARTING and self._state != BotState.STARTING:
            return
        if self._on_state_change:
            self._on_state_change(state)

    def start(
        self,
        config_snapshot: AppConfig,
        session_id: str,
    ) -> bool:
        """Begin async hunt startup. Returns True when accepted.

        A cancelled in-flight start thread may still be alive; this bumps the
        start generation so that thread's finish is ignored and restart works.
        """
        with self._ownership_lock:
            # A completed stop may have queued its UI callback but already
            # released ownership. Reconcile that state before applying the
            # normal start guards so a quick restart cannot be blocked by stale
            # STOPPING.
            if (
                self._state == BotState.STOPPING
                and not self._stopping
                and self._bot is None
                and (
                    self._stop_joiner is None
                    or not self._stop_joiner.is_alive()
                )
            ):
                self._state = BotState.OFF
                self._emit_state(BotState.OFF)
            if self._state not in (BotState.OFF,):
                return False
            if self._stopping:
                self._on_log(
                    "[STATE] Start refused — previous hunt is still stopping"
                )
                return False
            if self._start_thread is not None and self._start_thread.is_alive():
                self._on_log(
                    "[STATE] Start refused — previous startup is still stopping"
                )
                return False
            # A previous hunt may still be unwinding in the background after
            # the UI has returned to OFF. Never launch over that joiner.
            if self._stop_joiner is not None and self._stop_joiner.is_alive():
                self._on_log(
                    "[STATE] Start refused — previous hunt is still stopping"
                )
                return False
            if self._pending_start_callbacks > 0:
                self._on_log(
                    "[STATE] Start refused — previous startup callback is pending"
                )
                return False
            if self._failed_start_session_owner is not None:
                self._on_log(
                    "[STATE] Start refused — previous startup session is still owned"
                )
                return False
            if self._orphan_cleanup_bots:
                self._on_log(
                    "[STATE] Start refused — stale startup cleanup is still running"
                )
                return False

            self._start_cancelled = False
            self._start_generation += 1
            generation = self._start_generation
            self._state = BotState.STARTING
        self._post_to_main(
            lambda: self._emit_state(BotState.STARTING, generation=generation),
        )

        def _run_start() -> None:
            posted_terminal = False
            startup_session_opened = False
            session_cleanup_transferred = False
            bot: BotController | None = None
            try:
                self._on_log("[STATE] Start: waiting for prior hunt to exit")
                if not self._await_prior_stop_joiner():
                    self._on_log(
                        "[STATE] Start aborted — prior hunt is still stopping"
                    )
                    return
                if not self._is_current_start(generation):
                    return

                self._on_log("[STATE] Start: ensuring VIIPER devices")
                self._viiper.ensure_devices()
                if not self._is_current_start(generation):
                    return

                restore_and_activate(config_snapshot.window_id)
                self._session.open(session_id=session_id)
                startup_session_opened = True
                if not self._is_current_start(generation):
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
                    vitals=self._vitals,
                    stream_store=self._stream_store,
                )
                with self._ownership_lock:
                    if startup_session_opened:
                        self._failed_start_session_owner = bot
                if not self._is_current_start(generation):
                    return

                self._on_log(f"[STATE] Start: launching hunt thread mob={mob_name}")
                bot.start(mob_name=mob_name)
                if not self._is_current_start(generation):
                    pending = self._bot_shutdown_pending(bot)
                    if pending:
                        with self._ownership_lock:
                            self._bot = bot
                        bot.request_stop()
                        self._start_stop_joiner(
                            bot,
                            destroy_overlay=False,
                            end_session=startup_session_opened,
                        )
                        session_cleanup_transferred = startup_session_opened
                    elif startup_session_opened:
                        self._close_failed_start_session(
                            bot, "bot start superseded"
                        )
                    return

                with self._ownership_lock:
                    self._pending_start_callbacks += 1
                self._post_to_main(
                    lambda: self._consume_start_callback(
                        lambda: self._finish_start(
                            bot,
                            config_snapshot=config_snapshot,
                            session_id=session_id,
                            generation=generation,
                        )
                    ),
                )
                posted_terminal = True
            except Exception as exc:
                # Startup may have created a controller/runtime before failing
                # (including a Thread.start failure). Do not report OFF while
                # that object still owns workers or input; hand it to the same
                # bounded stop joiner used by normal Stop.
                pending = bot is not None and self._bot_shutdown_pending(bot)
                cleanup_started = False
                if pending and bot is not None:
                    try:
                        with self._ownership_lock:
                            self._bot = bot
                        bot.request_stop()
                        self._start_stop_joiner(
                            bot,
                            destroy_overlay=False,
                            end_session=startup_session_opened,
                        )
                        session_cleanup_transferred = startup_session_opened
                        cleanup_started = True
                    except Exception as cleanup_exc:
                        # Retain ownership even if the first cleanup handoff
                        # itself fails. A later Stop can retry request_stop and
                        # create the joiner without exposing a false OFF state.
                        with self._ownership_lock:
                            self._bot = bot
                            self._stopping = True
                            self._state = BotState.STOPPING
                        cleanup_started = True
                        self._on_log(
                            f"[STATE] Failed-start cleanup could not be scheduled: {cleanup_exc}"
                        )
                elif startup_session_opened:
                    # No runtime remains to own the session writer. This runs
                    # on the startup thread, never on Tk, so session.end cannot
                    # freeze the UI while flushing its queue.
                    if bot is not None:
                        self._close_failed_start_session(
                            bot, "bot start failed"
                        )
                    else:
                        try:
                            self._session.end("bot start failed")
                        except Exception as cleanup_exc:
                            self._on_log(
                                f"[STATE] Failed-start session cleanup failed: {cleanup_exc}"
                            )
                self._post_to_main(
                    lambda err=exc, cleanup_started=cleanup_started: self._fail_start(
                        err,
                        generation=generation,
                        cleanup_started=cleanup_started,
                    ),
                )
                posted_terminal = True
            finally:
                if not posted_terminal:
                    if startup_session_opened and not session_cleanup_transferred:
                        if bot is not None:
                            self._close_failed_start_session(
                                bot, "bot start cancelled"
                            )
                        else:
                            try:
                                self._session.end("bot start cancelled")
                            except Exception as cleanup_exc:
                                self._on_log(
                                    f"[STATE] Cancelled-start session cleanup failed: {cleanup_exc}"
                                )
                    self._post_to_main(
                        lambda: self._clear_stuck_starting(generation),
                    )

        start_thread = threading.Thread(
            target=_run_start,
            name="bot-start",
            daemon=True,
        )
        with self._ownership_lock:
            self._start_thread = start_thread
        try:
            start_thread.start()
        except Exception as exc:
            with self._ownership_lock:
                if self._start_thread is start_thread:
                    self._start_thread = None
                self._start_cancelled = True
                if self._state == BotState.STARTING:
                    self._state = BotState.OFF
                    self._emit_state(BotState.OFF)
            self._on_log(f"[STATE] Bot startup thread failed: {exc}")
            raise
        return True

    def _consume_start_callback(self, callback: Callable[[], None]) -> None:
        """Consume one queued terminal-start callback before executing it."""
        with self._ownership_lock:
            self._pending_start_callbacks = max(
                0, self._pending_start_callbacks - 1
            )
        callback()

    def _clear_stuck_starting(self, generation: int) -> None:
        """If start aborted without finish/fail, do not leave UI stuck on Starting."""
        if generation != self._start_generation:
            return
        if self._state != BotState.STARTING:
            return
        self._on_log("[STATE] Bot start aborted before hunt thread")
        self._state = BotState.OFF
        self._emit_state(BotState.OFF)

    def _start_orphan_stop_joiner(
        self,
        bot: BotController,
        *,
        end_session: bool,
    ) -> None:
        """Unwind a stale startup bot without changing current lifecycle state."""
        with self._ownership_lock:
            if bot in self._orphan_cleanup_bots:
                return
            self._orphan_cleanup_bots.add(bot)

        def _join_orphan() -> None:
            stopped = False
            try:
                for _attempt in range(_STOP_RETRY_ATTEMPTS):
                    try:
                        if bot.stop(join_timeout=DEFAULT_STOP_JOIN_TIMEOUT_S):
                            stopped = True
                            break
                    except Exception as exc:
                        self._on_log(
                            f"[STATE] Stale-start cleanup failed: {exc}"
                        )
                if stopped and end_session:
                    self._close_failed_start_session(
                        bot, "bot start superseded"
                    )
                if not stopped:
                    self._on_log(
                        "[STATE] Stale-start cleanup incomplete; "
                        "resources remain owned"
                    )
            finally:
                with self._ownership_lock:
                    self._orphan_cleanup_threads.discard(threading.current_thread())
                    if stopped:
                        self._orphan_cleanup_bots.discard(bot)

        thread = threading.Thread(
            target=_join_orphan,
            name="bot-stale-start-joiner",
            daemon=True,
        )
        with self._ownership_lock:
            self._orphan_cleanup_threads.add(thread)
        thread.start()

    def _finish_start(
        self,
        bot: BotController,
        *,
        config_snapshot: AppConfig,
        session_id: str,
        generation: int,
    ) -> None:
        if generation != self._start_generation:
            pending = self._bot_shutdown_pending(bot)
            with self._ownership_lock:
                current_bot = self._bot
            if pending and current_bot is not None and current_bot is not bot:
                # This callback is stale and another hunt owns lifecycle state;
                # never let the old bot's joiner set STOPPING/OFF globally.
                try:
                    bot.request_stop()
                except Exception as exc:
                    self._on_log(f"[STATE] Stale-start stop request failed: {exc}")
                self._start_orphan_stop_joiner(
                    bot,
                    end_session=self._failed_start_session_owner is bot,
                )
            elif pending:
                with self._ownership_lock:
                    self._bot = bot
                bot.request_stop()
                self._start_stop_joiner(
                    bot,
                    destroy_overlay=False,
                    end_session=True,
                )
            else:
                self._close_failed_start_session(bot, "bot start superseded")
            return

        if self._start_cancelled or self._state != BotState.STARTING:
            pending = self._bot_shutdown_pending(bot)
            with self._ownership_lock:
                current_bot = self._bot
            if pending and current_bot is not None and current_bot is not bot:
                try:
                    bot.request_stop()
                except Exception as exc:
                    self._on_log(f"[STATE] Cancelled-start stop request failed: {exc}")
                self._start_orphan_stop_joiner(
                    bot,
                    end_session=self._failed_start_session_owner is bot,
                )
            elif pending:
                with self._ownership_lock:
                    self._bot = bot
                bot.request_stop()
                self._start_stop_joiner(
                    bot,
                    destroy_overlay=False,
                    end_session=True,
                )
            else:
                self._close_failed_start_session(bot, "bot start cancelled")
            if self._state == BotState.STARTING:
                self._state = BotState.OFF
                self._emit_state(BotState.OFF)
            return

        if not bot.running:
            if self._bot_shutdown_pending(bot):
                # The top-level thread may have returned while workers remain
                # owned. Keep the controller and route it through the same
                # bounded stop joiner rather than abandoning those workers.
                self._bot = bot
                bot.request_stop()
                self._start_stop_joiner(
                    bot,
                    end_session=self._failed_start_session_owner is bot,
                )
                return
            self._on_log("[STATE] Bot start failed — hunt thread did not start")
            # The startup thread already transferred session ownership to this
            # callback, but no runtime exists to close it. Close it here before
            # exposing OFF so the next start cannot inherit a stale writer.
            self._close_failed_start_session(bot, "bot start failed")
            self._state = BotState.OFF
            self._emit_state(BotState.OFF)
            return

        with self._ownership_lock:
            if self._failed_start_session_owner is bot:
                # Normal startup now owns the open session; it will be closed
                # by MainWindow's final application shutdown, not bot Stop.
                self._failed_start_session_owner = None
        self._bot = bot
        self._state = BotState.RUNNING
        self._arm_focus_grace()
        self._overlay_epoch += 1

        self._session.write_block(
            "bot start",
            f"hwnd={config_snapshot.window_id}\n"
            f"mobIndex={config_snapshot.selected_monster}\n"
            f"huntSession={session_id}",
        )

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
        self._emit_state(BotState.RUNNING)
        self._root.after(300, self._poll_focus)
        self._on_log("[STATE] Hunt runtime started")

    def _close_failed_start_session(
        self,
        bot: BotController,
        reason: str,
    ) -> None:
        """Close a session only if this failed-start bot still owns it."""
        with self._ownership_lock:
            if self._failed_start_session_owner is not bot:
                return
            self._failed_start_session_owner = None
        try:
            self._session.end(reason)
        except Exception as exc:
            self._on_log(f"[STATE] Failed-start session close failed: {exc}")

    def _fail_start(
        self,
        exc: Exception,
        *,
        generation: int,
        cleanup_started: bool = False,
    ) -> None:
        if generation != self._start_generation:
            return
        self._on_log(f"[STATE] Bot start failed: {exc}")
        if cleanup_started:
            # The stop joiner owns the terminal transition. Keeping STOPPING
            # visible prevents a second Start from racing that cleanup.
            return
        if self._state != BotState.STARTING:
            return
        self._state = BotState.OFF
        self._emit_state(BotState.OFF)

    def stop(self) -> None:
        with self._ownership_lock:
            if self._stopping:
                # A bounded joiner may have finished without success. Preserve
                # the bot handle and allow a later Stop to start one retry.
                bot = self._bot
                joiner = self._stop_joiner
                if bot is not None and (
                    joiner is None or not joiner.is_alive()
                ):
                    try:
                        bot.request_stop()
                    except Exception as exc:
                        self._on_log(f"[STATE] Stop retry request failed: {exc}")
                        return
                    self._start_stop_joiner(
                        bot,
                        overlay_epoch=self._overlay_epoch,
                        end_session=self._failed_start_session_owner is bot,
                    )
                return
            if self._state == BotState.OFF and self._bot is None:
                return

            if self._state == BotState.STARTING:
                self._start_cancelled = True

            bot = self._bot
            if bot is not None:
                try:
                    bot.request_stop()
                except Exception as exc:
                    # Preserve ownership and STOPPING rather than reporting a
                    # successful stop when the runtime could not be signalled.
                    self._stopping = True
                    self._state = BotState.STOPPING
                    self._on_log(f"[STATE] Stop request failed: {exc}")
                    self._emit_state(BotState.STOPPING)
                    return

            overlay_epoch = self._overlay_epoch
            # Keep ``_bot`` until the joiner proves full shutdown. This is the
            # retry handle if the runtime thread returned with workers live.
            self._stopping = bot is not None
            self._state = BotState.STOPPING if bot is not None else BotState.OFF
            self._hunt_overlay.reset_stats()
            if bot is None:
                self._hunt_overlay.destroy()
                self._emit_state(BotState.OFF)
            else:
                self._on_log(
                    "[STATE] Bot stopping — waiting for workers to exit"
                )
                self._emit_state(BotState.STOPPING)
                self._start_stop_joiner(bot, overlay_epoch=overlay_epoch)

    def _destroy_hunt_overlay_if_epoch(self, overlay_epoch: int) -> None:
        """Destroy overlay only if no newer hunt has claimed it."""
        if overlay_epoch != self._overlay_epoch:
            return
        if self._state != BotState.OFF:
            return
        self._hunt_overlay.destroy()

    def _start_stop_joiner(
        self,
        bot: BotController,
        *,
        destroy_overlay: bool = True,
        overlay_epoch: int | None = None,
        end_session: bool = False,
    ) -> None:
        epoch = self._overlay_epoch if overlay_epoch is None else overlay_epoch
        # Every path that hands a live bot to the joiner must claim stopping
        # ownership, including a cancelled STARTING generation that finishes
        # launching just after the user pressed Stop.
        with self._ownership_lock:
            self._stopping = True
            if self._state != BotState.STOPPING:
                self._state = BotState.STOPPING
                self._post_to_main(lambda: self._emit_state(BotState.STOPPING))
            existing = self._stop_joiner
            if existing is not None and existing.is_alive():
                return

        def _join() -> None:
            stopped = False
            # Each call has a bounded join timeout. After a finite number of
            # attempts, yield the joiner while retaining ownership and the bot
            # handle so Stop can retry later without overlapping runtimes.
            for _attempt in range(_STOP_RETRY_ATTEMPTS):
                try:
                    if bot.stop(join_timeout=DEFAULT_STOP_JOIN_TIMEOUT_S):
                        stopped = True
                        break
                except Exception as exc:
                    self._on_log(f"[STATE] Hunt stop retry failed: {exc}")
                self._on_log(
                    "[STATE] Hunt thread still alive after stop join — retrying"
                )

            if not stopped:
                self._on_log(
                    "[STATE] Hunt stop incomplete; ownership retained for retry"
                )
                # Keep STOPPING visible and keep ``_stopping`` true. A later
                # Stop call will create another bounded joiner.
                return

            # Mark ownership complete before posting UI work. Keep the visible
            # STOPPING state until Tk processes the transition; the start
            # guard also reconciles this state if a callback is delayed.
            with self._ownership_lock:
                self._stopping = False
                if self._bot is bot:
                    self._bot = None
            if end_session:
                self._close_failed_start_session(bot, "bot start cancelled")
            self._post_to_main(self._refresh_stopped_state)
            if destroy_overlay:
                self._post_to_main(
                    lambda: self._destroy_hunt_overlay_if_epoch(epoch),
                )

        # Reserve, assign, and start the joiner in one critical section. A
        # concurrent Stop/Exit caller therefore observes the reservation rather
        # than creating a second joiner for the same BotController. Thread.start
        # itself does not wait for the worker; bot.stop remains outside the lock.
        with self._ownership_lock:
            existing = self._stop_joiner
            if existing is not None and existing.is_alive():
                return
            joiner = threading.Thread(
                target=_join,
                name="bot-stop-joiner",
                daemon=True,
            )
            self._stop_joiner = joiner
            joiner.start()

    def _refresh_stopped_state(self) -> None:
        """Refresh the UI after the owned stop joiner has completed."""
        with self._ownership_lock:
            if self._state == BotState.STOPPING and not self._stopping:
                self._state = BotState.OFF
                self._emit_state(BotState.OFF)

    def _await_prior_stop_joiner(self) -> bool:
        """Wait for the previous hunt; return False if it remains alive.

        A start must never continue after this bounded wait if the old stop
        joiner is still alive. The joiner remains owned by the lifecycle
        manager, so a later Start can retry without overlapping worker sets.
        """
        with self._ownership_lock:
            joiner = self._stop_joiner
        if joiner is None or not joiner.is_alive():
            with self._ownership_lock:
                if self._stop_joiner is joiner:
                    self._stop_joiner = None
            return True

        # Keep the UI responsive by waiting on the start worker, but do not
        # silently proceed over a stop that has not completed.
        joiner.join(timeout=DEFAULT_STOP_JOIN_TIMEOUT_S * 4)
        if joiner.is_alive():
            self._on_log(
                "[STATE] Prior hunt stop still running — start remains blocked"
            )
            return False
        with self._ownership_lock:
            if self._stop_joiner is joiner:
                self._stop_joiner = None
        return True

    def pause(self) -> None:
        if self._state != BotState.RUNNING:
            return
        if self._bot is not None:
            self._bot.pause()
        self._state = BotState.PAUSED
        self._on_log("[STATE] Bot paused")
        self._session.write_focus_change("paused (focus lost)")
        self._emit_state(BotState.PAUSED)

    def resume(self) -> bool:
        if self._state != BotState.PAUSED:
            return False
        if self._bot is not None:
            resumed = self._bot.resume()
            if resumed is False:
                self._on_log(
                    "[STATE] Bot resume deferred — input is still unwinding"
                )
                return False
        self._state = BotState.RUNNING
        self._arm_focus_grace()
        self._on_log("[STATE] Bot resumed")
        self._session.write_focus_change("resumed")
        self._emit_state(BotState.RUNNING)
        self._root.after(300, self._poll_focus)
        return True

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
