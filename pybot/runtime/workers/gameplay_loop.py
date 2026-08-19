"""Single owner for gameplay decisions and character input."""

from __future__ import annotations

import threading
import time
import traceback

from pybot.runtime.constants import ATTACK_IDLE_SPIN_S, WORKER_POLL_INTERVAL_S
from pybot.runtime.danger_detector import DangerLevel
from pybot.runtime.deferred_actions import DeferredActionScheduler
from pybot.runtime.event_utils import event_is_set
from pybot.runtime.hunt_tracks import monotonic_ms


class GameplayLoop:
    """Single owner for gameplay decisions and character input."""

    def __init__(self, ctx, *, attack, sit=None, storage=None,
                 hp_restore=None, buffs=None, timers=None, teleport=None,
                 input_backend=None) -> None:
        self._ctx = ctx
        self._attack = attack
        self._sit = sit
        self._storage = storage
        self._hp_restore = hp_restore
        self._buffs = buffs
        self._timers = timers
        self._teleport = teleport
        self._input_backend = input_backend
        self._scheduler = DeferredActionScheduler()
        self._scheduler_generation: int | None = None
        self._startup_seed_generation: int | None = None
        self._register_deferred_actions()

    def _register_deferred_actions(self) -> None:
        """Register periodic actions without creating more control threads."""
        if self._hp_restore is not None:
            self._scheduler.register(
                "hp_restore",
                interval_ms=1000,
                priority=20,
                # Let process_pending observe and report a blocked admission;
                # suppressing it in ready() would hide the blocked state from
                # the deterministic gameplay owner.
                ready=lambda: bool(self._hp_restore.needs_restore()),
                due_when=self._hp_restore.needs_restore,
                execute=self._hp_restore.process_pending,
                due_on_generation=False,
            )
        if self._buffs is not None:
            for buff in self._ctx.config.custom_behavior.buffs:
                if buff.scan_code > 0 and buff.button.strip() and buff.delay_ms > 0:
                    self._scheduler.register(
                        f"buff:{buff.scan_code}",
                        interval_ms=buff.delay_ms,
                        priority=30,
                        ready=lambda: bool(self._ctx.should_run_character_actions()),
                        execute=lambda code=buff.scan_code: self._buffs.execute_buff(code),
                    )
        if self._timers is not None:
            for timer in self._ctx.config.skill_timers:
                if timer.scan_code and timer.button.strip() and timer.interval_ms > 0:
                    self._scheduler.register(
                        f"timer:{timer.scan_code}",
                        interval_ms=timer.interval_ms,
                        priority=40,
                        ready=lambda: bool(self._ctx.should_run_timers()),
                        execute=lambda code=timer.scan_code: self._timers.execute_timer(code),
                    )
        if self._storage is not None:
            self._scheduler.register(
                "storage",
                interval_ms=1000,
                priority=50,
                ready=lambda: bool(self._storage.can_execute_now()),
                due_when=self._storage.storage_due,
                execute=self._storage.process_pending,
            )

    def _prepare_deferred_actions(self, now_ms: int) -> None:
        """Reconcile generation/startup success with periodic deadlines."""
        generation = int(self._ctx.hunt_generation)
        if self._scheduler_generation == generation:
            return
        self._scheduler.sync_generation(generation, now_ms=now_ms)
        # Startup timestamps are collected after the startup callbacks have
        # actually completed. They are intentionally seeded once per
        # generation; periodic executions must never be copied back into the
        # scheduler on later ticks.
        self._scheduler_generation = generation
        self._startup_seed_generation = None

    def _seed_startup_successes(self) -> None:
        """Seed periodic schedules from successful startup casts exactly once."""
        generation = int(self._ctx.hunt_generation)
        if self._startup_seed_generation == generation:
            return
        if (
            event_is_set(self._ctx.startup_buffs_done) is False
            or event_is_set(self._ctx.startup_timers_done) is False
        ):
            return
        found = False
        if self._buffs is not None:
            for buff in self._ctx.config.custom_behavior.buffs:
                if buff.scan_code > 0 and buff.button.strip() and buff.delay_ms > 0:
                    at = self._buffs.last_success_ms(buff.scan_code)
                    if at is not None:
                        action = self._scheduler.get(f"buff:{buff.scan_code}")
                        if action.last_executed_ms != at:
                            self._scheduler.seed_executed(f"buff:{buff.scan_code}", at_ms=at)
                        found = True
        if self._timers is not None:
            for timer in self._ctx.config.skill_timers:
                if timer.scan_code and timer.button.strip() and timer.interval_ms > 0:
                    at = self._timers.last_success_ms(timer.scan_code)
                    if at is not None:
                        self._scheduler.seed_executed(f"timer:{timer.scan_code}", at_ms=at)
                        found = True
        if found or (self._buffs is None and self._timers is None):
            self._startup_seed_generation = generation

    def run(self) -> None:
        self._ctx.logger.behavior("[GAMEPLAY] loop started")
        # All character input, including urgent danger escape, is sequenced
        # here. HP observation only publishes facts; this owner performs the
        # complete danger transaction.
        while not self._ctx.is_stopped():
            try:
                if self._process_critical_danger():
                    continue
                if self._ctx.danger_escape_active.is_set():
                    # An urgent transition is already in progress. Park without
                    # busy-spinning so observation workers keep CPU time.
                    self._wait_for_gameplay_delay(WORKER_POLL_INTERVAL_S)
                    continue
                if self._sit is not None and self._sit.process_pending():
                    continue

                now_ms = monotonic_ms()
                # Startup casts are a real execution phase. They run before
                # periodic due actions and their successful timestamps seed the
                # deferred deadlines below. A failed/unsafe startup stays
                # retryable and never resets a timer merely because it expired.
                if self._buffs is not None:
                    self._buffs.process_pending()
                if self._danger_is_active():
                    continue
                if self._timers is not None:
                    self._timers.process_pending()
                if self._danger_is_active():
                    continue
                # Do not let the periodic scheduler observe generation-due
                # actions until startup has completed. Startup callbacks already
                # performed the first buff/timer presses; running the scheduler
                # before both milestones are published would replay a completed
                # buff while later startup timers are still being pressed.
                if (
                    event_is_set(self._ctx.startup_buffs_done) is False
                    or event_is_set(self._ctx.startup_timers_done) is False
                ):
                    # Startup actions have not completed. The periodic
                    # scheduler must stay parked (buffs/timers would replay),
                    # but a recovered hunt (danger escape, sit landing) keeps
                    # combat live before the first clear scan: the landing
                    # area may be populated, and attack must run to clear it
                    # instead of standing still until an empty scan releases
                    # the startup milestones. ``should_run_combat`` already
                    # admits the pre-clear window via the startup sequence.
                    if self._ctx.should_run_combat():
                        self._attack.process_pending()
                    self._wait_for_gameplay_delay(ATTACK_IDLE_SPIN_S)
                    continue
                # Startup callbacks may have succeeded on this same generation;
                # seed their real success timestamps before observing deadlines.
                self._prepare_deferred_actions(now_ms)
                self._seed_startup_successes()

                if self._hp_restore is not None and self._hp_restore.needs_restore():
                    self._scheduler.mark_pending("hp_restore")
                # The scheduler observes monotonic deadlines and drains all
                # safe actions in priority order. Failed actions remain pending;
                # only successful callbacks restart their own deadline.
                hp_action = None
                hp_before = None
                if self._hp_restore is not None:
                    hp_action = self._scheduler.get("hp_restore")
                    hp_before = hp_action.last_executed_ms
                self._scheduler.run_pending(now_ms=monotonic_ms())
                # A successful HP-item press gets this gameplay tick to itself;
                # do not immediately send an offensive key on the same stale
                # low-HP snapshot. The next tick rechecks the vitals.
                if (
                    hp_action is not None
                    and hp_action.last_executed_ms is not None
                    and hp_action.last_executed_ms != hp_before
                ):
                    continue
                # Item healing is maintenance, not a combat gate. Critical
                # danger remains a real gate and is handled independently.
                # AttackLoop owns only the skill-heal recovery state above.
                if self._scheduler.requires_retry(
                    max_priority=40,
                    ignore_keys={"hp_restore"},
                ):
                    # A due buff/timer may be intentionally unsafe during a
                    # teleport settle. Keep its deadline pending and give the
                    # independent UI/danger workers time to run; do not spin
                    # the gameplay owner at 100% CPU while waiting for landing.
                    self._wait_for_gameplay_delay(WORKER_POLL_INTERVAL_S)
                    continue
                self._attack.process_pending()
                if self._ctx.danger_wake.is_set():
                    self._ctx.danger_wake.clear()
            except Exception:
                # The gameplay owner is the runtime's last safety boundary.
                # One malformed action must be logged and retried, not kill
                # the only thread that sequences all character input.
                self._ctx.logger.behavior(
                    f"[GAMEPLAY] step error:\n{traceback.format_exc()}"
                )
                self._wait_for_gameplay_delay(WORKER_POLL_INTERVAL_S)

    def _wait_for_gameplay_delay(self, timeout_s: float) -> None:
        """Keep gameplay waits interruptible by danger and fresh tracks."""
        danger_wake = self._ctx.danger_wake
        attack_wake = self._ctx.attack_wake
        deadline = time.monotonic() + max(0.0, timeout_s)
        while event_is_set(self._ctx.stop_event) is not True:
            if event_is_set(danger_wake) is True:
                danger_wake.clear()
                return
            if event_is_set(attack_wake) is True:
                attack_wake.clear()
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return
            stop_event = self._ctx.stop_event
            stop_event.wait(min(0.05, remaining))
            if not isinstance(stop_event, threading.Event):
                return

    def _danger_is_active(self) -> bool:
        """Read the pure observer fact; no request event is consulted."""
        detector = self._ctx.danger_detector
        if detector is None:
            return False
        return detector.danger_level() is not DangerLevel.SAFE

    def _process_critical_danger(self) -> bool:
        """Run one clean danger transaction on the gameplay owner."""
        controller = self._ctx.danger_controller
        if controller is None:
            return False
        return bool(controller.process(seated=False))
