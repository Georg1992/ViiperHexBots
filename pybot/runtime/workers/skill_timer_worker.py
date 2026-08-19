"""Periodic skill timer key presses — one worker for all configured timers.

Timers press the key only (no mouse click), same as teleport_key.
Paused while sitting/user-paused/teleporting. Teleport pause preserves
elapsed intervals; a new hunt generation (for example after sit recovery)
re-arms startup casts with ``SKILL_TIMER_STAGGER_MS`` staggering.
Storage sessions do not pause timers (combat only), so keys are not re-armed.

Timer presses share the :class:`~pybot.runtime.gate_controller.CharacterActionGate`
with character buff casts: buffs claim the slot first (a buff burst makes
timers yield), and a single stagger window spaces every keypress.
"""

from __future__ import annotations

from pybot.runtime.event_utils import event_is_set
from pybot.runtime.hunt_tracks import monotonic_ms
from pybot.runtime.input.input_backend import InputBackend, perform_if_allowed
from pybot.runtime.workers.worker_contexts import SkillTimerWorkerContext


class SkillTimerWorker:
    """Presses each skill-timer key at its own interval (key only, no click)."""

    def __init__(
        self,
        ctx: SkillTimerWorkerContext,
        input_backend: InputBackend,
    ) -> None:
        self._ctx = ctx
        self._input = input_backend
        self._last_press_ms: dict[int, int] = {}
        self._armed = False
        self._startup_cycle_generation: int | None = None
        self._startup_pressed: set[int] = set()
        self._startup_block_logged: set[tuple[int, int]] = set()

    def execute_timer(self, scan_code: int) -> bool:
        """Execute one scheduler-owned timer action.

        Deadline ownership lives in ``GameplayLoop``'s deferred scheduler;
        this method only performs the admitted input and records successful
        startup state for the existing lifecycle gates.
        """
        ctx = self._ctx
        timers = [
            timer for timer in ctx.config.skill_timers
            if timer.scan_code == scan_code and timer.button.strip() and timer.interval_ms > 0
        ]
        if not timers or ctx.is_stopped() or not ctx.should_run_timers():
            return False
        if event_is_set(getattr(ctx, "startup_buffs_done", None)) is False:
            return False
        generation = int(getattr(ctx, "hunt_generation", 0))
        if self._startup_cycle_generation != generation:
            self._startup_pressed.clear()
            self._startup_block_logged.clear()
            self._startup_cycle_generation = generation
            self._armed = False
        if not self._armed:
            self._arm_timers([
                timer for timer in ctx.config.skill_timers
                if timer.scan_code and timer.button.strip() and timer.interval_ms > 0
            ])
            self._armed = True
        timer = timers[0]
        if not self._wait_stagger_gap():
            return False
        pressed = perform_if_allowed(
            self._input,
            ctx.should_run_timers,
            lambda: self._input.teleport_key(timer.scan_code),
            lifecycle=ctx,
        )
        if pressed is False:
            return False
        pressed_at = monotonic_ms()
        self._last_press_ms[timer.scan_code] = pressed_at
        self._startup_pressed.add(timer.scan_code)
        ctx.character_action_gate.note_action(pressed_at)
        ctx.logger.behavior(
            f"[TIMER] key executed key={timer.button} scanCode={timer.scan_code}"
        )
        startup_scans = {
            item.scan_code for item in ctx.config.skill_timers
            if item.scan_code and item.button.strip() and item.interval_ms > 0
        }
        if startup_scans.issubset(self._startup_pressed):
            mark = getattr(ctx, "mark_startup_timers_done", None)
            if callable(mark):
                mark(expected_generation=generation)
        return True

    def process_pending(self) -> bool:
        """Advance startup timers. Periodic casts are scheduler-owned."""
        ctx = self._ctx
        timers = [t for t in ctx.config.skill_timers if t.scan_code and t.button.strip() and t.interval_ms > 0]
        if not timers or ctx.is_stopped() or not ctx.should_run_timers():
            return False
        if self._critical_pending():
            return False
        # Startup buffs are a prerequisite, not merely a convention. The
        # gameplay owner may call this step immediately after a blocked buff
        # attempt, so fail closed until the milestone is actually published.
        if event_is_set(getattr(ctx, "startup_buffs_done", None)) is False:
            return False
        generation = int(getattr(ctx, "hunt_generation", 0))
        if self._startup_cycle_generation != generation:
            self._startup_pressed.clear()
            self._startup_cycle_generation = generation
            self._armed = False
        if self._critical_pending():
            return False
        missing = [timer for timer in timers if timer.scan_code not in self._startup_pressed]
        if not missing:
            return True
        return self.execute_timer(missing[0].scan_code)

    def last_success_ms(self, scan_code: int) -> int | None:
        """Return the last successful press timestamp for scheduler seeding."""
        return self._last_press_ms.get(scan_code)


    def _arm_timers(self, timers) -> None:
        """Start timers for a new hunt so each configured key is due once."""
        for timer in timers:
            self._last_press_ms[timer.scan_code] = -timer.interval_ms

    def _critical_pending(self) -> bool:
        """Return whether the pure danger observer requires preemption."""
        danger = getattr(self._ctx, "danger_detector", None)
        if danger is None:
            return False
        from pybot.runtime.danger_detector import DangerLevel
        return danger.danger_level() is DangerLevel.CRITICAL

    def _wait_stagger_gap(self) -> bool:
        """Ensure the shared buff/timer keypress slot is open before a press.

        Buffs win priority: while a buff burst is pending, timer presses
        yield even if the stagger window has elapsed. Otherwise waits out
        the shared ``SKILL_TIMER_STAGGER_MS`` window since the last buff cast
        or timer press. Returns False if hunt stopped/paused/sitting first.
        """
        ctx = self._ctx
        gate = ctx.character_action_gate
        while not ctx.is_stopped():
            if self._critical_pending():
                return False
            if not ctx.should_run_timers():
                return False
            now = monotonic_ms()
            if gate.try_claim(is_buff=False, now_ms=now):
                return True
            remaining_ms = gate.stagger_remaining_ms(now)
            if remaining_ms <= 0:
                # A buff burst holds the slot; poll until it clears.
                if ctx.stop_event.wait(0.05):
                    return False
                continue
            if ctx.stop_event.wait(remaining_ms / 1000.0):
                return False
        return False
