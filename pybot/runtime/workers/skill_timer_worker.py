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

import traceback

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
        self._startup_generation: int | None = None
        self._startup_cycle_generation: int | None = None
        self._armed_generation: int | None = None
        self._startup_pressed: set[int] = set()
        self._startup_block_logged: set[tuple[int, int]] = set()

    def process_pending(self) -> bool:
        """Press due timers once; gameplay loop supplies all scheduling."""
        ctx = self._ctx
        timers = [t for t in ctx.config.skill_timers if t.scan_code and t.interval_ms > 0]
        if not timers or ctx.is_stopped() or not ctx.should_run_timers():
            return False
        # Startup buffs are a prerequisite, not merely a convention. The
        # gameplay owner may call this step immediately after a blocked buff
        # attempt, so fail closed until the milestone is actually published.
        startup_buffs = getattr(ctx, "startup_buffs_done", None)
        if startup_buffs is not None:
            is_set = getattr(startup_buffs, "is_set", None)
            if callable(is_set) and type(is_set()) is bool and not is_set():
                return False
        generation = int(getattr(ctx, "hunt_generation", 0))
        if self._startup_cycle_generation != generation:
            self._startup_pressed.clear()
            self._startup_cycle_generation = generation
            self._armed = False
        if not self._armed:
            self._arm_timers(timers)
            self._armed = True
        now = monotonic_ms()
        for timer in timers:
            if now - self._last_press_ms.get(timer.scan_code, 0) < timer.interval_ms:
                continue
            if not self._wait_stagger_gap():
                return False
            pressed = perform_if_allowed(
                self._input, ctx.should_run_timers,
                lambda code=timer.scan_code: self._input.teleport_key(code),
                lifecycle=ctx,
            )
            if pressed is False:
                continue
            pressed_at = monotonic_ms()
            self._last_press_ms[timer.scan_code] = pressed_at
            self._startup_pressed.add(timer.scan_code)
            ctx.character_action_gate.note_action(pressed_at)
            ctx.logger.behavior(
                f"[TIMER] key executed key={timer.button} scanCode={timer.scan_code}"
            )
        if {t.scan_code for t in timers}.issubset(self._startup_pressed):
            mark = getattr(ctx, "mark_startup_timers_done", None)
            if callable(mark):
                mark(expected_generation=generation)
            self._startup_generation = generation
        return True

    def run(self) -> None:
        ctx = self._ctx
        timers = [
            t
            for t in ctx.config.skill_timers
            if t.scan_code and t.interval_ms > 0
        ]
        if not timers:
            return

        for timer in timers:
            ctx.logger.behavior(
                f"[TIMER] started key={timer.button} interval={timer.interval_ms}ms "
                f"scanCode={timer.scan_code}"
            )
            self._last_press_ms[timer.scan_code] = -timer.interval_ms

        while not ctx.is_stopped():
            try:
                generation = getattr(ctx, "hunt_generation", 0)
                if not self._wait_for_startup_buffs(generation):
                    if ctx.is_stopped():
                        break
                    continue
                if not ctx.should_run_timers():
                    if self._armed:
                        self._armed = False
                        ctx.logger.behavior("[TIMER] paused (sit/pause/danger/teleport)")
                    # Discovery suspension and a queued danger request do not
                    # stop ordinary workers, so the general pause wait would
                    # return immediately and spin. Use a bounded stop wait for
                    # those transient safety gates; use the resume gate for
                    # user pause/sit as before.
                    if ctx.should_run_workers():
                        ctx.stop_event.wait(0.25)
                    else:
                        ctx.wait_while_stopped_or_paused(0.25)
                    continue

                generation = int(getattr(ctx, "hunt_generation", 0))
                if self._startup_cycle_generation != generation:
                    # A sit/stand transition creates a new hunt generation.
                    # Re-arm immediately so every configured timer fires once
                    # for the new hunt instead of waiting for its old interval.
                    self._startup_pressed.clear()
                    self._startup_block_logged.clear()
                    self._startup_cycle_generation = generation
                    self._armed = False

                now = monotonic_ms()
                if not self._armed:
                    # Teleport suspension pauses the worker without creating a
                    # new hunt. Preserve elapsed timer time across that pause;
                    # only a genuinely new hunt generation re-arms timers from
                    # zero for its startup cast.
                    if self._armed_generation != generation:
                        self._arm_timers(timers)
                        self._armed_generation = generation
                    self._armed = True
                    ctx.logger.behavior("[TIMER] armed (hunt running)")
                    now = monotonic_ms()

                due = [
                    timer
                    for timer in timers
                    if now - self._last_press_ms.get(timer.scan_code, 0)
                    >= timer.interval_ms
                ]
                startup_pending = self._startup_generation != generation
                for timer in due:
                    if int(getattr(ctx, "hunt_generation", 0)) != generation:
                        break
                    if not ctx.should_run_timers():
                        break
                    # A blocked startup cast must not burn the shared stagger
                    # slot: check admission before claiming so the gate stays
                    # free for buffs while area-clear/danger holds the timer.
                    if startup_pending and not self._startup_action_allowed():
                        block_key = (generation, timer.scan_code)
                        if block_key not in self._startup_block_logged:
                            self._startup_block_logged.add(block_key)
                            ctx.logger.behavior(
                                f"[TIMER] key blocked key={timer.button} "
                                f"scanCode={timer.scan_code} "
                                f"reason={self._startup_block_reason()}"
                            )
                        break
                    if not self._wait_stagger_gap():
                        break
                    if int(getattr(ctx, "hunt_generation", 0)) != generation:
                        break
                    if not ctx.should_run_timers():
                        break
                    pressed = perform_if_allowed(
                        self._input,
                        ctx.should_run_timers,
                        lambda: self._input.teleport_key(timer.scan_code),
                        lifecycle=ctx,
                    )
                    if pressed is False:
                        ctx.logger.behavior(
                            f"[TIMER] key rejected key={timer.button} "
                            f"scanCode={timer.scan_code}"
                        )
                        continue
                    ctx.logger.behavior(
                        f"[TIMER] key executed key={timer.button} "
                        f"scanCode={timer.scan_code}"
                    )
                    if int(getattr(ctx, "hunt_generation", 0)) != generation:
                        break
                    pressed_at = monotonic_ms()
                    self._last_press_ms[timer.scan_code] = pressed_at
                    # Record on the shared gate so buff casts and other timer
                    # presses wait out the stagger window from this keypress.
                    self._ctx.character_action_gate.note_action(pressed_at)
                    self._startup_pressed.add(timer.scan_code)

                # Release combat only after every normal timer has fired once
                # for this hunt generation. The generation changes when sit
                # ends, so the startup sequence repeats on the next hunt.
                generation = int(getattr(ctx, "hunt_generation", 0))
                startup_scans = {timer.scan_code for timer in timers}
                if (
                    self._startup_generation != generation
                    and startup_scans.issubset(self._startup_pressed)
                ):
                    mark_done = getattr(ctx, "mark_startup_timers_done", None)
                    if callable(mark_done):
                        completed = mark_done(expected_generation=generation)
                        if completed is False:
                            continue
                    self._startup_generation = generation

                now = monotonic_ms()
                next_wait_ms = 1000
                for timer in timers:
                    elapsed = now - self._last_press_ms.get(timer.scan_code, 0)
                    remaining = max(0, timer.interval_ms - elapsed)
                    next_wait_ms = min(next_wait_ms, remaining)

                ctx.stop_event.wait(max(0.05, next_wait_ms / 1000.0))
            except Exception:
                ctx.logger.behavior(f"[TIMER] tick error:\n{traceback.format_exc()}")
                # Bound repeated failures so a bad timer/input backend cannot
                # spin this daemon thread and flood the logger queue.
                if ctx.stop_event.wait(0.25):
                    break

    def _startup_block_reason(self) -> str:
        """Describe the first startup milestone currently blocking a timer."""
        ctx = self._ctx
        area = getattr(ctx, "startup_area_clear", None)
        if area is not None and not area.is_set():
            return "area_clear"
        buffs = getattr(ctx, "startup_buffs_done", None)
        if buffs is not None and not buffs.is_set():
            return "buffs_pending"
        timers = getattr(ctx, "startup_timers_done", None)
        if timers is not None and not timers.is_set():
            return "timers_pending"
        if not ctx.should_run_timers():
            return "lifecycle"
        return "danger_or_character_safety"

    def _startup_action_allowed(self) -> bool:
        checker = getattr(self._ctx, "should_run_startup_actions", None)
        if checker is None:
            return self._ctx.should_run_timers()
        return bool(checker())

    def _wait_for_startup_buffs(self, generation: int) -> bool:
        """Wait until character buffs finish before firing normal timers."""
        custom = getattr(self._ctx.config, "custom_behavior", None)
        buffs = getattr(custom, "buffs", ())
        if not buffs:
            mark_done = getattr(self._ctx, "mark_startup_buffs_done", None)
            if callable(mark_done):
                completed = mark_done(expected_generation=generation)
                if completed is False:
                    return False
            return True
        event = getattr(self._ctx, "startup_buffs_done", None)
        if event is None or event.is_set():
            return True
        while not self._ctx.is_stopped():
            if generation != getattr(self._ctx, "hunt_generation", generation):
                return False
            if not self._ctx.should_run_timers():
                # Teleport/danger suspension leaves ordinary workers runnable,
                # so the general pause wait would return immediately and spin.
                # User pause/sit still use the resume gate.
                should_run_workers = getattr(
                    self._ctx,
                    "should_run_workers",
                    None,
                )
                if callable(should_run_workers) and should_run_workers():
                    self._ctx.stop_event.wait(0.25)
                else:
                    self._ctx.wait_while_stopped_or_paused(0.25)
                continue
            if event.wait(0.05):
                return True
        return False

    def _arm_timers(self, timers) -> None:
        """Start timers for a new hunt so each configured key is due once."""
        for timer in timers:
            self._last_press_ms[timer.scan_code] = -timer.interval_ms

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
