"""Attack loop — simple round-robin with skill delay after each attack."""

from __future__ import annotations

import traceback

from pybot.game_state import PlayerVitals
from pybot.runtime.constants import (
    ATTACK_IDLE_SPIN_S,
    IDLE_DEAD_ATTACK_COUNT,
    IDLE_UNREACHABLE_ATTACK_COUNT,
    LOG_REPEAT_INTERVAL_MS,
    WORKER_POLL_INTERVAL_S,
)
from pybot.runtime.combat_observer import CombatObserver
from pybot.runtime.deferred_actions import DeferredActionScheduler
from pybot.runtime.hunt_tracks import monotonic_ms
from pybot.runtime.hunt_mode import HuntModeController
from pybot.runtime.input.input_backend import InputBackend, perform_if_allowed
from pybot.runtime.mob_behaviors import MobBehavior
from pybot.runtime.workers.worker_contexts import AttackLoopContext


class AttackLoop:
    def __init__(
        self,
        ctx: AttackLoopContext,
        hunt_mode: HuntModeController,
        input_backend: InputBackend,
        *,
        mob_behavior: MobBehavior | None = None,
        vitals: PlayerVitals | None = None,
        char_x: int = 0,
        char_y: int = 0,
    ) -> None:
        self._ctx = ctx
        self._hunt_mode = hunt_mode
        self._input = input_backend
        self._mob_behavior = MobBehavior() if mob_behavior is None else mob_behavior
        self._vitals = PlayerVitals() if vitals is None else vitals
        self._char_x = char_x
        self._char_y = char_y
        self._combat_observer = CombatObserver()
        self._last_sp_unknown_log_ms = 0

    def run(self) -> None:
        """Compatibility loop; production ownership is ``GameplayLoop``."""
        self._ctx.logger.behavior("[ATTACK] loop started")
        while not self._ctx.is_stopped():
            self.process_pending()

    def process_pending(self) -> bool:
        """Advance one deterministic combat decision/action step.

        This method deliberately performs no independent scheduling. The
        gameplay owner calls it after higher-priority danger/session steps.
        """
        try:
            if not self._ctx.should_run_combat():
                self._ctx.wait_while_combat_blocked(WORKER_POLL_INTERVAL_S)
                return False

            tick = monotonic_ms()
            policy_tracks = self._ctx.tracks.tracks_for_policy(tick)
            target_id = self._ctx.policy.select_target(policy_tracks, tick)
            if target_id:
                self._attempt_heal()
                self._attack_one(target_id, tick)
                self._ctx.stop_event.wait(ATTACK_IDLE_SPIN_S)
                return True

            self._attempt_heal()
            self._hunt_mode.on_no_attackable_targets()
            self._ctx.stop_event.wait(ATTACK_IDLE_SPIN_S)
            return False
        except Exception:
            self._ctx.logger.behavior(
                f"[ATTACK] CRASH:\n{traceback.format_exc()}"
            )
            return False


    def _attempt_heal(self) -> None:
        """Attempt one custom heal through the shared heal admission gate."""
        heal = getattr(self._mob_behavior, "heal_if_needed", None)
        if not callable(heal):
            return
        try:
            hx, hy = self._character_pos()
            allowed = getattr(
                self._ctx,
                "should_run_custom_heal_actions",
                getattr(
                    self._ctx,
                    "should_run_character_actions",
                    self._ctx.should_run_combat,
                ),
            )
            admit = getattr(type(self._ctx), "perform_heal_if_allowed", None)
            if callable(admit):
                self._ctx.perform_heal_if_allowed(
                    allowed,
                    lambda: bool(heal(hx, hy, self._input)),
                    cooldown_s=1.0,
                )
            else:
                perform_if_allowed(
                    self._input,
                    allowed,
                    lambda: bool(heal(hx, hy, self._input)),
                    lifecycle=self._ctx,
                )
        except Exception:
            self._ctx.logger.behavior(
                f"[HEAL] self-heal error:\\n{traceback.format_exc()}"
            )

    def _character_pos(self) -> tuple[int, int]:
        """Screen position used for the melee-range idle guard."""
        pos = self._ctx.character_screen_pos()
        if pos is None:
            return self._char_x, self._char_y
        return int(pos[0]), int(pos[1])

    def _wait_for_gameplay_delay(self, timeout_s: float) -> None:
        """Wait a gameplay delay without hiding an urgent critical request."""
        critical = getattr(self._ctx, "critical_danger_requested", None)
        is_set = getattr(critical, "is_set", None)
        if not callable(is_set) or type(is_set()) is not bool:
            self._ctx.stop_event.wait(timeout_s)
            return
        deadline = monotonic_ms() + int(timeout_s * 1000)
        while not self._ctx.is_stopped() and not critical.is_set():
            remaining_ms = deadline - monotonic_ms()
            if remaining_ms <= 0:
                return
            if self._ctx.stop_event.wait(min(0.05, remaining_ms / 1000.0)):
                return

    def _log_unknown_observation(
        self,
        target_id: int,
        result,
        *,
        pre_sp: int | None,
        post_sp: int | None,
        now_tick: int,
        post_observed_ms: int,
        sample_now_ms: int,
    ) -> None:
        """Throttle diagnostics for inconclusive post-attack evidence."""
        if now_tick - self._last_sp_unknown_log_ms < LOG_REPEAT_INTERVAL_MS:
            return
        self._last_sp_unknown_log_ms = now_tick
        age_text = ""
        if result.reason == "obs-stale":
            age_text = f" age_ms={sample_now_ms - post_observed_ms}"
        self._ctx.logger.behavior(
            f"[IDLE] path=sp-unknown id={target_id} reason={result.reason}"
            f"{age_text} pre_sp={pre_sp} post_sp={post_sp} "
            "— idle/death counters frozen"
        )

    def _attack_one(self, target_id: int, now_tick: int) -> None:
        ctx = self._ctx

        # Snapshot coords under the store lock.
        snap = ctx.tracks.snapshot_for_track(target_id, now_tick)
        if snap is None:
            return

        click_x, click_y = snap.x, snap.y
        char_x, char_y = self._character_pos()

        # A configured debuff is cast once for this stable track before its
        # first attack. Failed input leaves the flag unset so the next cycle
        # can retry instead of attacking an unprepared target.
        try:
            prepared = perform_if_allowed(
                self._input,
                ctx.should_run_combat,
                lambda: self._mob_behavior.prepare_target(
                    target_id,
                    click_x,
                    click_y,
                    self._input,
                    target_debuffed=getattr(snap, "debuff_applied", False),
                    mark_debuffed=lambda: ctx.tracks.mark_debuff_applied(target_id),
                ),
                lifecycle=ctx,
            )
        except Exception:
            ctx.logger.behavior(
                f"[ATTACK] custom target preparation error id={target_id}:\\n"
                f"{traceback.format_exc()}"
            )
            return
        if not prepared:
            return

        # Custom behavior runs before the attack. Healing is handled by the
        # outer loop before this method; this hook only prepares the target.
        # Its input methods are atomic, so the cursor cannot be interleaved with
        # another worker's self-cast or storage action.
        try:
            all_mobs = ctx.tracks.positions_snapshot()
            perform_if_allowed(
                self._input,
                ctx.should_run_combat,
                lambda: self._mob_behavior.before_attack(
                    char_x, char_y, self._input, all_mobs=all_mobs,
                ),
                lifecycle=ctx,
            )
        except Exception:
            ctx.logger.behavior(
                f"[ATTACK] custom pre-attack error id={target_id}:\n"
                f"{traceback.format_exc()}"
            )

        # Idle death: cheap cache samples around the configured skill delay.
        # Pacing is exactly skill_delay_ms (plus click) — no OCR / capture here.
        pre_sp, pre_obs_ms, pre_chg_ms = self._vitals.sp_sample()
        try:
            # Atomic target move + skill key + click. This prevents a periodic
            # self-buff or heal worker from stealing the cursor between move
            # and attack input.
            attack_started = perform_if_allowed(
                self._input,
                ctx.should_run_combat,
                lambda: self._input.skill_click_at(
                    ctx.config.skill_scan_code, click_x, click_y
                ),
                lifecycle=ctx,
            )
        except Exception:
            ctx.logger.behavior(
                f"[ATTACK] input error id={target_id}:\n{traceback.format_exc()}"
            )
            return
        if not attack_started:
            return

        # Start kiting immediately after the attack input, before the skill
        # delay. This gives movement the full delay window to take effect.
        try:
            kite_x, kite_y = self._character_pos()
            all_mobs = ctx.tracks.positions_snapshot()
            perform_if_allowed(
                self._input,
                ctx.should_run_combat,
                lambda: self._mob_behavior.kite_after_attack(
                    kite_x, kite_y, self._input, all_mobs=all_mobs,
                ),
                lifecycle=ctx,
            )
        except Exception:
            ctx.logger.behavior(
                f"[ATTACK] kite error id={target_id}:\n{traceback.format_exc()}"
            )

        # Sole inter-skill wait — game applies SP cost; UI may refresh vitals.
        # In production this is sliced when a real critical notification is
        # pending, returning control to GameplayLoop so escape wins the next
        # deterministic step instead of waiting out the full skill delay.
        self._wait_for_gameplay_delay(ctx.config.skill_delay_ms / 1000.0)

        # Sit/heal/pause may have claimed the gate during the skill delay.
        if not ctx.should_run_combat():
            return

        post_sp, post_obs_ms, post_chg_ms = self._vitals.sp_sample()
        sample_now = monotonic_ms()
        # Kiting may have moved the character during the skill delay; use the
        # fresh position when deciding whether the target is melee-guarded.
        idle_char_x, idle_char_y = self._character_pos()

        observation = self._combat_observer.classify_sp(
            pre_sp=pre_sp,
            post_sp=post_sp,
            pre_observed_ms=pre_obs_ms,
            post_observed_ms=post_obs_ms,
            pre_changed_ms=pre_chg_ms,
            post_changed_ms=post_chg_ms,
            sample_now_ms=sample_now,
        )
        if observation.was_idle is None:
            self._log_unknown_observation(
                target_id,
                observation,
                pre_sp=pre_sp,
                post_sp=post_sp,
                now_tick=now_tick,
                post_observed_ms=post_obs_ms,
                sample_now_ms=sample_now,
            )
        was_idle = observation.was_idle

        accessible = snap.was_accessible
        blob_stationary = snap.discovery_stationary
        moving = snap.moving
        idle_before = snap.idle_attack_count

        # sprite.grf: no death animations → idle-dead is meaningless, but
        # unreachable is about pathfinding and still matters.
        action, idle_count = ctx.tracks.evaluate_idle_attack(
            target_id,
            was_idle=was_idle,
            mob_x=click_x,
            mob_y=click_y,
            char_x=idle_char_x,
            char_y=idle_char_y,
            now_tick=now_tick,
        )
        if not ctx.config.use_sprite_grf and action == "dead":
            ctx.logger.behavior(
                f"[DEATH] path=idle-dead id={target_id} @{click_x},{click_y} "
                f"idle={idle_count} accessible={accessible} "
                f"blob_stationary={blob_stationary} moving={moving} "
                f"pre_sp={pre_sp} post_sp={post_sp} — track removed, death-site recorded"
            )
            return
        if action == "unreachable":
            ctx.logger.behavior(
                f"[DEATH] path=idle-unreachable id={target_id} @{click_x},{click_y} "
                f"idle={idle_count} accessible={accessible} "
                f"blob_stationary={blob_stationary} moving={moving} "
                f"pre_sp={pre_sp} post_sp={post_sp} — track removed, death-site recorded"
            )
            return
        if was_idle is True and idle_count > 0:
            ctx.logger.behavior(
                f"[IDLE] path=progress id={target_id} "
                f"idle={idle_count}/{IDLE_UNREACHABLE_ATTACK_COUNT} "
                f"(dead_at={IDLE_DEAD_ATTACK_COUNT} if accessible+stationary) "
                f"accessible={accessible} blob_stationary={blob_stationary} "
                f"moving={moving} idle_before={idle_before} "
                f"pre_sp={pre_sp} post_sp={post_sp}"
            )
        elif was_idle is False and not accessible:
            ctx.logger.behavior(
                f"[IDLE] path=first-hit id={target_id} "
                f"pre_sp={pre_sp} post_sp={post_sp} — SP spent, now accessible"
            )

        ctx.tracks.apply_attack_event(target_id, now_tick=now_tick)
        ctx.policy.note_attack_target(target_id)
        ctx.overlay.increment_attacks()
        ctx.logger.behavior(
            f"[ATTACK] id={target_id} @{click_x},{click_y} "
            f"mob_attacks={snap.attack_count + 1}"
        )


class GameplayLoop:
    """Single owner for gameplay decisions and character input."""

    def __init__(self, ctx, *, attack, critical, sit=None, storage=None,
                 hp_restore=None, buffs=None, timers=None) -> None:
        self._ctx = ctx
        self._attack = attack
        self._critical = critical
        self._sit = sit
        self._storage = storage
        self._hp_restore = hp_restore
        self._buffs = buffs
        self._timers = timers
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
                ready=lambda: bool(self._hp_restore.can_restore_now()),
                due_when=self._hp_restore.needs_restore,
                execute=self._hp_restore.process_pending,
                due_on_generation=False,
            )
        if self._buffs is not None:
            for buff in getattr(self._ctx.config.custom_behavior, "buffs", ()):
                if buff.scan_code > 0 and buff.delay_ms > 0:
                    self._scheduler.register(
                        f"buff:{buff.scan_code}",
                        interval_ms=buff.delay_ms,
                        priority=30,
                        ready=lambda: bool(self._ctx.should_run_character_actions()),
                        execute=lambda code=buff.scan_code: self._buffs.execute_buff(code),
                    )
        if self._timers is not None:
            for timer in getattr(self._ctx.config, "skill_timers", ()):
                if timer.scan_code and timer.interval_ms > 0:
                    self._scheduler.register(
                        f"timer:{timer.scan_code}",
                        interval_ms=timer.interval_ms,
                        priority=40,
                        ready=lambda: bool(self._ctx.should_run_timers()),
                        execute=lambda code=timer.scan_code: self._timers.execute_timer(code),
                    )

    def _prepare_deferred_actions(self, now_ms: int) -> None:
        """Reconcile generation/startup success with periodic deadlines."""
        generation = int(getattr(self._ctx, "hunt_generation", 0))
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
        generation = int(getattr(self._ctx, "hunt_generation", 0))
        if self._startup_seed_generation == generation:
            return
        buffs_done = getattr(self._ctx, "startup_buffs_done", None)
        timers_done = getattr(self._ctx, "startup_timers_done", None)
        if (
            hasattr(buffs_done, "is_set") and not buffs_done.is_set()
        ) or (
            hasattr(timers_done, "is_set") and not timers_done.is_set()
        ):
            return
        found = False
        if self._buffs is not None:
            for buff in getattr(self._ctx.config.custom_behavior, "buffs", ()):
                if buff.scan_code > 0 and buff.delay_ms > 0:
                    at = self._buffs.last_success_ms(buff.scan_code)
                    if at is not None:
                        action = self._scheduler.get(f"buff:{buff.scan_code}")
                        if action.last_executed_ms != at:
                            self._scheduler.seed_executed(f"buff:{buff.scan_code}", at_ms=at)
                        found = True
        if self._timers is not None:
            for timer in getattr(self._ctx.config, "skill_timers", ()):
                if timer.scan_code and timer.interval_ms > 0:
                    at = self._timers.last_success_ms(timer.scan_code)
                    if at is not None:
                        self._scheduler.seed_executed(f"timer:{timer.scan_code}", at_ms=at)
                        found = True
        if found or (self._buffs is None and self._timers is None):
            self._startup_seed_generation = generation

    def deferred_statuses(self, *, now_ms: int | None = None):
        """Expose scheduler state for diagnostics and focused tests."""
        return self._scheduler.statuses(now_ms=monotonic_ms() if now_ms is None else now_ms)

    def run(self) -> None:
        self._ctx.logger.behavior("[GAMEPLAY] loop started")
        while not self._ctx.is_stopped():
            try:
                if self._critical.process_pending():
                    continue
                if self._ctx.danger_escape_active.is_set():
                    continue
                if self._sit is not None and self._sit.process_pending():
                    continue
                if self._storage is not None and self._storage.process_pending():
                    continue

                now_ms = monotonic_ms()
                # Startup casts are a real execution phase. They run before
                # periodic due actions and their successful timestamps seed the
                # deferred deadlines below. A failed/unsafe startup stays
                # retryable and never resets a timer merely because it expired.
                if self._buffs is not None:
                    self._buffs.process_pending(startup_only=True)
                if self._timers is not None:
                    self._timers.process_pending(startup_only=True)
                # Startup callbacks may have succeeded on this same generation;
                # seed their real success timestamps before observing deadlines.
                self._prepare_deferred_actions(now_ms)
                self._seed_startup_successes()

                if self._hp_restore is not None and self._hp_restore.needs_restore():
                    self._scheduler.mark_pending("hp_restore")
                # The scheduler observes monotonic deadlines and drains all
                # safe actions in priority order. Failed actions remain pending;
                # only successful callbacks restart their own deadline.
                ran = self._scheduler.run_pending(now_ms=monotonic_ms())
                # A due high-priority action that failed or is not yet safe must
                # remain the next gameplay concern; do not attack around it.
                if self._scheduler.requires_retry(max_priority=40):
                    continue
                self._attack.process_pending()
            except Exception:
                # The gameplay owner is the runtime's last safety boundary.
                # One malformed action must be logged and retried, not kill
                # the only thread that sequences all character input.
                self._ctx.logger.behavior(
                    f"[GAMEPLAY] step error:\n{traceback.format_exc()}"
                )
                self._ctx.stop_event.wait(WORKER_POLL_INTERVAL_S)
