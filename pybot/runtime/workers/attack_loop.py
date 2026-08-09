"""Attack loop — simple round-robin with skill delay after each attack."""

from __future__ import annotations

import traceback
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pybot.runtime.teleport import TeleportController

from pybot.game_state import PlayerVitals
from pybot.runtime.constants import (
    ATTACK_IDLE_SPIN_S,
    CRITICAL_PREEMPT_RELEASE_TIMEOUT_S,
    IDLE_DEAD_ATTACK_COUNT,
    IDLE_UNREACHABLE_ATTACK_COUNT,
    HEAL_VERIFY_DELAY_MS,
    HP_RESTORE_COOLDOWN_S,
    LOG_REPEAT_INTERVAL_MS,
    WORKER_POLL_INTERVAL_S,
)
from pybot.runtime.combat_observer import CombatObserver
from pybot.runtime.deferred_actions import DeferredActionScheduler
from pybot.runtime.event_utils import event_is_set
from pybot.runtime.hunt_tracks import monotonic_ms
from pybot.runtime.hunt_mode import HuntModeController
from pybot.runtime.input.input_backend import InputBackend, perform_if_allowed
from pybot.runtime.mob_behaviors import MobBehavior
from pybot.runtime.danger_detector import DangerLevel
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
        teleport_controller: TeleportController | None = None,
        char_x: int = 0,
        char_y: int = 0,
    ) -> None:
        self._ctx = ctx
        self._hunt_mode = hunt_mode
        self._input = input_backend
        self._mob_behavior = MobBehavior() if mob_behavior is None else mob_behavior
        self._vitals = PlayerVitals() if vitals is None else vitals
        self._teleport = teleport_controller
        self._last_skill_heal_ms: int | None = None
        self._skill_heal_retry_pending = False
        self._char_x = char_x
        self._char_y = char_y
        self._combat_observer = CombatObserver()
        self._last_sp_unknown_log_ms = 0
        self._last_low_hp_priority_log_ms = 0

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
            # A blocked/failed skill heal has already consumed this location's
            # one attempt. Do not cast again until its teleport succeeds.
            if self._skill_heal_retry_pending:
                # Keep retrying the transition, never the blocked skill. A
                # failed teleport must not strand the bot in a cast loop at
                # the same location; a successful teleport clears this state
                # and the next tick performs the one post-settle skill attempt.
                if self._retry_post_teleport_heal():
                    self._skill_heal_retry_pending = False
                self._ctx.stop_event.wait(ATTACK_IDLE_SPIN_S)
                return False

            # Post-teleport skill recovery owns the first decision. It must run
            # before the ordinary combat gate: a blocked skill cast is the
            # evidence that this location cannot heal, and must trigger the
            # teleport immediately instead of waiting at the same spot.
            if self._post_teleport_hp_requires_heal():
                if not self._post_teleport_recovery_step():
                    self._ctx.stop_event.wait(ATTACK_IDLE_SPIN_S)
                    return False

            if not self._ctx.should_run_combat():
                self._ctx.wait_while_combat_blocked(WORKER_POLL_INTERVAL_S)
                return False

            # After teleport, do not start fighting until the HP bar is full.
            # Give the configured heal the first opportunity, then yield this
            # tick even when the cast is blocked or unavailable. Normal hunting
            # HP thresholds do not apply outside this post-teleport window.
            tick = monotonic_ms()
            policy_tracks = self._ctx.tracks.tracks_for_policy(tick)
            target_id = self._ctx.policy.select_target(policy_tracks, tick)
            if target_id:
                selected_epoch = next(
                    (
                        int(track.area_epoch)
                        for track in policy_tracks
                        if track.id == target_id
                        and type(getattr(track, "area_epoch", None)) is int
                    ),
                    None,
                )
                # Capture the selected track's area identity before any heal or
                # other action can yield to a teleport/reset. Track IDs are
                # intentionally reusable after an area reset, so the ID alone
                # is not a safe target identity.
                if selected_epoch is None:
                    # Preserve the compatibility path for lightweight callers
                    # that do not expose area epochs; real MobTrack snapshots
                    # always carry one.
                    self._attack_one(target_id, tick)
                else:
                    self._attack_one(
                        target_id,
                        tick,
                        expected_epoch=selected_epoch,
                    )
                self._ctx.stop_event.wait(ATTACK_IDLE_SPIN_S)
                return True

            # A clear-area transition owns the next teleport. Do it before
            # attempting a skill heal so the required order is:
            # clear -> teleport -> inspect/heal -> hunt.
            transitioned = self._hunt_mode.on_no_attackable_targets()
            if transitioned is True:
                self._ctx.stop_event.wait(ATTACK_IDLE_SPIN_S)
                return False
            self._ctx.stop_event.wait(ATTACK_IDLE_SPIN_S)
            return False
        except Exception:
            self._ctx.logger.behavior(
                f"[ATTACK] CRASH:\n{traceback.format_exc()}"
            )
            return False


    def _post_teleport_hp_requires_heal(self) -> bool:
        """Return whether post-teleport combat must wait for a full HP bar."""
        in_window = getattr(self._ctx, "in_post_teleport_heal_window", None)
        if not callable(in_window) or not bool(in_window()):
            return False
        hp, hp_max = self._vitals.hp_pair()
        if hp is not None and hp_max is not None and hp_max > 0 and hp >= hp_max:
            clear = getattr(self._ctx, "clear_post_teleport_heal", None)
            if callable(clear):
                clear()
            return False
        # During post-teleport recovery, an unreadable HP bar is not proof of
        # full health. Fail closed until a valid full reading arrives.
        return True

    def _post_teleport_recovery_step(self) -> bool:
        """Run one deterministic post-teleport recovery transition.

        Returns ``True`` only when combat may continue. A successful cast yields
        until the next fresh HP sample. A blocked/failed cast is immediately
        rechecked; if HP is still incomplete, the teleport controller starts a
        fresh settled teleport before the next loop step.
        """
        hp, hp_max = self._vitals.hp_pair()
        if hp is not None and hp_max is not None and hp_max > 0 and hp >= hp_max:
            self._clear_post_teleport_heal()
            return True
        if hp is None or hp_max is None or hp_max <= 0:
            self._log_post_teleport_heal("hp unknown; recovery paused")
            return False

        result = self._attempt_heal()
        if result == "cast":
            self._log_post_teleport_heal("heal cast; waiting for HP recheck")
            return False
        if result in {"waiting", "cooldown"}:
            self._log_post_teleport_heal("waiting for fresh HP after heal")
            return False

        if result in {"blocked", "failed"}:
            # The old cast is no longer a pending cooldown once this
            # location has been rejected. Clear it before teleporting so the
            # post-settle retry is a fresh skill attempt.
            self._last_skill_heal_ms = None
            self._skill_heal_retry_pending = True
            # A skill heal that cannot execute must not be retried at the same
            # location. Teleport immediately; the next settled recovery tick
            # is the single retry point for this same skill-heal state machine.
            retried = self._retry_post_teleport_heal()
            if retried:
                self._skill_heal_retry_pending = False
            self._log_post_teleport_heal(
                f"heal {result}; "
                f"{'teleported for retry' if retried else 'teleport retry unavailable'}"
            )
        else:
            # No configured recovery skill means there is no deterministic way
            # to satisfy the post-teleport full-HP rule. Keep the recovery
            # marker active: combat must not resume on a known low-HP sample.
            self._log_post_teleport_heal("no skill configured; recovery remains active")
        return False

    def _clear_post_teleport_heal(self) -> None:
        clear = getattr(self._ctx, "clear_post_teleport_heal", None)
        if callable(clear):
            clear()

    def _retry_post_teleport_heal(self) -> bool:
        """Teleport again when post-teleport healing cannot be admitted/cast."""
        teleport = self._teleport
        retry = getattr(teleport, "retry_post_teleport_heal", None)
        if not callable(retry):
            return False
        critical = getattr(self._ctx, "critical_danger_requested", None)
        if event_is_set(critical) is True:
            return False
        # This is intentionally immediate. A blocked skill heal means the
        # current area/session cannot accept the cast; waiting here only
        # repeats the same blocked decision at the same location.
        try:
            retried = bool(retry())
        except Exception:
            self._ctx.logger.behavior(
                f"[HEAL] retry teleport error:\\n{traceback.format_exc()}"
            )
            return False
        if retried:
            self._ctx.logger.behavior(
                "[HEAL] post-teleport skill heal blocked; teleported to retry"
            )
        return retried

    def _log_post_teleport_heal(self, detail: str) -> None:
        """Throttle diagnostics for post-teleport full-HP priority."""
        now = monotonic_ms()
        if now - self._last_low_hp_priority_log_ms < LOG_REPEAT_INTERVAL_MS:
            return
        self._last_low_hp_priority_log_ms = now
        hp, hp_max = self._vitals.hp_pair()
        self._ctx.logger.behavior(
            f"[HEAL] post-teleport priority HP={hp}/{hp_max} "
            f"until-full: {detail}"
        )

    def _attempt_heal(self) -> str:
        """Attempt one skill heal synchronously in the hunt decision loop."""
        try:
            scan_code = int(self._ctx.config.custom_behavior.heal_scan_code)
        except (AttributeError, TypeError, ValueError):
            scan_code = 0
        if scan_code <= 0:
            return "unavailable"

        hp, hp_max, _observed_ms, hp_changed_ms = self._vitals.hp_sample()
        now_ms = monotonic_ms()
        if hp is None or hp_max is None or hp_max <= 0 or hp >= hp_max:
            return "not_needed"
        if self._last_skill_heal_ms is not None:
            elapsed_ms = now_ms - self._last_skill_heal_ms
            if elapsed_ms < HEAL_VERIFY_DELAY_MS:
                return "waiting"
            # One verification window is enough. If the HP publisher has not
            # reported any change by then, the skill did not take effect at
            # this location (blocked input/session), so recovery must teleport
            # instead of waiting indefinitely and standing still.
            if hp_changed_ms <= self._last_skill_heal_ms:
                return "blocked"
            if elapsed_ms < int(HP_RESTORE_COOLDOWN_S * 1000):
                return "waiting"

        def cast() -> bool:
            hx, hy = self._character_pos()
            return bool(self._input.skill_click_at(scan_code, hx, hy))

        try:
            result = self._ctx.try_heal_if_allowed(
                self._ctx.should_run_custom_heal_actions,
                cast,
            )
        except Exception:
            self._ctx.logger.behavior(
                f"[HEAL] skill cast error:\\n{traceback.format_exc()}"
            )
            return "failed"
        if result == "cast":
            self._last_skill_heal_ms = now_ms
            return "cast"
        if result in {"blocked", "cooldown", "failed"}:
            return result
        return "blocked"

    def _perform_target_input(
        self,
        target_id: int,
        expected_epoch: int | None,
        action,
    ) -> bool:
        """Admit target input atomically with the final track identity check."""
        # Enter the existing session/input ownership boundary first. The
        # track store then validates identity under its own lock, preserving
        # the runtime lock order used by teleport/reset paths.
        lifecycle_admit = getattr(type(self._ctx), "perform_input_if_allowed", None)
        if callable(lifecycle_admit):
            return bool(
                self._ctx.perform_input_if_allowed(
                    lambda: self._ctx.should_run_combat(),
                    lambda: self._ctx.tracks.perform_if_current(
                        target_id,
                        expected_epoch,
                        action,
                    ),
                )
            )
        admit = getattr(type(self._ctx.tracks), "perform_if_current", None)
        if callable(admit):
            return bool(
                self._ctx.tracks.perform_if_current(
                    target_id,
                    expected_epoch,
                    action,
                )
            )
        # Lightweight compatibility stores do not expose the atomic helper;
        # retain their existing gate path for tests/older integrations.
        return bool(
            perform_if_allowed(
                self._input,
                lambda: (
                    self._ctx.should_run_combat()
                    and self._target_is_current_compat(target_id, expected_epoch)
                ),
                action,
                lifecycle=self._ctx,
            )
        )

    def _target_is_current_compat(
        self,
        target_id: int,
        expected_epoch: int | None,
    ) -> bool:
        current = self._ctx.tracks.snapshot_for_track(target_id, monotonic_ms())
        if current is None:
            return False
        return (
            expected_epoch is None
            or getattr(current, "area_epoch", None) == expected_epoch
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
        # Lightweight/mock contexts have no real critical event; their legacy
        # behavior is a single blocking wait.
        if event_is_set(critical) is None:
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

    def _attack_one(
        self,
        target_id: int,
        now_tick: int,
        *,
        expected_epoch: int | None = None,
    ) -> None:
        ctx = self._ctx

        # Snapshot coords under the store lock.
        snap = ctx.tracks.snapshot_for_track(target_id, now_tick)
        if snap is None:
            return

        click_x, click_y = snap.x, snap.y
        snap_epoch = expected_epoch
        if snap_epoch is None:
            snap_epoch = getattr(snap, "area_epoch", None)
            if type(snap_epoch) is not int:
                snap_epoch = None
        elif getattr(snap, "area_epoch", None) != snap_epoch:
            ctx.logger.behavior(
                f"[ATTACK] stale target dropped id={target_id} before preparation"
            )
            return
        char_x, char_y = self._character_pos()

        # A configured debuff is cast once for this stable track before its
        # first attack. Failed input leaves the flag unset so the next cycle
        # can retry instead of attacking an unprepared target. It re-reads the
        # freshest position so the debuff click lands on the sprite, matching
        # the attack click.
        try:
            def _prepare_fresh_target() -> bool:
                fresh = ctx.tracks.snapshot_for_track(target_id, monotonic_ms())
                if fresh is None:
                    return False
                return self._mob_behavior.prepare_target(
                    target_id,
                    fresh.x,
                    fresh.y,
                    self._input,
                    target_debuffed=getattr(snap, "debuff_applied", False),
                    mark_debuffed=lambda: ctx.tracks.mark_debuff_applied(target_id),
                )

            prepared = self._perform_target_input(
                target_id,
                snap_epoch,
                _prepare_fresh_target,
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
            self._perform_target_input(
                target_id,
                snap_epoch,
                lambda: self._mob_behavior.before_attack(
                    char_x, char_y, self._input, all_mobs=all_mobs,
                ),
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
            # and attack input. Click at the freshest stored position: debuff,
            # pre-attack hooks and SP sampling run between the snapshot above
            # and this click, and a re-read under the store lock lands the
            # cursor on the sprite instead of where the mob was a moment ago.
            def _click_freshest_target() -> bool:
                fresh = ctx.tracks.snapshot_for_track(target_id, monotonic_ms())
                if fresh is None:
                    return False
                return self._input.skill_click_at(
                    ctx.config.skill_scan_code,
                    fresh.x,
                    fresh.y,
                )

            attack_started = self._perform_target_input(
                target_id,
                snap_epoch,
                _click_freshest_target,
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
        # Discovery may remove a target while its skill delay is in flight.
        # Do not apply idle/death bookkeeping, attack counters, or target
        # rotation to that stale track after it has disappeared.
        post_snap = ctx.tracks.snapshot_for_track(target_id, now_tick)
        if post_snap is None or (
            snap_epoch is not None
            and getattr(post_snap, "area_epoch", None) != snap_epoch
        ):
            ctx.logger.behavior(
                f"[ATTACK] stale target dropped id={target_id} after skill delay"
            )
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
        if (
            event_is_set(getattr(self._ctx, "startup_buffs_done", None)) is False
            or event_is_set(getattr(self._ctx, "startup_timers_done", None)) is False
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
        # All character input, including urgent danger escape, is sequenced
        # here. HP observation only publishes the request; it never owns input.
        while not self._ctx.is_stopped():
            try:
                if self._process_critical_danger():
                    continue
                if self._ctx.danger_escape_active.is_set():
                    # An urgent transition is already in progress. Park without
                    # busy-spinning so observation workers keep CPU time.
                    self._ctx.stop_event.wait(WORKER_POLL_INTERVAL_S)
                    continue
                if self._sit is not None and self._sit.process_pending():
                    continue

                now_ms = monotonic_ms()
                # Startup casts are a real execution phase. They run before
                # periodic due actions and their successful timestamps seed the
                # deferred deadlines below. A failed/unsafe startup stays
                # retryable and never resets a timer merely because it expired.
                if self._buffs is not None:
                    self._buffs.process_pending(startup_only=True)
                if self._ctx.critical_danger_requested.is_set():
                    continue
                if self._timers is not None:
                    self._timers.process_pending(startup_only=True)
                if self._ctx.critical_danger_requested.is_set():
                    continue
                # Do not let the periodic scheduler observe generation-due
                # actions until startup has completed. Startup callbacks already
                # performed the first buff/timer presses; running the scheduler
                # before both milestones are published would replay a completed
                # buff while later startup timers are still being pressed.
                startup_buffs_done = event_is_set(
                    getattr(self._ctx, "startup_buffs_done", None)
                )
                startup_timers_done = event_is_set(
                    getattr(self._ctx, "startup_timers_done", None)
                )
                if startup_buffs_done is False or startup_timers_done is False:
                    self._ctx.stop_event.wait(ATTACK_IDLE_SPIN_S)
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
                    self._ctx.stop_event.wait(WORKER_POLL_INTERVAL_S)
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

    def _process_critical_danger(self) -> bool:
        """Consume one urgent HP-danger signal on this gameplay owner.

        The HP observer only sets the request and cancels an in-flight input
        operation. This method is the sole path that turns the request into a
        teleport, so no second gameplay controller can compete for input.
        """
        ctx = self._ctx
        teleport = self._teleport
        if teleport is None or ctx.is_stopped() or ctx.pause_event.is_set():
            return False
        if not ctx.critical_danger_requested.is_set():
            return False

        # SP recovery owns seated danger; its synchronous recovery path handles
        # the escape without allowing this hunting path to tear down the sit
        # lifecycle mid-regen.
        if (
            ctx.sitting_event.is_set()
            and not ctx.critical_danger_escape_active.is_set()
        ):
            return False

        # Damage can age out before this loop consumes the request. Never
        # teleport for a stale critical signal.
        if ctx.danger_detector.danger_level() is not DangerLevel.CRITICAL:
            ctx.pop_critical_danger()
            ctx.pop_danger_sit_request()
            return True

        if not ctx.try_begin_critical_escape_ops(override=True):
            return False
        try:
            if not ctx.wait_for_preempted_session_release(
                CRITICAL_PREEMPT_RELEASE_TIMEOUT_S
            ):
                # Never press the teleport key through a storage/heal owner
                # that did not release in time. Keep the urgent request
                # pending; the next gameplay tick retries after the owner has
                # unwound. The finally below releases the temporary escape
                # claim so this failure cannot stall the runtime forever.
                ctx.request_critical_danger()
                return False
            prefer_safe_key = bool(ctx.preempted_sessions()[0])
            if ctx.pause_event.is_set():
                return False
            if not ctx.pop_critical_danger():
                return False
            ctx.pop_danger_sit_request()
            if ctx.pause_event.is_set():
                ctx.request_critical_danger()
                return False

            # The observer may have canceled the previous input operation. It
            # has unwound before this tick, so re-arm the backend before the
            # emergency teleport key is emitted.
            begin = getattr(self._input_backend, "begin_session", None)
            if callable(begin) and begin() is False:
                ctx.request_critical_danger()
                return False

            try:
                escape_kwargs = {"reason": "critical_hunt"}
                if prefer_safe_key:
                    escape_kwargs["prefer_safe_key"] = True
                escaped = bool(teleport.danger_teleport(**escape_kwargs))
            except Exception as exc:
                ctx.logger.behavior(
                    f"[DANGER] critical hunting teleport failed: {exc}"
                )
                escaped = False
            if escaped:
                ctx.logger.behavior("[DANGER] critical hunting escape succeeded")
                return True
            ctx.request_critical_danger()
            return False
        finally:
            ctx.end_critical_escape_ops()
