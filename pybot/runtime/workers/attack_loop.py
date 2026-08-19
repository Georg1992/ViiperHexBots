"""Attack loop — simple round-robin with skill delay after each attack."""

from __future__ import annotations

import threading
import time
import traceback
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pybot.runtime.teleport import TeleportController

from pybot.game_state import PlayerVitals
from pybot.runtime.constants import (
    ATTACK_IDLE_SPIN_S,
    IDLE_DEAD_ATTACK_COUNT,
    IDLE_UNREACHABLE_ATTACK_COUNT,
    HEAL_VERIFY_DELAY_MS,
    HP_RESTORE_COOLDOWN_S,
    LOG_REPEAT_INTERVAL_MS,
    MAX_ATTACK_COORD_AGE_MS,
    WORKER_POLL_INTERVAL_S,
)
from pybot.runtime.combat_observer import CombatObserver
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

        The producer-side ``attack_wake`` is only a low-latency hint; the
        shared track store remains authoritative and is sampled every step.
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
                self._wait_for_gameplay_delay(ATTACK_IDLE_SPIN_S)
                return False

            # Post-teleport skill recovery owns the first decision. It must run
            # before the ordinary combat gate: a blocked skill cast is the
            # evidence that this location cannot heal, and must trigger the
            # teleport immediately instead of waiting at the same spot.
            if self._post_teleport_hp_requires_heal():
                if not self._post_teleport_recovery_step():
                    self._wait_for_gameplay_delay(ATTACK_IDLE_SPIN_S)
                    return False

            try:
                attack_scan = int(self._ctx.config.skill_scan_code)
            except (AttributeError, TypeError, ValueError):
                attack_scan = 0
            raw_attack_button = getattr(self._ctx.config, "skill_button", None)
            if attack_scan <= 0 or not str(raw_attack_button or "").strip():
                return False

            if not self._ctx.should_run_combat():
                self._ctx.wait_while_combat_blocked(WORKER_POLL_INTERVAL_S)
                return False

            # After teleport, do not start fighting until the HP bar is full.
            # Give the configured heal the first opportunity, then yield this
            # tick even when the cast is blocked or unavailable. Normal hunting
            # HP thresholds do not apply outside this post-teleport window.
            tick = monotonic_ms()
            attack_wake = getattr(self._ctx, "attack_wake", None)
            # A wake can race with the first empty policy snapshot: tracking may
            # have committed a track just after this step read the store. Re-read
            # once immediately before considering an area-clear transition so a
            # fresh track is attacked in this same gameplay step instead of
            # waiting for another polling cycle (or risking a no-target teleport).
            for _ in range(2):
                policy_tracks = self._attackable_policy_tracks(
                    self._ctx.tracks.tracks_for_policy(tick),
                    tick,
                )
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
                    # Capture the selected track's area identity before any heal
                    # or other action can yield to a teleport/reset. Track IDs
                    # are intentionally reusable after an area reset, so the ID
                    # alone is not a safe target identity.
                    if selected_epoch is None:
                        self._attack_one(target_id, tick)
                    else:
                        self._attack_one(
                            target_id,
                            tick,
                            expected_epoch=selected_epoch,
                        )
                    self._wait_for_gameplay_delay(ATTACK_IDLE_SPIN_S)
                    return True
                if (
                    attack_wake is None
                    or not callable(getattr(attack_wake, "is_set", None))
                    or not attack_wake.is_set()
                ):
                    break
                attack_wake.clear()

            # One final authoritative read immediately before the clear-area
            # transition closes the normal wake race. The transition itself
            # owns its area/publication locks; do not hold them across attack or
            # teleport input.
            final_tick = monotonic_ms()
            final_tracks = self._attackable_policy_tracks(
                self._ctx.tracks.tracks_for_policy(final_tick),
                final_tick,
            )
            final_target = self._ctx.policy.select_target(
                final_tracks,
                final_tick,
            )
            if final_target:
                final_epoch = next(
                    (
                        int(track.area_epoch)
                        for track in final_tracks
                        if track.id == final_target
                        and type(getattr(track, "area_epoch", None)) is int
                    ),
                    None,
                )
                self._attack_one(
                    final_target,
                    monotonic_ms(),
                    expected_epoch=final_epoch,
                )
                self._wait_for_gameplay_delay(ATTACK_IDLE_SPIN_S)
                return True

            # A clear-area transition owns the next teleport. Do it before
            # attempting a skill heal so the required order is:
            # clear -> teleport -> inspect/heal -> hunt. The strategy performs
            # its own authoritative alive-track/discovery checks under the
            # transition boundary, so this call must remain the final decision
            # after the last track-store read above.
            transitioned = self._hunt_mode.on_no_attackable_targets()
            if transitioned is True:
                self._wait_for_gameplay_delay(ATTACK_IDLE_SPIN_S)
                return False
            # A newly committed track publishes attack_wake. Wait on that
            # producer signal instead of making attack discovery-bound or
            # burning a fixed polling delay before the first target is seen.
            self._wait_for_gameplay_delay(ATTACK_IDLE_SPIN_S)
            return False
        except Exception:
            self._ctx.logger.behavior(
                f"[ATTACK] CRASH:\n{traceback.format_exc()}"
            )
            return False


    def _post_teleport_hp_requires_heal(self) -> bool:
        """Return whether post-teleport combat must wait for a full HP bar."""
        try:
            heal_scan = int(self._ctx.config.custom_behavior.heal_scan_code)
        except (AttributeError, TypeError, ValueError):
            heal_scan = 0
        raw_heal_button = getattr(
            self._ctx.config.custom_behavior, "heal_button", None
        )
        heal_button = str(raw_heal_button or "").strip()
        # No configured custom heal means this recovery path is disabled. Do
        # not hold combat behind a full-HP gate that cannot perform an action.
        if heal_scan <= 0 or not heal_button:
            return False
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
        danger = getattr(self._ctx, "danger_detector", None)
        level = danger.danger_level() if danger is not None else None
        if isinstance(level, DangerLevel) and level is not DangerLevel.SAFE:
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
        raw_heal_button = getattr(
            self._ctx.config.custom_behavior, "heal_button", None
        )
        heal_button = str(raw_heal_button or "").strip()
        if scan_code <= 0 or not heal_button:
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
            # Keep the complete skill-heal cooldown before deciding that the
            # cast was stale and starting a retry teleport. The teleport is a
            # recovery action for the failed cast, so it must not happen during
            # the same 1.8-second heal window.
            if elapsed_ms < int(HP_RESTORE_COOLDOWN_S * 1000):
                return "waiting"
            if hp_changed_ms <= self._last_skill_heal_ms:
                return "blocked"

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
            # Start the cooldown from the completed skill input so the real
            # heal-to-heal interval cannot be shorter than the policy.
            self._last_skill_heal_ms = monotonic_ms()
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
        return bool(
            perform_if_allowed(
                self._input,
                lambda: self._ctx.should_run_combat(),
                lambda: self._ctx.tracks.perform_if_current(
                    target_id,
                    expected_epoch,
                    action,
                ),
                lifecycle=self._ctx,
            )
        )

    def _attackable_policy_tracks(self, tracks, now_tick: int):
        """Exclude held coordinates without deleting the live Track.

        A local tracking miss should not make an old coordinate actionable, but
        it also must not let one stale Track monopolize round-robin selection.
        Discovery/tracking still own liveness and recovery; this is only the
        attack-input freshness gate.
        """
        return [
            track
            for track in tracks
            if not (
                type(getattr(track, "last_found_tick", None)) is int
                and now_tick - track.last_found_tick > MAX_ATTACK_COORD_AGE_MS
            )
        ]

    def _character_pos(self) -> tuple[int, int]:
        """Screen position used for the melee-range idle guard."""
        pos = self._ctx.character_screen_pos()
        if pos is None:
            return self._char_x, self._char_y
        return int(pos[0]), int(pos[1])

    def _wait_for_gameplay_delay(self, timeout_s: float) -> None:
        """Wait without hiding danger or fresh-track producer wakes."""
        danger_wake = getattr(self._ctx, "danger_wake", None)
        attack_wake = getattr(self._ctx, "attack_wake", None)
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
            if event_is_set(stop_event) is True:
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
        try:
            attack_scan = int(ctx.config.skill_scan_code)
        except (AttributeError, TypeError, ValueError):
            attack_scan = 0
        raw_attack_button = getattr(ctx.config, "skill_button", None)
        attack_button = str(raw_attack_button or "").strip()
        # An empty attack binding disables combat input. Do not even snapshot
        # or prepare a target when the configured attack action cannot run.
        if attack_scan <= 0 or not attack_button:
            return

        # Snapshot coords under the store lock.
        snap = ctx.tracks.snapshot_for_track(target_id, now_tick)
        if snap is None:
            return

        # A held coordinate is not a fresh attack coordinate. Production
        # snapshots expose last_found_tick; lightweight compatibility fixtures
        # may not, so they retain their existing behavior.
        last_found_tick = getattr(snap, "last_found_tick", None)
        if type(last_found_tick) is int and now_tick - last_found_tick > MAX_ATTACK_COORD_AGE_MS:
            ctx.logger.behavior(
                f"[ATTACK] stale coordinate dropped id={target_id} "
                f"age_ms={now_tick - last_found_tick}"
            )
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

        # Idle death: cheap cache samples around the configured skill delay.
        # Pacing is exactly skill_delay_ms (plus click) — no OCR / capture here.
        pre_sp, pre_obs_ms, pre_chg_ms = self._vitals.sp_sample()
        try:
            # Atomic target move + skill key + click. This prevents a periodic
            # self-buff or heal worker from stealing the cursor between move
            # and attack input. Click at the freshest stored position so the
            # cursor lands on the sprite instead of where the mob was a moment
            # ago.
            def _click_freshest_target() -> bool:
                click_now = monotonic_ms()
                fresh = ctx.tracks.snapshot_for_track(target_id, click_now)
                if fresh is None:
                    return False
                fresh_last_found = getattr(fresh, "last_found_tick", None)
                if (
                    type(fresh_last_found) is int
                    and click_now - fresh_last_found > MAX_ATTACK_COORD_AGE_MS
                ):
                    ctx.logger.behavior(
                        f"[ATTACK] stale coordinate dropped id={target_id} "
                        f"age_ms={click_now - fresh_last_found} before click"
                    )
                    return False
                return self._input.skill_click_at(
                    attack_scan,
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

        ctx.tracks.apply_attack_event(target_id)
        ctx.policy.note_attack_target(target_id)
        ctx.overlay.increment_attacks()
        ctx.logger.behavior(
            f"[ATTACK] id={target_id} @{click_x},{click_y} "
            f"mob_attacks={snap.attack_count + 1}"
        )
