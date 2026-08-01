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
from pybot.runtime.hunt_mode import HuntModeController
from pybot.runtime.hunt_tracks import monotonic_ms
from pybot.runtime.input.input_backend import InputBackend
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
        self._mob_behavior = mob_behavior or MobBehavior()
        self._vitals = vitals or PlayerVitals()
        self._char_x = char_x
        self._char_y = char_y
        self._combat_observer = CombatObserver()
        self._last_sp_unknown_log_ms = 0

    def run(self) -> None:
        self._ctx.logger.behavior("[ATTACK] loop started")
        while not self._ctx.is_stopped():
            try:
                if not self._ctx.should_run_combat():
                    self._ctx.wait_while_combat_blocked(WORKER_POLL_INTERVAL_S)
                    continue

                tick = monotonic_ms()
                policy_tracks = self._ctx.tracks.tracks_for_policy(tick)

                target_id = self._ctx.policy.select_target(policy_tracks, tick)
                if target_id:
                    # Safe self-heal is attempted every attack-loop tick. The
                    # behavior cooldown prevents duplicate casts when the
                    # pre-attack or post-kite hook already healed.
                    defer_heal = getattr(
                        self._mob_behavior,
                        "defers_heal_until_after_kite",
                        lambda: False,
                    )
                    if not defer_heal():
                        heal = getattr(self._mob_behavior, "heal_if_needed", None)
                        if callable(heal):
                            try:
                                hx, hy = self._character_pos()
                                heal(hx, hy, self._input)
                            except Exception:
                                self._ctx.logger.behavior(
                                    f"[HEAL] self-heal error:\\n{traceback.format_exc()}"
                                )
                    self._attack_one(target_id, tick)
                    self._ctx.stop_event.wait(ATTACK_IDLE_SPIN_S)
                    continue

                # Also heal while the area is temporarily empty.
                heal = getattr(self._mob_behavior, "heal_if_needed", None)
                if callable(heal):
                    try:
                        hx, hy = self._character_pos()
                        heal(hx, hy, self._input)
                    except Exception:
                        self._ctx.logger.behavior(
                            f"[HEAL] idle self-heal error:\\n{traceback.format_exc()}"
                        )

                self._hunt_mode.on_no_attackable_targets()
                self._ctx.stop_event.wait(ATTACK_IDLE_SPIN_S)
            except Exception:
                self._ctx.logger.behavior(
                    f"[ATTACK] CRASH:\n{traceback.format_exc()}"
                )
                break

    def _character_pos(self) -> tuple[int, int]:
        """Screen position used for the melee-range idle guard."""
        pos = self._ctx.character_screen_pos()
        if pos is None:
            return self._char_x, self._char_y
        return int(pos[0]), int(pos[1])

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
            prepared = self._mob_behavior.prepare_target(
                target_id,
                click_x,
                click_y,
                self._input,
                target_debuffed=getattr(snap, "debuff_applied", False),
                mark_debuffed=lambda: ctx.tracks.mark_debuff_applied(target_id),
            )
        except Exception:
            ctx.logger.behavior(
                f"[ATTACK] custom target preparation error id={target_id}:\\n"
                f"{traceback.format_exc()}"
            )
            return
        if not prepared:
            return

        # Custom behavior runs before the attack: kite, then safe self-heal.
        # Its input methods are atomic, so the cursor cannot be interleaved with
        # another worker's self-cast or storage action.
        try:
            all_mobs = ctx.tracks.positions_snapshot()
            self._mob_behavior.before_attack(
                char_x, char_y, self._input, all_mobs=all_mobs,
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
            attack_started = self._input.skill_click_at(
                ctx.config.skill_scan_code, click_x, click_y
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
        kite_x, kite_y = char_x, char_y
        kite_acted = False
        try:
            kite_x, kite_y = self._character_pos()
            all_mobs = ctx.tracks.positions_snapshot()
            kite_acted = bool(self._mob_behavior.kite_after_attack(
                kite_x, kite_y, self._input, all_mobs=all_mobs,
            ))
        except Exception:
            ctx.logger.behavior(
                f"[ATTACK] kite error id={target_id}:\n{traceback.format_exc()}"
            )

        # Retry safe self-healing only after a successful kite input. A
        # throttled/failed kite is not a valid healing window while a target
        # remains active; a successful kite's cooldown suppresses duplicates
        # when its behavior already healed internally.
        if kite_acted:
            heal = getattr(self._mob_behavior, "heal_if_needed", None)
            if callable(heal):
                try:
                    heal(kite_x, kite_y, self._input)
                except Exception:
                    ctx.logger.behavior(
                        f"[HEAL] post-kite fallback error id={target_id}:\n"
                        f"{traceback.format_exc()}"
                    )

        # Sole inter-skill wait — game applies SP cost; UI may refresh vitals.
        self._ctx.stop_event.wait(ctx.config.skill_delay_ms / 1000.0)

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
