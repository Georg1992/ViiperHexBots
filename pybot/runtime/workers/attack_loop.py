"""Attack loop — simple round-robin with skill delay after each attack."""

from __future__ import annotations

from pybot.game_state import PlayerVitals
from pybot.runtime.constants import (
    LOG_REPEAT_INTERVAL_MS,
    SP_IDLE_MAX_OBSERVATION_AGE_MS,
    WORKER_POLL_INTERVAL_S,
)
from pybot.runtime.hunt_mode import HuntModeController
from pybot.runtime.hunt_tracks import HuntTracks, monotonic_ms
from pybot.runtime.input.input_backend import InputBackend
from pybot.runtime.workers.worker_contexts import AttackLoopContext


class AttackLoop:
    def __init__(
        self,
        ctx: AttackLoopContext,
        hunt_mode: HuntModeController,
        input_backend: InputBackend,
        *,
        vitals: PlayerVitals | None = None,
        char_x: int = 0,
        char_y: int = 0,
    ) -> None:
        self._ctx = ctx
        self._hunt_mode = hunt_mode
        self._input = input_backend
        self._vitals = vitals or PlayerVitals()
        self._char_x = char_x
        self._char_y = char_y
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
                    self._attack_one(target_id, tick)
                    self._ctx.stop_event.wait(0.025)
                    continue

                self._hunt_mode.on_no_attackable_targets()
                self._ctx.stop_event.wait(0.025)
            except Exception:
                import traceback
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

    def _attack_one(self, target_id: int, now_tick: int) -> None:
        ctx = self._ctx

        # Snapshot coords under the store lock.
        snap = ctx.tracks.snapshot_for_track(target_id, now_tick)
        if snap is None:
            return

        click_x, click_y = snap.x, snap.y
        char_x, char_y = self._character_pos()

        # Idle death: cheap cache samples around the configured skill delay.
        # Pacing is exactly skill_delay_ms (plus click) — no OCR / capture here.
        pre_sp, pre_obs_ms, pre_chg_ms = self._vitals.sp_sample()
        try:
            self._input.move_mouse(click_x, click_y)
            self._input.skill_click(ctx.config.skill_scan_code)
        except Exception as exc:
            ctx.logger.behavior(
                f"[ATTACK] input error id={target_id}: {exc}"
            )
            return

        # Sole inter-skill wait — game applies SP cost; UI may refresh vitals.
        self._ctx.stop_event.wait(ctx.config.skill_delay_ms / 1000.0)
        post_sp, post_obs_ms, post_chg_ms = self._vitals.sp_sample()
        sample_now = monotonic_ms()

        # Hit: SP dropped after a fresh observation.
        # Idle: same SP, fresh observation, no value change since pre, and the
        # observation is recent (rejects early mid-wait republish of pre-cost SP).
        # Otherwise unknown — freeze idle/death counters.
        was_idle: bool | None
        if pre_sp is None or post_sp is None or post_obs_ms <= pre_obs_ms:
            was_idle = None
            if now_tick - self._last_sp_unknown_log_ms >= LOG_REPEAT_INTERVAL_MS:
                self._last_sp_unknown_log_ms = now_tick
                reason = (
                    "sp-unread"
                    if pre_sp is None or post_sp is None
                    else "vitals-stale"
                )
                ctx.logger.behavior(
                    f"[IDLE] path=sp-unknown id={target_id} reason={reason} "
                    f"pre_sp={pre_sp} post_sp={post_sp} — idle/death counters frozen"
                )
        elif post_sp < pre_sp:
            was_idle = False
        elif post_sp > pre_sp:
            was_idle = None
            if now_tick - self._last_sp_unknown_log_ms >= LOG_REPEAT_INTERVAL_MS:
                self._last_sp_unknown_log_ms = now_tick
                ctx.logger.behavior(
                    f"[IDLE] path=sp-unknown id={target_id} reason=sp-increased "
                    f"pre_sp={pre_sp} post_sp={post_sp} — idle/death counters frozen"
                )
        elif post_chg_ms > pre_chg_ms:
            # Value changed away and back within the window — inconclusive.
            was_idle = None
        elif sample_now - post_obs_ms > SP_IDLE_MAX_OBSERVATION_AGE_MS:
            was_idle = None
            if now_tick - self._last_sp_unknown_log_ms >= LOG_REPEAT_INTERVAL_MS:
                self._last_sp_unknown_log_ms = now_tick
                ctx.logger.behavior(
                    f"[IDLE] path=sp-unknown id={target_id} reason=obs-stale "
                    f"age_ms={sample_now - post_obs_ms} "
                    f"pre_sp={pre_sp} post_sp={post_sp} — idle/death counters frozen"
                )
        else:
            was_idle = True

        accessible = snap.was_accessible
        blob_stationary = snap.discovery_stationary
        moving = snap.moving
        idle_before = snap.idle_attack_count

        action, idle_count = ctx.tracks.evaluate_idle_attack(
            target_id,
            was_idle=was_idle,
            mob_x=click_x,
            mob_y=click_y,
            char_x=char_x,
            char_y=char_y,
            now_tick=now_tick,
        )
        if action == "dead":
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
                f"idle={idle_count}/{HuntTracks._IDLE_UNREACHABLE_THRESHOLD} "
                f"(dead_at={HuntTracks._IDLE_DEAD_THRESHOLD} if accessible+stationary) "
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
