"""Attack loop — simple round-robin with skill delay after each attack."""

from __future__ import annotations

from pybot.game_state import PlayerVitals
from pybot.runtime.hunt_mode import HuntModeController
from pybot.runtime.hunt_tracks import HuntTracks, monotonic_ms
from pybot.runtime.input.input_backend import InputBackend
from pybot.runtime.constants import LOG_REPEAT_INTERVAL_MS, WORKER_POLL_INTERVAL_S
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

    def _read_sp(self) -> int | None:
        """Current SP from shared ``PlayerVitals`` (UI memory/OCR publish)."""
        return self._vitals.sp

    def _attack_one(self, target_id: int, now_tick: int) -> None:
        ctx = self._ctx

        # Snapshot coords under the store lock.
        snap = ctx.tracks.snapshot_for_track(target_id, now_tick)
        if snap is None:
            return

        click_x, click_y = snap.x, snap.y

        # ── Idle-attack death detection ────────────────────────────────
        # Read SP before the attack, then again after the skill delay
        # so the game has processed SP consumption.  Measuring the delta
        # per-attack keeps each track's counter independent — other tracks'
        # SP consumption between attacks cannot contaminate the comparison.
        pre_sp = self._read_sp()
        try:
            self._input.move_mouse(click_x, click_y)
            self._input.skill_click(ctx.config.skill_scan_code)
        except Exception as exc:
            ctx.logger.behavior(
                f"[ATTACK] input error id={target_id}: {exc}"
            )
            return

        # Wait for the skill delay so the game updates SP post-consumption
        self._ctx.stop_event.wait(ctx.config.skill_delay_ms / 1000.0)
        post_sp = self._read_sp()

        # Idle requires two valid SP samples. Unreadable SP must not be
        # treated as a hit (that would reset the idle streak / fake accessibility).
        if pre_sp is None or post_sp is None:
            was_idle: bool | None = None
            # Throttle: SP unread freezes idle death; spam would drown other [DEATH] lines.
            if now_tick - self._last_sp_unknown_log_ms >= LOG_REPEAT_INTERVAL_MS:
                self._last_sp_unknown_log_ms = now_tick
                ctx.logger.behavior(
                    f"[DEATH] path=idle-sp-unknown id={target_id} "
                    f"pre_sp={pre_sp} post_sp={post_sp} — idle/death counters frozen"
                )
        else:
            was_idle = pre_sp == post_sp

        pre_track = ctx.tracks.get_track_by_id(target_id)
        accessible = bool(pre_track.was_accessible) if pre_track else False
        blob_stationary = bool(pre_track.discovery_stationary) if pre_track else False
        moving = bool(pre_track.moving) if pre_track else False
        idle_before = int(pre_track.idle_attack_count) if pre_track else 0

        action, idle_count = ctx.tracks.evaluate_idle_attack(
            target_id,
            was_idle=was_idle,
            mob_x=click_x,
            mob_y=click_y,
            char_x=self._char_x,
            char_y=self._char_y,
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
                f"pre_sp={pre_sp} post_sp={post_sp} — marked unreachable"
            )
            return
        if was_idle is True and idle_count > 0:
            ctx.logger.behavior(
                f"[DEATH] path=idle-progress id={target_id} "
                f"idle={idle_count}/{HuntTracks._IDLE_UNREACHABLE_THRESHOLD} "
                f"(dead_at={HuntTracks._IDLE_DEAD_THRESHOLD} if accessible+stationary) "
                f"accessible={accessible} blob_stationary={blob_stationary} "
                f"moving={moving} idle_before={idle_before} "
                f"pre_sp={pre_sp} post_sp={post_sp}"
            )
        elif was_idle is False and not accessible:
            ctx.logger.behavior(
                f"[DEATH] path=idle-first-hit id={target_id} "
                f"pre_sp={pre_sp} post_sp={post_sp} — SP spent, now accessible"
            )

        ctx.tracks.apply_attack_event(target_id, now_tick=now_tick)
        ctx.policy.note_attack_target(target_id)
        ctx.overlay.increment_attacks()
        ctx.logger.behavior(
            f"[ATTACK] id={target_id} @{click_x},{click_y} "
            f"mob_attacks={snap.attack_count + 1}"
        )
