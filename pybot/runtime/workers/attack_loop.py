"""Attack loop — simple round-robin with skill delay after each attack."""

from __future__ import annotations

from pybot.runtime.hunt_mode import HuntModeController
from pybot.runtime.hunt_tracks import monotonic_ms
from pybot.runtime.input.input_backend import InputBackend
from pybot.runtime.constants import WORKER_POLL_INTERVAL_S
from pybot.runtime.workers.worker_contexts import AttackLoopContext


class AttackLoop:
    def __init__(
        self,
        ctx: AttackLoopContext,
        hunt_mode: HuntModeController,
        input_backend: InputBackend,
    ) -> None:
        self._ctx = ctx
        self._hunt_mode = hunt_mode
        self._input = input_backend
        self._last_attack_ms = 0

    def run(self) -> None:
        self._ctx.logger.behavior("[ATTACK] loop started")
        while not self._ctx.is_stopped():
            try:
                if not self._ctx.should_run_combat():
                    self._ctx.wait_while_combat_blocked(WORKER_POLL_INTERVAL_S)
                    continue

                tick = monotonic_ms()
                policy_tracks = self._ctx.tracks.tracks_for_policy(tick)
        

                # Respect skill delay after each attack
                if self._is_on_cooldown(tick):
                    self._ctx.stop_event.wait(0.025)
                    continue

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

    def _is_on_cooldown(self, now_tick: int) -> bool:
        if not self._last_attack_ms:
            return False
        elapsed = now_tick - self._last_attack_ms
        return elapsed < self._ctx.config.skill_delay_ms

    def _attack_one(self, target_id: int, now_tick: int) -> None:
        ctx = self._ctx

        # Snapshot coords under the store lock.
        snap = ctx.tracks.snapshot_for_track(target_id, now_tick)
        if snap is None:
            return

        click_x, click_y = snap.x, snap.y

        try:
            self._input.move_mouse(click_x, click_y)
            self._input.skill_click(ctx.config.skill_scan_code)
        except Exception as exc:
            ctx.logger.behavior(
                f"[ATTACK] input error id={target_id}: {exc}"
            )
            return

        ctx.tracks.apply_attack_event(target_id, now_tick=now_tick)
        ctx.policy.note_attack_target(target_id)
        self._last_attack_ms = now_tick
        ctx.overlay.increment_attacks()
        ctx.logger.behavior(
            f"[ATTACK] id={target_id} @{click_x},{click_y} "
            f"mob_attacks={snap.attack_count + 1}"
        )
