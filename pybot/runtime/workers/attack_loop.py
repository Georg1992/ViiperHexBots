"""Attack loop — simple round-robin with skill delay after each attack."""

from __future__ import annotations

from pybot.runtime.hunt_tracks import monotonic_ms
from pybot.runtime.input.input_backend import InputBackend
from pybot.runtime.constants import WORKER_POLL_INTERVAL_S
from pybot.runtime.workers.worker_contexts import AttackLoopContext


class AttackLoop:
    def __init__(
        self,
        ctx: AttackLoopContext,
        hunt_mode,
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
                if not self._ctx.should_run_workers():
                    self._ctx.wait_while_stopped_or_paused(WORKER_POLL_INTERVAL_S)
                    continue

                tick = monotonic_ms()
                policy_tracks = self._ctx.tracks.tracks_for_policy(tick)
                self._ctx.policy.set_max_attacks(
                    self._ctx.tracks.max_attacks_per_mob_before_unreachable
                )

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

        # Snapshot coords under the store lock. The tracking thread mutates
        # (and may remove) the live MobTrack concurrently, so we must not read
        # a live reference outside the lock.
        snap = ctx.tracks.snapshot_for_track(target_id, now_tick)
        if snap is None:
            return

        click_x, click_y = snap.x, snap.y

        ctx.tracks.mark_attack_pending(target_id)

        # Move mouse and click skill – wrap in try/except so input
        # failures (Viiper connection, game window, etc.) don't kill
        # the entire attack loop thread.
        try:
            self._input.move_mouse(click_x, click_y)
            self._input.skill_click(ctx.config.skill_scan_code)
        except Exception as exc:
            ctx.tracks.clear_attack_pending(target_id)
            ctx.logger.behavior(
                f"[ATTACK] input error id={target_id}: {exc}"
            )
            return

        # Record attack and start cooldown
        still_tracked = ctx.tracks.apply_attack_event(target_id, now_tick=now_tick)
        ctx.policy.note_attack_target(target_id)
        self._last_attack_ms = now_tick
        ctx.overlay.increment_attacks()

        if still_tracked:
            ctx.logger.behavior(
                f"[ATTACK] id={target_id} @{click_x},{click_y} "
                f"mob_attacks={snap.attack_count + 1}"
            )
        else:
            limit = ctx.tracks.max_attacks_per_mob_before_unreachable
            avg = ctx.tracks.average_attacks_till_death
            ctx.logger.behavior(
                f"[ATTACK] id={target_id} @{click_x},{click_y} "
                f"unreachable after {snap.attack_count + 1} attacks "
                f"(max={limit} avg={avg:.1f} delay={ctx.config.skill_delay_ms}ms) "
                f"— tracking will drop on next tick"
            )
