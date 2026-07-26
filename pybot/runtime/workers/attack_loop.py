"""Attack loop — simple round-robin with skill delay after each attack."""

from __future__ import annotations

from pybot.config.clients import MemoryAddresses
from pybot.game_state import GameMemoryPoller
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
        *,
        poller: GameMemoryPoller | None = None,
        memory: MemoryAddresses | None = None,
        char_x: int = 0,
        char_y: int = 0,
    ) -> None:
        self._ctx = ctx
        self._hunt_mode = hunt_mode
        self._input = input_backend
        self._poller = poller or GameMemoryPoller()
        self._memory = memory or MemoryAddresses()
        self._char_x = char_x
        self._char_y = char_y
        self._sp_reading_available = self._memory.current_sp > 0 and self._memory.max_sp > 0

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

    def _read_sp(self) -> int:
        """Read current SP from game memory. Returns 0 if unavailable."""
        if not self._sp_reading_available:
            return 0
        try:
            snap = self._poller.read(self._ctx.config.hwnd, self._memory)
            if snap.ok and snap.sp is not None:
                return snap.sp
        except Exception:
            pass
        return 0

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
        if not self._sp_reading_available or pre_sp <= 0 or post_sp <= 0:
            was_idle: bool | None = None
        else:
            was_idle = pre_sp == post_sp
        action, idle_count = ctx.tracks.evaluate_idle_attack(
            target_id,
            was_idle=was_idle,
            mob_x=click_x,
            mob_y=click_y,
            char_x=self._char_x,
            char_y=self._char_y,
        )
        if action == "dead":
            ctx.logger.behavior(
                f"[ATTACK] idle-death id={target_id} @{click_x},{click_y} "
                f"sp={post_sp} — {idle_count} idle attacks, track removed"
            )
            return
        if action == "unreachable":
            ctx.logger.behavior(
                f"[ATTACK] idle-unreachable id={target_id} @{click_x},{click_y} "
                f"sp={post_sp} — {idle_count} idle attacks, track marked unreachable"
            )
            return
        if idle_count > 0:
            ctx.logger.behavior(
                f"[ATTACK] idle id={target_id} count={idle_count} "
                f"pre_sp={pre_sp} post_sp={post_sp}"
            )

        ctx.tracks.apply_attack_event(target_id, now_tick=now_tick)
        ctx.policy.note_attack_target(target_id)
        ctx.overlay.increment_attacks()
        ctx.logger.behavior(
            f"[ATTACK] id={target_id} @{click_x},{click_y} "
            f"mob_attacks={snap.attack_count + 1}"
        )
