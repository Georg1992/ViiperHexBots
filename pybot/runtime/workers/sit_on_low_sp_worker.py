"""Sit when SP is low; pause hunting (and timers) until SP recovers.

Before sitting: teleport until discovery sees no mobs, then sit.
While sitting: if DangerDetector detects damage (HP drop), the sit session
returns ``"interrupted"`` so ``_recover_sp`` finds a new spot and sits again.
On SP recover: stand and resume.

SP comes from shared ``PlayerVitals``. Danger detection is handled by
``DangerDetector`` — the sit worker only checks a flag.
"""

from __future__ import annotations

import time

from pybot.game_state import PlayerVitals
from pybot.runtime.clear_area import HuntModeAreaReset, teleport_until_quiet
from pybot.runtime.constants import (
    SIT_LOW_SP_RATIO,
    SIT_POSE_SETTLE_S,
    SIT_RESUME_SP_RATIO,
    SIT_SP_POLL_INTERVAL_S,
    SIT_STAND_RESUME_DELAY_S,
)
from pybot.runtime.danger_detector import DangerDetector
from pybot.runtime.input.input_backend import InputBackend
from pybot.runtime.mob_behaviors import MobBehavior
from pybot.runtime.workers.worker_contexts import SitOnLowSpWorkerContext


class SitOnLowSpWorker:
    """When SP drops below 5%, clear the area, sit until SP ≥ 98%, then stand."""

    def __init__(
        self,
        ctx: SitOnLowSpWorkerContext,
        input_backend: InputBackend,
        hunt_mode: HuntModeAreaReset,
        *,
        danger: DangerDetector | None = None,
        mob_behavior: MobBehavior | None = None,
        vitals: PlayerVitals | None = None,
    ) -> None:
        self._ctx = ctx
        self._input = input_backend
        self._hunt_mode = hunt_mode
        self._mob_behavior = mob_behavior or MobBehavior()
        self._vitals = vitals or PlayerVitals()
        self._danger = danger or DangerDetector(
            ctx, input_backend, self._mob_behavior, vitals=self._vitals,
        )
        self._last_fail_log = ""

    def run(self) -> None:
        ctx = self._ctx
        ctx.logger.behavior(
            f"[SIT] worker started key={ctx.config.sit_on_low_sp_button} "
            f"scanCode={ctx.config.sit_on_low_sp_scan_code} "
            f"low<{SIT_LOW_SP_RATIO:.0%} resume>={SIT_RESUME_SP_RATIO:.0%}"
        )
        while not ctx.is_stopped():
            try:
                if ctx.pause_event.is_set():
                    ctx.wait_while_stopped_or_paused(SIT_SP_POLL_INTERVAL_S)
                    continue
                ratio = self._sp_ratio()
                if ratio is None:
                    ctx.stop_event.wait(SIT_SP_POLL_INTERVAL_S)
                    continue
                if ratio < SIT_LOW_SP_RATIO:
                    self._recover_sp(ratio)
                else:
                    ctx.stop_event.wait(SIT_SP_POLL_INTERVAL_S)
            except Exception:
                import traceback

                ctx.logger.behavior(f"[SIT] tick error:\n{traceback.format_exc()}")

    def _sp_snapshot(self) -> tuple[int, float] | None:
        """Return ``(sp, ratio)`` or None when SP is unavailable."""
        ctx = self._ctx
        sp, sp_max = self._vitals.sp_pair()
        if sp is None or sp_max is None or sp_max <= 0:
            reason = "sp_unavailable"
            if reason != self._last_fail_log:
                self._last_fail_log = reason
                ctx.logger.behavior(f"[SIT] SP read failed: {reason}")
            return None
        self._last_fail_log = ""
        return sp, sp / sp_max

    def _sp_ratio(self) -> float | None:
        snap = self._sp_snapshot()
        return None if snap is None else snap[1]

    def _recover_sp(self, low_ratio: float) -> None:
        ctx = self._ctx
        if not ctx.begin_sit_regen():
            return
        try:
            ctx.logger.behavior(
                f"[SIT] low SP ratio={low_ratio:.1%} — pausing hunt/timers, "
                "teleport until clear before sit"
            )
            while not ctx.is_stopped():
                # teleport_until_quiet scans the screen for a safe place
                # using the normal (creamy-first) teleport key.
                if not teleport_until_quiet(
                    ctx,
                    self._input,
                    self._hunt_mode,
                    log_tag="SIT",

                ):
                    return
                outcome = self._sit_session()
                if outcome == "recovered":
                    return
                if outcome is None:
                    return

                ctx.logger.behavior(
                    f"[SIT] {outcome} while regenerating — "
                    "finding another sit spot"
                )
        finally:
            ctx.end_sit_regen()
            ctx.discovery_wake.set()

    def _sit_session(self) -> str:
        """Sit until SP recovers.

        Returns:
            ``"recovered"`` — stood after SP ≥ resume threshold.
            ``"interrupted"`` — DangerDetector detected damage (HP drop while sitting).
        """
        ctx = self._ctx
        sit_scan = ctx.config.sit_on_low_sp_scan_code

        # Reset the flag so we only detect damage WHILE sitting.
        self._danger.pop_damage_detected()

        self._ensure_sitting(sit_scan)
        ctx.logger.behavior("[SIT] Pressed sit button, waiting for regen")

        while not ctx.is_stopped():
            if self._danger.pop_damage_detected():
                ctx.logger.behavior("[SIT] interrupted by damage — finding new spot")
                return "interrupted"

            sp_state = self._sp_snapshot()
            if sp_state is not None:
                sp, ratio = sp_state
                if ratio >= SIT_RESUME_SP_RATIO:
                    ctx.logger.behavior(
                        f"[SIT] SP recovered ratio={ratio:.1%} — standing"
                    )
                    self._ensure_standing(sit_scan)
                    ctx.wait_unless_stopped(SIT_STAND_RESUME_DELAY_S)
                    return "recovered"

            ctx.stop_event.wait(SIT_SP_POLL_INTERVAL_S)

    def _ensure_sitting(
        self,
        sit_scan: int,
    ) -> None:
        """Press sit once."""
        self._input.teleport_key(sit_scan)
        self._ctx.wait_unless_stopped(SIT_POSE_SETTLE_S)

    def _ensure_standing(
        self,
        sit_scan: int,
    ) -> None:
        """Press stand once."""
        self._input.teleport_key(sit_scan)
        self._ctx.wait_unless_stopped(SIT_POSE_SETTLE_S)
