"""Escape hunting characters only when the danger detector reports CRITICAL."""

from __future__ import annotations

import traceback

from pybot.runtime.constants import (
    CRITICAL_DANGER_POLL_INTERVAL_S,
    CRITICAL_PREEMPT_RELEASE_TIMEOUT_S,
)
from pybot.runtime.danger_detector import DangerLevel
from pybot.runtime.teleport import TeleportController


class CriticalDangerWorker:
    """Perform one urgent teleport for each critical hunting danger event.

    Critical danger is the highest-priority signal in the runtime: the escape
    claims the input boundary even while another session (sit recovery,
    storage UI, heal) is active, and it polls at the danger detector's
    cadence so a critical drop is answered within ~100ms.
    """

    def __init__(self, ctx, teleport: TeleportController) -> None:
        self._ctx = ctx
        self._teleport = teleport

    def process_pending(self) -> bool:
        """Handle one pending critical escape; return whether it succeeded."""
        ctx = self._ctx
        # Stop and focus-loss pause must not turn an interrupted teleport into
        # a retry loop or admit emergency input after shutdown begins. Leave the
        # request queued for a later resume when paused.
        if ctx.is_stopped() or ctx.pause_event.is_set():
            return False
        if not ctx.critical_danger_requested.is_set():
            return False
        # True SP-sit recovery owns seated danger on the gameplay thread.
        # Never steal that session: preemption clears sitting_event, sit
        # abandons mid-regen, and SP is often already above the low-SP start
        # threshold so recovery never restarts (hunt goes stale).
        sit_owned = (
            ctx.sitting_event.is_set()
            and not ctx.critical_danger_escape_active.is_set()
        )
        if sit_owned:
            return False
        # The character may have recovered (sit regeneration, a heal, or the
        # damage simply aged out) since the drop was observed. Never fire a
        # random-wing teleport on a stale request — consume it and let the
        # hunt continue; the safe-heal path tops HP back up. The mirrored
        # requests are popped between two level reads: a fresh critical hit
        # queued in that window is restored below instead of being lost.
        if ctx.danger_detector.danger_level() is not DangerLevel.CRITICAL:
            ctx.pop_critical_danger()
            ctx.pop_danger_sit_request()
            return True
        # Preempt storage/heal UI sessions, but not SP-sit recovery (above).
        if not ctx.try_begin_critical_escape_ops(override=True):
            return False
        ctx.wait_for_preempted_session_release(CRITICAL_PREEMPT_RELEASE_TIMEOUT_S)

        # A preempted sit session keeps the safe-key escape (creamy / save
        # point first, the same key the recovery session would use): the
        # character lands somewhere it can sit and finish SP recovery. The
        # random fly wing can drop it back next to mobs and cause a repeat
        # escape->sit loop. Standing hunting escapes keep the urgent wing.
        prefer_safe_key = bool(ctx.preempted_sessions()[0])
        try:
            # Re-check after claiming the gate because focus can be lost between
            # the initial check and the ownership transition.
            if ctx.pause_event.is_set():
                return False
            # Consume both mirrored requests before teleport. A seated session
            # cannot claim this path because it already holds the sit gate.
            if not ctx.pop_critical_danger():
                return False
            ctx.pop_danger_sit_request()
            # Pause may arrive while consuming the mirrored requests. Restore
            # the critical request without sending input during focus loss.
            if ctx.pause_event.is_set():
                ctx.request_critical_danger()
                return False
            escaped = False
            try:
                escape_kwargs = {"reason": "critical_hunt"}
                if prefer_safe_key:
                    escape_kwargs["prefer_safe_key"] = True
                escaped = bool(self._teleport.danger_teleport(**escape_kwargs))
            except Exception as exc:
                ctx.logger.behavior(
                    f"[DANGER] critical hunting teleport failed: {exc}"
                )
            if escaped:
                ctx.logger.behavior(
                    "[DANGER] critical hunting escape succeeded"
                )
                return True
            # Keep the event pending for a retry rather than resuming combat
            # after an unavailable or interrupted teleport.
            ctx.request_critical_danger()
            return False
        finally:
            ctx.end_critical_escape_ops()

    def run(self) -> None:
        ctx = self._ctx
        while not ctx.is_stopped():
            try:
                handled = self.process_pending()
            except Exception:
                # This thread is the last safety boundary. A transient backend,
                # detector, or lifecycle failure must not silently terminate
                # emergency protection; log it and retry on the next poll.
                try:
                    ctx.logger.behavior(
                        "[DANGER] critical worker tick failed — retrying:\\n"
                        f"{traceback.format_exc()}"
                    )
                except Exception:
                    pass
                handled = False
            if not handled:
                # Poll at the danger detector's cadence so a critical drop is
                # answered as fast as the detector can observe it.
                ctx.stop_event.wait(CRITICAL_DANGER_POLL_INTERVAL_S)
