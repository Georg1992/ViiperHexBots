"""Escape hunting characters only when the danger detector reports CRITICAL."""

from __future__ import annotations

from pybot.runtime.constants import SIT_SP_POLL_INTERVAL_S
from pybot.runtime.teleport import TeleportController


class CriticalDangerWorker:
    """Perform one urgent teleport for each critical hunting danger event."""

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
        if ctx.sitting_event.is_set() or ctx.storage_event.is_set():
            return False
        if not ctx.critical_danger_requested.is_set():
            return False
        claim_critical = getattr(ctx, "try_begin_critical_escape_ops", None)
        if callable(claim_critical):
            if not claim_critical():
                return False
        elif not ctx.try_begin_sit_ops():
            return False
        try:
            # Re-check after claiming the gate because focus can be lost between
            # the initial check and the ownership transition.
            if ctx.pause_event.is_set():
                return False
            # Compatibility contexts may not expose the atomic critical claim.
            # Claim the explicit escape phase before consuming the event.
            begin_escape = getattr(ctx, "begin_danger_escape", None)
            if not callable(claim_critical) and callable(begin_escape):
                if not begin_escape():
                    return False
            # Consume both mirrored requests before teleport. A seated session
            # cannot claim this path because it already holds the sit gate.
            if not ctx.pop_critical_danger():
                if not callable(claim_critical) and callable(begin_escape):
                    ctx.end_danger_escape()
                return False
            ctx.pop_danger_sit_request()
            # Pause may arrive while consuming the mirrored requests. Restore
            # the critical request without sending input during focus loss.
            if ctx.pause_event.is_set():
                ctx.request_critical_danger()
                return False
            escaped = False
            try:
                escaped = bool(
                    self._teleport.danger_teleport(reason="critical_hunt")
                )
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
            end_critical = getattr(ctx, "end_critical_escape_ops", None)
            if callable(end_critical):
                end_critical()
            else:
                end_escape = getattr(ctx, "end_danger_escape", None)
                if callable(end_escape):
                    end_escape()
                # Legacy contexts without end_critical_escape_ops release the
                # borrowed sit gate here. A random fly-wing escape landing is
                # not verified safe, so startup actions must wait for the
                # first discovery scan rather than trusting the area clear.
                ctx.end_sit_ops(trusted_clear=False)

    def run(self) -> None:
        ctx = self._ctx
        while not ctx.is_stopped():
            if not self.process_pending():
                ctx.stop_event.wait(SIT_SP_POLL_INTERVAL_S)
