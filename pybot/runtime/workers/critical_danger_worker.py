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
        if not ctx.try_begin_sit_ops():
            return False
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
            ctx.end_sit_ops()

    def run(self) -> None:
        ctx = self._ctx
        while not ctx.is_stopped():
            if not self.process_pending():
                ctx.stop_event.wait(SIT_SP_POLL_INTERVAL_S)
