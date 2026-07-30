"""Centralized danger detection — triggers danger teleport unconditionally.

Callers own their condition checks (surrounded, HP drop, near objects).
The :class:`DangerDetector` owns the *response*: when any danger fires,
:meth:`trigger` executes ``MobBehavior.execute_danger_teleport`` with the
wing-first key and returns ``True``.

This keeps danger escape logic in one place — add cooldowns, stats, or
pre/post hooks here without touching workers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pybot.runtime.input.input_backend import InputBackend
    from pybot.runtime.mob_behaviors import MobBehavior


class DangerDetector:
    """Stateless dispatcher — triggers danger teleport unconditionally.

    Usage::

        if some_danger_condition:
            if detector.trigger(ctx, hunt_mode, input, mob_behavior, reason="..."):
                return  # danger handled
    """

    def trigger(
        self,
        ctx,
        hunt_mode,
        input_backend: InputBackend,
        mob_behavior: MobBehavior,
        *,
        reason: str = "",
    ) -> bool:
        """Execute danger teleport.  Always returns ``True`` (danger handled).

        Callers should check their own conditions *before* calling this.
        """
        return mob_behavior.execute_danger_teleport(
            ctx, hunt_mode, input_backend, reason=reason,
        )
