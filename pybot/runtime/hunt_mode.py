"""Hunt mode controller — thin delegator to strategy implementations.

Modes (teleport / hybrid / walk) are implemented as separate strategy classes.
``teleport`` is the active area-changing mode, ``walk`` waits without
teleporting, and ``hybrid`` is intentionally a documented placeholder that
waits without teleporting until a product behavior is specified.
in :mod:`pybot.runtime.hunt_mode_strategies`.  The controller simply
forwards calls to the active strategy, keeping the OCP clean:
new modes require only a new strategy class, no controller changes.
"""

from __future__ import annotations

from pybot.runtime.hunt_mode_strategies import (
    HuntModeStrategy,
    HybridStrategy,
    TeleportStrategy,
    WalkStrategy,
)
from pybot.runtime.input.input_backend import InputBackend
from pybot.runtime.workers.worker_contexts import HuntModeControllerContext


class HuntModeController:
    """Thin delegator that forwards to a :class:`HuntModeStrategy`.

    Mode-specific behaviour (teleport, hybrid, walk) lives in strategy
    classes.  The controller preserves the stable public API that
    workers and tests depend on.
    """

    MODE_TELEPORT = "teleport"
    MODE_HYBRID = "hybrid"
    MODE_WALK = "walk"

    def __init__(self, strategy: HuntModeStrategy) -> None:
        self._strategy = strategy

    @property
    def discovery_since_reset(self) -> bool:
        return self._strategy.discovery_since_reset

    @property
    def discovery_confirmed_clear(self) -> bool:
        return self._strategy.discovery_confirmed_clear

    def on_area_reset(self) -> None:
        self._strategy.on_area_reset()

    def note_discovery_scan_completed(
        self, *, living_count: int, added_count: int, area_epoch: int
    ) -> None:
        self._strategy.note_discovery_scan_completed(
            living_count=living_count,
            added_count=added_count,
            area_epoch=area_epoch,
        )

    def note_discovery_scan_failed(self, reason: str) -> None:
        self._strategy.note_discovery_scan_failed(reason)

    def on_no_attackable_targets(self) -> bool:
        return self._strategy.on_no_attackable_targets()


# ── Factory ───────────────────────────────────────────────────────

def create_hunt_mode(
    ctx: HuntModeControllerContext,
    input_backend: InputBackend,
    teleport_controller=None,
) -> HuntModeController:
    """Create a HuntModeController wired to the appropriate strategy.

    Selects the strategy based on ``ctx.config.hunt_mode``:

    * ``"teleport"`` → :class:`TeleportStrategy`
    * ``"hybrid"`` → :class:`HybridStrategy` (documented placeholder; waits)
    * ``"walk"`` → :class:`WalkStrategy`
    """
    mode = ctx.config.hunt_mode
    if mode == HuntModeController.MODE_WALK:
        strategy: HuntModeStrategy = WalkStrategy(ctx, input_backend)
    elif mode == HuntModeController.MODE_HYBRID:
        strategy = HybridStrategy(ctx, input_backend)
    elif mode == HuntModeController.MODE_TELEPORT:
        strategy = TeleportStrategy(ctx, input_backend, teleport_controller)
    else:
        raise ValueError(f"unknown hunt mode: {mode!r}")
    return HuntModeController(strategy)
