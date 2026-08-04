"""HP item restoration when vision HP falls below the configured threshold."""

from __future__ import annotations

import time
import traceback

from pybot.game_state import PlayerVitals
from pybot.runtime.constants import (
    HP_RESTORE_COOLDOWN_S,
    HP_RESTORE_POLL_S,
    HP_RESTORE_RATIO,
)
from pybot.runtime.input.input_backend import InputBackend, perform_if_allowed
from pybot.runtime.workers.worker_contexts import HpRestoreWorkerContext


class HpRestoreWorker:
    """Press the configured HP item key while HP is below 50%."""

    def __init__(
        self,
        ctx: HpRestoreWorkerContext,
        input_backend: InputBackend,
        vitals: PlayerVitals,
    ) -> None:
        self._ctx = ctx
        self._input = input_backend
        self._vitals = vitals
        self._last_press_mono = 0.0

    def run(self) -> None:
        ctx = self._ctx
        scan = int(ctx.config.hp_scan_code)
        if scan <= 0:
            return
        ctx.logger.behavior(
            f"[HP] item worker started key={ctx.config.hp_button!r} scanCode={scan} "
            f"threshold<{HP_RESTORE_RATIO:.0%}"
        )
        while not ctx.is_stopped():
            self.process_pending()
            # ``process_pending`` is deliberately non-blocking for the
            # deterministic gameplay owner. The compatibility run loop still
            # needs a bounded cadence so legacy callers cannot spin forever.
            ctx.stop_event.wait(HP_RESTORE_POLL_S)

    def process_pending(self) -> bool:
        """Evaluate one item-heal step; the gameplay loop owns scheduling."""
        ctx = self._ctx
        scan = int(ctx.config.hp_scan_code)
        if scan <= 0 or not ctx.should_run_workers():
            return False
        can_heal = getattr(ctx, "should_run_heal_actions", None)
        in_window = bool(can_heal()) if callable(can_heal) else False
        ratio = self._hp_ratio()
        if not in_window or ratio is None or ratio >= HP_RESTORE_RATIO:
            return False
        if time.monotonic() - self._last_press_mono < HP_RESTORE_COOLDOWN_S:
            return False
        ctx.logger.behavior(
            f"[HP] item key={ctx.config.hp_button!r} ratio={ratio:.1%}"
        )
        if hasattr(type(ctx), "perform_heal_if_allowed"):
            healed = bool(ctx.perform_heal_if_allowed(
                ctx.should_run_heal_actions,
                lambda: self._press_if_still_needed(scan),
                cooldown_s=HP_RESTORE_COOLDOWN_S,
            ))
        else:
            healed = perform_if_allowed(
                self._input, ctx.should_run_heal_actions,
                lambda: self._press_if_still_needed(scan), lifecycle=ctx,
            )
        if healed:
            self._last_press_mono = time.monotonic()
        return healed

    def _press_if_still_needed(self, scan: int) -> bool:
        """Recheck HP immediately before sending the item key.

        The worker performs a preflight ratio check before entering the shared
        admission gate. HP can become full between that check and the admitted
        action, so the input boundary must fail closed as well.
        """
        ratio = self._hp_ratio()
        if ratio is None or ratio >= HP_RESTORE_RATIO:
            return False
        return bool(self._input.key_tap(scan, after_s=0.0))

    def _hp_ratio(self) -> float | None:
        """Return HP/max HP, or None when the shared vitals are unavailable."""
        hp, hp_max = self._vitals.hp_pair()
        if hp is None or hp_max is None or hp_max <= 0:
            return None
        return hp / float(hp_max)
