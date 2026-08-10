"""HP item restoration when vision HP falls below the configured threshold."""

from __future__ import annotations

from pybot.game_state import PlayerVitals
from pybot.runtime.constants import HP_RESTORE_POLL_S, HP_RESTORE_RATIO
from pybot.runtime.input.input_backend import InputBackend
from pybot.runtime.workers.worker_contexts import HpRestoreWorkerContext


class HpRestoreWorker:
    """Press the configured HP item when HP is below 50%."""

    def __init__(
        self,
        ctx: HpRestoreWorkerContext,
        input_backend: InputBackend,
        vitals: PlayerVitals,
    ) -> None:
        self._ctx = ctx
        self._input = input_backend
        self._vitals = vitals

    def run(self) -> None:
        ctx = self._ctx
        scan = int(ctx.config.hp_scan_code)
        if scan <= 0 or not str(getattr(ctx.config, "hp_button", "") or "").strip():
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

    def needs_restore(self) -> bool:
        """Return whether HP is below the item-use threshold."""
        ratio = self._hp_ratio()
        return ratio is not None and ratio < HP_RESTORE_RATIO

    def process_pending(self) -> bool:
        """Evaluate one item-heal step; the gameplay loop owns scheduling."""
        ctx = self._ctx
        scan = int(ctx.config.hp_scan_code)
        ratio = self._hp_ratio()
        if (
            scan <= 0
            or not str(getattr(ctx.config, "hp_button", "") or "").strip()
            or ratio is None
            or ratio >= HP_RESTORE_RATIO
        ):
            return False
        ctx.logger.behavior(
            f"[HP] item key={ctx.config.hp_button!r} ratio={ratio:.1%}"
        )
        # Item healing is deliberately simple and independent from skill-heal
        # admission. It is never a blocked-heal transition and never requests
        # a teleport: below 50% means press the configured item key.
        healed = self._press_if_still_needed(scan)
        return healed

    def _press_if_still_needed(self, scan: int) -> bool:
        """Recheck HP immediately before sending the item key."""
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
