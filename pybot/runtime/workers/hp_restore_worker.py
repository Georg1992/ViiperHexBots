"""Press HP Restore Key when HP falls below the restore threshold.

Reads from shared ``PlayerVitals`` (UI publishes from status-panel OCR)
rather than doing its own OCR — clean architecture, one source of truth.
"""

from __future__ import annotations

import time
import traceback

from pybot.game_state import PlayerVitals
from pybot.runtime.constants import (
    HP_RESTORE_COOLDOWN_S,
    HP_RESTORE_POLL_S,
    HP_RESTORE_RATIO,
)
from pybot.runtime.input.input_backend import InputBackend
from pybot.runtime.workers.worker_contexts import HpRestoreWorkerContext


class HpRestoreWorker:
    """When HP < ``HP_RESTORE_RATIO``, press the HP Restore Key."""

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
        cfg = ctx.config
        scan = int(cfg.hp_scan_code)
        if scan <= 0:
            return
        ctx.logger.behavior(
            f"[HP] worker started key={cfg.hp_button!r} scanCode={scan} "
            f"threshold<{HP_RESTORE_RATIO:.0%} "
            f"healSkill={'on' if cfg.heal_skill else 'off'} (item path only)"
        )
        while not ctx.is_stopped():
            try:
                if not ctx.should_run_workers():
                    ctx.wait_while_stopped_or_paused(HP_RESTORE_POLL_S)
                    continue
                ratio = self._hp_ratio()
                if ratio is None:
                    ctx.stop_event.wait(HP_RESTORE_POLL_S)
                    continue
                if ratio >= HP_RESTORE_RATIO:
                    ctx.stop_event.wait(HP_RESTORE_POLL_S)
                    continue
                now = time.monotonic()
                if now - self._last_press_mono < HP_RESTORE_COOLDOWN_S:
                    ctx.stop_event.wait(HP_RESTORE_POLL_S)
                    continue
                ctx.logger.behavior(
                    f"[HP] restore key={cfg.hp_button!r} ratio={ratio:.1%}"
                )
                self._input.teleport_key(scan)
                self._last_press_mono = time.monotonic()
                ctx.stop_event.wait(HP_RESTORE_COOLDOWN_S)
            except Exception:
                ctx.logger.behavior(f"[HP] tick error:\n{traceback.format_exc()}")

    def _hp_ratio(self) -> float | None:
        """HP / max from shared vitals, or None when unavailable."""
        hp, hp_max = self._vitals.hp_pair()
        if hp is None or hp_max is None or hp_max <= 0:
            return None
        return hp / float(hp_max)
