"""Press HP Restore Key when vision HP falls below the restore threshold.

Enabled when ``hp_scan_code`` is set. Heal-skill path is reserved for later
(``heal_skill`` is ignored here).
"""

from __future__ import annotations

import time

from pybot.recognition.ui.status_panel import read_status_panel
from pybot.runtime.constants import (
    HP_RESTORE_COOLDOWN_S,
    HP_RESTORE_POLL_S,
    HP_RESTORE_RATIO,
)
from pybot.runtime.input.input_backend import InputBackend
from pybot.runtime.workers.worker_contexts import HpRestoreWorkerContext


class HpRestoreWorker:
    """When vision HP < ``HP_RESTORE_RATIO``, press the HP Restore Key."""

    def __init__(
        self,
        ctx: HpRestoreWorkerContext,
        input_backend: InputBackend,
    ) -> None:
        self._ctx = ctx
        self._input = input_backend
        self._last_press_mono = 0.0
        self._last_fail_log = ""

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
                import traceback

                ctx.logger.behavior(f"[HP] tick error:\n{traceback.format_exc()}")

    def _hp_ratio(self) -> float | None:
        """Vision HP / max, or None when the status panel cannot be read."""
        ctx = self._ctx
        if not ctx.capture.is_valid():
            return None
        frame = ctx.capture.capture_client()
        if frame is None or getattr(frame, "size", 0) == 0:
            return None
        values = read_status_panel(frame)
        if values is None or values.hp_max <= 0:
            reason = "status_panel_unavailable"
            if reason != self._last_fail_log:
                self._last_fail_log = reason
                ctx.logger.behavior(f"[HP] panel read failed: {reason}")
            return None
        self._last_fail_log = ""
        return values.hp / float(values.hp_max)
