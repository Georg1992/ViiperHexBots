"""HP restore — item key below threshold, or heal skill until full HP.

Item path (``heal_skill`` off): press HP Restore Key when HP < ``HP_RESTORE_RATIO``.

Heal-skill path (``heal_skill`` on): when HP < ``HP_RESTORE_RATIO``, pause
combat and cast heal on the character sprite (move cursor → skill button + LMB)
until HP is full. Yields while a teleport settle is in progress so danger
teleport can finish first.
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
    """Restore HP via item key, or heal-until-full when heal skill is enabled."""

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
        if cfg.heal_skill:
            ctx.logger.behavior(
                f"[HP] worker started key={cfg.hp_button!r} scanCode={scan} "
                f"healSkill=on threshold<{HP_RESTORE_RATIO:.0%} "
                f"(heal until full)"
            )
            self._run_heal_skill(scan)
            return
        ctx.logger.behavior(
            f"[HP] worker started key={cfg.hp_button!r} scanCode={scan} "
            f"threshold<{HP_RESTORE_RATIO:.0%} healSkill=off (item path)"
        )
        self._run_item_restore(scan)

    def _run_item_restore(self, scan: int) -> None:
        ctx = self._ctx
        cfg = ctx.config
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

    def _run_heal_skill(self, scan: int) -> None:
        ctx = self._ctx
        while not ctx.is_stopped():
            try:
                if not ctx.should_run_workers():
                    ctx.wait_while_stopped_or_paused(HP_RESTORE_POLL_S)
                    continue
                # Let in-flight danger/mode teleport finish before starting heal.
                if ctx.discovery_suspend.is_set():
                    ctx.stop_event.wait(HP_RESTORE_POLL_S)
                    continue
                ratio = self._hp_ratio()
                if ratio is None or ratio >= HP_RESTORE_RATIO:
                    ctx.stop_event.wait(HP_RESTORE_POLL_S)
                    continue
                self._heal_until_full(scan)
            except Exception:
                ctx.logger.behavior(f"[HP] heal tick error:\n{traceback.format_exc()}")

    def _heal_until_full(self, scan: int) -> None:
        ctx = self._ctx
        cfg = ctx.config
        if not ctx.begin_heal_ops():
            return
        try:
            ctx.logger.behavior(
                f"[HP] heal-until-full start key={cfg.hp_button!r}"
            )
            delay_s = float(cfg.skill_delay_ms) / 1000.0
            while not ctx.is_stopped():
                if not ctx.should_run_workers():
                    if not ctx.wait_while_stopped_or_paused(HP_RESTORE_POLL_S):
                        return
                    continue
                # Danger teleport has priority — wait out settle before casting.
                if ctx.discovery_suspend.is_set():
                    ctx.stop_event.wait(HP_RESTORE_POLL_S)
                    continue
                hp, hp_max = self._vitals.hp_pair()
                if hp is None or hp_max is None or hp_max <= 0:
                    ctx.stop_event.wait(HP_RESTORE_POLL_S)
                    continue
                if hp >= hp_max:
                    ctx.logger.behavior(
                        f"[HP] heal-until-full done hp={hp}/{hp_max}"
                    )
                    return
                pos = ctx.character_screen_pos()
                if pos is None:
                    ctx.stop_event.wait(HP_RESTORE_POLL_S)
                    continue
                cx, cy = pos
                ctx.logger.behavior(
                    f"[HP] heal cast key={cfg.hp_button!r} "
                    f"at=({cx},{cy}) hp={hp}/{hp_max}"
                )
                self._input.move_mouse(cx, cy)
                self._input.skill_click(scan)
                ctx.stop_event.wait(delay_s)
        finally:
            ctx.end_heal_ops()

    def _hp_ratio(self) -> float | None:
        """HP / max from shared vitals, or None when unavailable."""
        hp, hp_max = self._vitals.hp_pair()
        if hp is None or hp_max is None or hp_max <= 0:
            return None
        return hp / float(hp_max)
