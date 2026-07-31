"""HP restore — item key below threshold, or heal skill until full HP.

Item path (``heal_skill`` off): press HP Restore Key when HP < ``HP_RESTORE_RATIO``.

Heal-skill path (``heal_skill`` on): when HP < ``HP_RESTORE_RATIO``, pause
combat and repeatedly:
  1. move cursor to the character sprite
  2. press HP Restore Key
  3. left-click
until HP is full. Danger teleport always preempts heal (``discovery_suspend``).
User pause aborts the heal-until-full session.
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
                f"(cursor→key→LMB until full HP)"
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
                    ctx.wait_unless_stopped(HP_RESTORE_POLL_S)
                    continue
                ratio = self._hp_ratio()
                if ratio is None or ratio >= HP_RESTORE_RATIO:
                    ctx.stop_event.wait(HP_RESTORE_POLL_S)
                    continue
                self._heal_until_full(scan)
            except Exception:
                ctx.logger.behavior(f"[HP] heal tick error:\n{traceback.format_exc()}")

    def _heal_until_full(self, scan: int) -> None:
        """Cast heal on self until HP is full. Triggered when HP < 50%."""
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
                # Pause (alt-tab) aborts heal — do not resume this session.
                if ctx.pause_event.is_set() or not ctx.should_run_workers():
                    ctx.logger.behavior(
                        "[HP] heal-until-full aborted (pause/sit) — releasing"
                    )
                    return
                # Danger teleport has priority — never cast during TP settle.
                if ctx.discovery_suspend.is_set():
                    if not ctx.wait_unless_stopped(HP_RESTORE_POLL_S):
                        ctx.logger.behavior(
                            "[HP] heal-until-full aborted (pause) — releasing"
                        )
                        return
                    continue
                hp, hp_max = self._vitals.hp_pair()
                if hp is None or hp_max is None or hp_max <= 0:
                    if not ctx.wait_unless_stopped(HP_RESTORE_POLL_S):
                        return
                    continue
                if hp >= hp_max:
                    ctx.logger.behavior(
                        f"[HP] heal-until-full done hp={hp}/{hp_max} — resuming hunt"
                    )
                    return
                pos = ctx.character_screen_pos()
                if pos is None:
                    if not ctx.wait_unless_stopped(HP_RESTORE_POLL_S):
                        return
                    continue
                # Re-check gates immediately before taking the input lock.
                if (
                    ctx.pause_event.is_set()
                    or not ctx.should_run_workers()
                    or ctx.discovery_suspend.is_set()
                ):
                    continue
                cx, cy = int(pos[0]), int(pos[1])
                ctx.logger.behavior(
                    f"[HP] heal cast key={cfg.hp_button!r} "
                    f"at=({cx},{cy}) hp={hp}/{hp_max}"
                )
                # Character sprite → HP Restore Key → left click (atomic).
                self._input.skill_click_at(scan, cx, cy)
                # Abort cast gap on pause or danger teleport settle.
                if not ctx.wait_unless_paused_or_suspended(delay_s):
                    if ctx.pause_event.is_set() or ctx.is_stopped():
                        ctx.logger.behavior(
                            "[HP] heal-until-full aborted (pause) — releasing"
                        )
                        return
                    # discovery_suspend — yield to danger TP, then continue.
                    continue
        finally:
            ctx.end_heal_ops()

    def _hp_ratio(self) -> float | None:
        """HP / max from shared vitals, or None when unavailable."""
        hp, hp_max = self._vitals.hp_pair()
        if hp is None or hp_max is None or hp_max <= 0:
            return None
        return hp / float(hp_max)
