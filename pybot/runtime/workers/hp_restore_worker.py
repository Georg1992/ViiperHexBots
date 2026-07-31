"""HP restore — item key below threshold, or heal skill until full HP.

Item path (``heal_skill`` off): press HP Restore Key when HP < ``HP_RESTORE_RATIO``.

Heal-skill path (``heal_skill`` on): when HP is not full and safe (or within
the post-teleport aggro-free window), pause combat and cast heal until full.
``end_heal_ops`` releases hunt immediately. Danger teleport always preempts
heal via ``discovery_suspend``.
"""

from __future__ import annotations

import time
import traceback

from pybot.game_state import PlayerVitals
from pybot.runtime.constants import (
    HP_HEAL_DAMAGE_QUIET_S,
    HP_HEAL_SKILL_POLL_S,
    HP_RESTORE_COOLDOWN_S,
    HP_RESTORE_POLL_S,
    HP_RESTORE_RATIO,
)
from pybot.runtime.danger_detector import DangerDetector
from pybot.runtime.input.input_backend import InputBackend
from pybot.runtime.workers.worker_contexts import HpRestoreWorkerContext


class HpRestoreWorker:
    """Restore HP via item key, or heal-until-full when heal skill is enabled."""

    def __init__(
        self,
        ctx: HpRestoreWorkerContext,
        input_backend: InputBackend,
        vitals: PlayerVitals,
        *,
        danger: DangerDetector,
    ) -> None:
        self._ctx = ctx
        self._input = input_backend
        self._vitals = vitals
        self._danger = danger
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
                f"healSkill=on (heal when HP not full and safe)"
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

    def _is_safe_to_heal(self) -> bool:
        """True when heal may cast.

        Mid-teleport (``discovery_suspend``) always blocks — urgent TP wins.
        After teleport settle, the post-TP window allows heal immediately.
        Otherwise require no nearby mobs and no recent HP drop.
        """
        if self._ctx.discovery_suspend.is_set():
            return False
        if self._ctx.in_post_teleport_heal_window():
            return True
        if self._danger.has_nearby_threat():
            return False
        if self._danger.has_recent_damage(HP_HEAL_DAMAGE_QUIET_S):
            return False
        return True

    def _hp_not_full(self) -> bool | None:
        """True if HP < max, False if full, None if unavailable."""
        hp, hp_max = self._vitals.hp_pair()
        if hp is None or hp_max is None or hp_max <= 0:
            return None
        return hp < hp_max

    def _run_heal_skill(self, scan: int) -> None:
        ctx = self._ctx
        while not ctx.is_stopped():
            try:
                if not ctx.should_run_workers():
                    ctx.wait_while_stopped_or_paused(HP_HEAL_SKILL_POLL_S)
                    continue
                if not self._is_safe_to_heal():
                    ctx.wait_unless_stopped(HP_HEAL_SKILL_POLL_S)
                    continue
                need = self._hp_not_full()
                if need is None or not need:
                    ctx.stop_event.wait(HP_HEAL_SKILL_POLL_S)
                    continue
                self._heal_until_full(scan)
            except Exception:
                ctx.logger.behavior(f"[HP] heal tick error:\n{traceback.format_exc()}")

    def _heal_until_full(self, scan: int) -> None:
        """Cast heal on self until HP is full, while safe / in post-TP window."""
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
                if ctx.pause_event.is_set() or not ctx.should_run_workers():
                    ctx.logger.behavior(
                        "[HP] heal-until-full aborted (pause/sit) — releasing"
                    )
                    return
                # Urgent TP (suspend) or danger outside post-TP window.
                if not self._is_safe_to_heal():
                    ctx.logger.behavior(
                        "[HP] heal-until-full aborted (danger) — releasing"
                    )
                    return
                hp, hp_max = self._vitals.hp_pair()
                if hp is None or hp_max is None or hp_max <= 0:
                    if not ctx.wait_unless_stopped(HP_HEAL_SKILL_POLL_S):
                        return
                    continue
                if hp >= hp_max:
                    ctx.logger.behavior(
                        f"[HP] heal-until-full done hp={hp}/{hp_max} — resuming hunt"
                    )
                    return
                pos = ctx.character_screen_pos()
                if pos is None:
                    if not ctx.wait_unless_stopped(HP_HEAL_SKILL_POLL_S):
                        return
                    continue
                if (
                    ctx.pause_event.is_set()
                    or not ctx.should_run_workers()
                    or not self._is_safe_to_heal()
                ):
                    continue
                cx, cy = int(pos[0]), int(pos[1])
                ctx.logger.behavior(
                    f"[HP] heal cast key={cfg.hp_button!r} "
                    f"at=({cx},{cy}) hp={hp}/{hp_max}"
                )
                self._input.skill_click_at(scan, cx, cy)
                # Urgent TP / pause abort the cast gap immediately.
                if not ctx.wait_unless_paused_or_suspended(delay_s):
                    if ctx.pause_event.is_set() or ctx.is_stopped():
                        ctx.logger.behavior(
                            "[HP] heal-until-full aborted (pause) — releasing"
                        )
                        return
                    ctx.logger.behavior(
                        "[HP] heal-until-full aborted (danger TP) — releasing"
                    )
                    return
                if not self._is_safe_to_heal():
                    ctx.logger.behavior(
                        "[HP] heal-until-full aborted (danger) — releasing"
                    )
                    return
        finally:
            ctx.end_heal_ops()

    def _hp_ratio(self) -> float | None:
        """HP / max from shared vitals, or None when unavailable."""
        hp, hp_max = self._vitals.hp_pair()
        if hp is None or hp_max is None or hp_max <= 0:
            return None
        return hp / float(hp_max)
