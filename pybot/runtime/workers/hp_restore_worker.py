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
        cfg = ctx.config
        scan = int(cfg.hp_scan_code)
        if scan <= 0:
            return

        ctx.logger.behavior(
            f"[HP] item worker started key={cfg.hp_button!r} scanCode={scan} "
            f"threshold<{HP_RESTORE_RATIO:.0%}"
        )
        while not ctx.is_stopped():
            try:
                if not ctx.should_run_workers():
                    ctx.wait_while_stopped_or_paused(HP_RESTORE_POLL_S)
                    continue

                # HP-item healing is allowed only in the explicit post-
                # teleport safety window; never consume a heal item during a
                # live hunt/fight.
                can_heal = getattr(
                    ctx, "should_run_heal_actions", None
                )
                if callable(can_heal):
                    in_heal_window = bool(can_heal())
                else:
                    in_heal_window = bool(
                        getattr(ctx, "in_post_teleport_heal_window", lambda: False)()
                    ) and bool(ctx.should_run_workers())
                if not in_heal_window:
                    ctx.stop_event.wait(HP_RESTORE_POLL_S)
                    continue

                ratio = self._hp_ratio()
                if ratio is None or ratio >= HP_RESTORE_RATIO:
                    ctx.stop_event.wait(HP_RESTORE_POLL_S)
                    continue

                now = time.monotonic()
                if now - self._last_press_mono < HP_RESTORE_COOLDOWN_S:
                    ctx.stop_event.wait(HP_RESTORE_POLL_S)
                    continue

                ctx.logger.behavior(
                    f"[HP] item key={cfg.hp_button!r} ratio={ratio:.1%}"
                )
                # Recheck the same heal-specific admission predicate at the
                # input boundary. A danger/teleport transition may claim the
                # lifecycle after the preflight window check; the broad worker
                # gate alone would still allow a stale HP press.
                perform_heal = getattr(type(ctx), "perform_heal_if_allowed", None)
                if callable(perform_heal):
                    healed = bool(
                        ctx.perform_heal_if_allowed(
                            ctx.should_run_heal_actions,
                            lambda: self._input.key_tap(scan, after_s=0.0),
                            cooldown_s=HP_RESTORE_COOLDOWN_S,
                        )
                    )
                else:
                    healed = perform_if_allowed(
                        self._input,
                        ctx.should_run_heal_actions,
                        lambda: self._input.key_tap(scan, after_s=0.0),
                        lifecycle=ctx,
                    )
                if healed:
                    self._last_press_mono = time.monotonic()
                    ctx.stop_event.wait(HP_RESTORE_COOLDOWN_S)
                else:
                    ctx.stop_event.wait(HP_RESTORE_POLL_S)
            except Exception:
                ctx.logger.behavior(f"[HP] tick error:\n{traceback.format_exc()}")
                if ctx.stop_event.wait(0.25):
                    break

    def _hp_ratio(self) -> float | None:
        """Return HP/max HP, or None when the shared vitals are unavailable."""
        hp, hp_max = self._vitals.hp_pair()
        if hp is None or hp_max is None or hp_max <= 0:
            return None
        return hp / float(hp_max)
