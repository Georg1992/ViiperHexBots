"""Deposit inventory to storage when weight is high; restock fly wings.

Faithful port of AHK HexBots ``ItemsToStorage`` / ``GetFlyWings``:
image templates under ``assets/UI``, Alt+E inventory toggle, Alt+RMB deposit.

Weight and HP are read from ``PlayerVitals`` which is published by the UI
(either from status-panel OCR or process-memory polls).  The worker must
never re-OCR or re-read process memory.
"""

from __future__ import annotations

# Keep the historical timing imports/module patch points for storage tests and
# custom integrations; production waits use _wait below.
import threading
import time
import traceback

from pybot.game_state import PlayerVitals
from pybot.recognition.ui.inventory import (
    InventoryUiError,
    find_storage_wing,
    find_template,
    find_wings_in_use_grid,
    is_inventory_open,  # noqa: F401 - retained compatibility patch point
    require_inventory_panel,
    slot_contains_template as _slot_contains_template,  # noqa: F401 - compatibility
    slot_looks_empty,
)
from pybot.recognition.ui.inventory_automation import InventoryAutomation
from pybot.runtime.constants import (
    FLY_WING_WEIGHT,
    STORAGE_INV_COLS,
    STORAGE_INV_ROWS,
    STORAGE_UI_SETTLE_S,
    STORAGE_WEIGHT_MODIFIER_MIN,
    STORAGE_WEIGHT_POLL_INTERVAL_S,
    STORAGE_WING_AIM_SETTLE_S,
)
from pybot.runtime.input.input_backend import InputBackend
from pybot.runtime.input.scan_codes import key_name_to_scan_code
from pybot.runtime.teleport import TeleportController
from pybot.runtime.workers.worker_contexts import ItemsToStorageWorkerContext

# These names remain module-level compatibility patch points for storage tests
# and external integrations, even though InventoryAutomation owns production UI
# calls. Listing them explicitly also documents why static analyzers see no use.
__all__ = ["ItemsToStorageWorker", "is_inventory_open", "_slot_contains_template"]


class ItemsToStorageWorker:
    """When weight ≥ WeightModifier%, deposit / restock in a quiet area."""

    def __init__(
        self,
        ctx: ItemsToStorageWorkerContext,
        input_backend: InputBackend,
        teleport: TeleportController,
        *,
        vitals: PlayerVitals | None = None,
    ) -> None:
        self._ctx = ctx
        self._input = input_backend
        self._teleport = teleport
        self._vitals = vitals or PlayerVitals()
        self._ui = InventoryAutomation(ctx, input_backend)

    def _wait(self, seconds: float) -> None:
        """Wait for storage UI settling while honoring input cancellation."""
        wait = getattr(self._input, "wait_interruptible", None)
        # Real input backends expose the interruptible wait. Lightweight
        # custom doubles without it still retain the original UI-settle delay.
        if callable(wait):
            completed = wait(seconds)
        else:
            stop_event = getattr(self._ctx, "stop_event", None)
            if isinstance(stop_event, threading.Event):
                completed = not stop_event.wait(seconds)
            else:
                # Keep lightweight/test doubles patchable and preserve their
                # historical timing without treating MagicMock as stopped.
                time.sleep(seconds)
                completed = True
        stopped = self._ctx.is_stopped()
        if completed is False or stopped is True:
            raise InventoryUiError("storage input wait cancelled")

    def run(self) -> None:
        ctx = self._ctx
        cfg = ctx.config
        chain_keys = ",".join(step[0] for step in cfg.open_storage_steps)
        ctx.logger.behavior(
            f"[STORAGE] worker started chain=[{chain_keys}] "
            f"steps={len(cfg.open_storage_steps)} "
            f"weight>={cfg.weight_modifier}% "
            f"flyWings={cfg.take_fly_wings}"
        )
        while not ctx.is_stopped():
            try:
                if not cfg.open_storage_steps:
                    ctx.stop_event.wait(STORAGE_WEIGHT_POLL_INTERVAL_S)
                    continue
                if ctx.pause_event.is_set() or ctx.sitting_event.is_set():
                    ctx.wait_while_stopped_or_paused(STORAGE_WEIGHT_POLL_INTERVAL_S)
                    continue
                heavy = self._weight_over_threshold()
                need_wings = ctx.should_restock_fly_wings()
                dump_for_wings = need_wings and self._fly_wings_would_hit_threshold()
                dump = heavy or dump_for_wings
                if not dump and not need_wings:
                    ctx.stop_event.wait(STORAGE_WEIGHT_POLL_INTERVAL_S)
                    continue
                if not ctx.begin_storage_ops():
                    continue
                try:
                    if dump and need_wings:
                        reason = (
                            "weight high"
                            if heavy
                            else (
                                f"GetFlyWings would hit threshold "
                                f"(+{int(cfg.fly_wings_amount) * FLY_WING_WEIGHT}wt)"
                            )
                        )
                        ctx.logger.behavior(
                            f"[STORAGE] {reason} + wingcount=0 — "
                            "merged ItemsToStorage+GetFlyWings"
                        )
                    elif dump:
                        ctx.logger.behavior("[STORAGE] weight high — ItemsToStorage")
                    else:
                        ctx.logger.behavior("[STORAGE] wingcount=0 — GetFlyWings")
                    ctx.logger.behavior(
                        "[STORAGE] teleport until quiet before storage UI"
                    )
                    if not self._teleport.teleport_until_quiet(
                        log_tag="STORAGE"
                    ):
                        ctx.logger.behavior(
                            "[STORAGE] area clear aborted — skip storage session"
                        )
                        continue
                    self.storage_session(dump=dump, restock=need_wings)
                except InventoryUiError as exc:
                    ctx.logger.behavior(f"[STORAGE] UI miss: {exc}")
                except Exception:
                    ctx.logger.behavior(
                        f"[STORAGE] cycle error:\n{traceback.format_exc()}"
                    )
                finally:
                    ctx.end_storage_ops()
                    ctx.discovery_wake.set()
            except Exception:
                ctx.logger.behavior(f"[STORAGE] tick error:\n{traceback.format_exc()}")

    def _weight_threshold(self, weight_max: int) -> float:
        modifier = int(self._ctx.config.weight_modifier)
        return weight_max * modifier / 100.0

    def _weight_over_threshold(self) -> bool:
        ctx = self._ctx
        if int(ctx.config.weight_modifier) < STORAGE_WEIGHT_MODIFIER_MIN:
            return False
        weight, weight_max = self._vitals.weight_pair()
        if weight is None or weight_max is None or weight_max <= 0:
            return False
        return weight >= self._weight_threshold(weight_max)

    def _fly_wings_would_hit_threshold(self) -> bool:
        """True when restocking ``fly_wings_amount`` would reach the storage gate."""
        ctx = self._ctx
        if int(ctx.config.weight_modifier) < STORAGE_WEIGHT_MODIFIER_MIN:
            return False
        amount = int(ctx.config.fly_wings_amount)
        if amount <= 0:
            return False
        weight, weight_max = self._vitals.weight_pair()
        if weight is None or weight_max is None or weight_max <= 0:
            return False
        projected = weight + amount * FLY_WING_WEIGHT
        return projected >= self._weight_threshold(weight_max)

    # ── Scan helpers (use inventory_automation for low-level UI) ────

    def _scan_use_grid_wings(self) -> list[tuple[int, int, int, int]]:
        def scan() -> list[tuple[int, int, int, int]]:
            frame = self._ui.capture_client()
            panel = require_inventory_panel(frame)
            return find_wings_in_use_grid(frame, panel)

        return self._ui.recognize("Use-grid wing scan", scan)

    def _deposit_wings_from_use_grid(self) -> int:
        """Find each Use-tab fly wing, aim bottom-left, Alt+RMB into storage."""
        log = self._ctx.logger.behavior
        ox, oy = self._ui.client_origin()
        deposited = 0
        max_passes = STORAGE_INV_COLS * STORAGE_INV_ROWS
        for pass_i in range(max_passes):
            wings = self._scan_use_grid_wings()
            log(
                f"[STORAGE] GetFlyWings Use grid scan "
                f"pass={pass_i + 1} wings={len(wings)} deposited={deposited}"
            )
            if not wings:
                return deposited
            col, row, aim_x, aim_y = wings[0]
            log(
                f"[STORAGE] GetFlyWings move to wing slot "
                f"col={col} row={row} low-left ({aim_x},{aim_y})"
            )
            self._input.move_mouse(ox + aim_x, oy + aim_y)
            self._wait(STORAGE_WING_AIM_SETTLE_S)
            log(
                f"[STORAGE] GetFlyWings Alt+RMB deposit "
                f"col={col} row={row}"
            )
            self._ui.alt_rmb_deposit()
            deposited += 1
        raise InventoryUiError(
            f"Use-tab wing deposit did not clear after {max_passes} passes"
        )

    def _deposit_tab_grid(self, *, tab_label: str) -> None:
        """Deposit items from the current inventory tab.

        Move to the first slot once. If it is empty, we are done.
        Otherwise, keep sending Alt+Right Click until the slot clears.
        """
        log = self._ctx.logger.behavior
        ox, oy = self._ui.client_origin()
        guard = STORAGE_INV_COLS * STORAGE_INV_ROWS

        # Determine the first slot's center and aim position
        frame_init = self._ui.capture_client()
        panel_init = require_inventory_panel(frame_init)
        first_col, first_row = 0, 0
        first_cx, first_cy = panel_init.slot_center(first_col, first_row)
        first_ax, first_ay = panel_init.slot_aim(first_col, first_row)

        log(f"[STORAGE] Move to first slot of {tab_label} tab ({first_ax},{first_ay})")
        self._input.move_mouse(ox + first_ax, oy + first_ay)
        self._wait(STORAGE_WING_AIM_SETTLE_S)

        for pass_i in range(guard):
            frame = self._ui.capture_client()

            if slot_looks_empty(frame, first_cx, first_cy):
                log(f"[STORAGE] ItemsToStorage {tab_label} tab first slot empty — done")
                return

            log(f"[STORAGE] ItemsToStorage deposit {tab_label} item from first slot")
            self._ui.alt_rmb_deposit()
            # Let the slot clear/next item move into the first slot before next loop
            self._wait(STORAGE_UI_SETTLE_S)

        raise InventoryUiError(
            f"{tab_label}-tab deposit did not finish within guard limit"
        )

    def _select_inventory_tab(self, name: str) -> None:
        """Click an inventory tab (``use`` / ``eqp`` / ``etc``) and settle.

        Tab BMPs match the *unselected* look. When the tab is already active
        (selected tint), the template misses — treat that as already selected.
        """
        log = self._ctx.logger.behavior

        def find() -> tuple[int, int] | None:
            return find_template(self._ui.capture_client(), name)

        self._ui.cursor_off_screen()
        loc = find()
        if loc is None:
            self._ui.cursor_off_screen()
            loc = find()
        if loc is None:
            log(f"[STORAGE] {name} tab already active ({name}_img not visible)")
            return
        log(f"[STORAGE] click {name} tab")
        ox, oy = self._ui.client_origin()
        self._input.move_mouse(ox + loc[0], oy + loc[1])
        self._wait(0.2)
        self._input.left_click()
        self._ui.cursor_off_screen()
        self._wait(STORAGE_UI_SETTLE_S)

    def _deposit_inventory_to_storage(self) -> None:
        """Deposit Use / Eqp / Etc tabs."""
        self._deposit_tab_grid(tab_label="Use")

        self._wait(STORAGE_UI_SETTLE_S)
        self._select_inventory_tab("eqp")
        self._deposit_tab_grid(tab_label="Eqp")

        self._select_inventory_tab("etc")
        self._deposit_tab_grid(tab_label="Etc")

    def _restock_fly_wings_from_open_storage(
        self, *, ensure_use_tab: bool = False
    ) -> bool:
        """Deposit Use-tab wings then pull amount from storage.

        Inventory + storage must already be open. Returns False if abandoned
        (menus already closed). Does not close menus on success.

        ``ensure_use_tab``: click the Use tab first. Needed after a dump (ends
        on Etc). Skip when Use was already selected for this session —
        ``use_img`` matches the *unselected* tab and misses once Use is active.
        """
        inp = self._input
        ctx = self._ctx
        wings = int(ctx.config.fly_wings_amount)
        if wings <= 0:
            self._abandon_fly_wings("fly_wings_amount is 0")
            return False
        amount = str(wings)
        enter_sc = key_name_to_scan_code("enter")
        if enter_sc <= 0:
            raise InventoryUiError("enter scan code unresolved")

        log = ctx.logger.behavior
        log(f"[STORAGE] GetFlyWings restock amount={wings}")

        if ensure_use_tab:
            log("[STORAGE] GetFlyWings select Use tab before restock")
            self._ui.select_use_tab()
        else:
            log("[STORAGE] GetFlyWings Use tab already selected")

        log("[STORAGE] GetFlyWings sleep 800ms")
        self._wait(0.8)
        log(
            f"[STORAGE] GetFlyWings scan Use grid "
            f"{STORAGE_INV_COLS}x{STORAGE_INV_ROWS} for wings"
        )
        self._deposit_wings_from_use_grid()

        log("[STORAGE] GetFlyWings sleep 500ms")
        self._wait(0.5)

        def find_storage() -> tuple[int, int]:
            frame = self._ui.capture_client()
            panel = require_inventory_panel(frame)
            storage_wing = find_storage_wing(frame, panel)
            if storage_wing is None:
                raise InventoryUiError("no fly wings in storage")
            return storage_wing

        try:
            storage_wing = self._ui.recognize("storage wing", find_storage)
        except InventoryUiError:
            self._abandon_fly_wings("no fly wings in storage")
            return False

        log(
            f"[STORAGE] GetFlyWings move storage wing at {storage_wing} "
            "(sleep 200ms)"
        )
        ox, oy = self._ui.client_origin()
        self._input.move_mouse(ox + storage_wing[0], oy + storage_wing[1])
        self._wait(0.2)
        log("[STORAGE] GetFlyWings sleep 100ms before LMB down")
        self._wait(0.1)
        log("[STORAGE] GetFlyWings LMB down")
        if not inp.set_left_button(True):
            raise InventoryUiError("failed to start fly wing drag")
        try:
            log("[STORAGE] GetFlyWings sleep 100ms")
            self._wait(0.1)
            log("[STORAGE] GetFlyWings drag to etc +100,+20 (sleep 200ms)")
            # LMB held — do not clear cursor (would drag the stack off-screen).
            self._ui.move_to_template("etc", 100, 20, clear_cursor=False)
        finally:
            # Focus loss/stop may interrupt the drag before the normal mouse-up.
            # Always send the release report so the desktop cannot be left stuck.
            log("[STORAGE] GetFlyWings LMB up")
            inp.set_left_button(False)
        log("[STORAGE] GetFlyWings sleep 200ms before type")
        self._wait(0.2)

        log(f"[STORAGE] GetFlyWings type_text {amount!r} (50ms hold + 50ms/digit)")
        if not inp.type_text(amount):
            raise InventoryUiError(f"failed to type fly wing amount {amount!r}")
        log("[STORAGE] GetFlyWings sleep 200ms before Enter")
        self._wait(0.2)
        log("[STORAGE] GetFlyWings Enter confirm (press 50ms)")
        if not inp.key_tap(enter_sc, press_s=0.05, after_s=0.0):
            raise InventoryUiError("failed to confirm fly wing amount (Enter)")

        ctx.wingcount = wings
        log(f"[STORAGE] GetFlyWings restocked wingcount={wings}")
        return True

    def storage_session(self, *, dump: bool, restock: bool) -> None:
        """One inventory/storage open: optional dump, optional wing restock, close.
        On success (or non-critical UI failure) menus are closed after the session.
        """
        if not dump and not restock:
            return
        log = self._ctx.logger.behavior
        try:
            self._wait(0.5)
            self._ui.ensure_inventory_open()
            self._wait(0.5)

            self._ui.select_use_tab()
            self._ui.ensure_storage_open()

            if dump:
                log("[STORAGE] deposit inventory tabs")
                self._deposit_inventory_to_storage()

            if restock:
                # After dump we are on Etc; restock-only keeps Use from above.
                self._restock_fly_wings_from_open_storage(ensure_use_tab=dump)

            self._wait(0.1)
            self._ui.close_menus()
            self._wait(0.5)
        except InventoryUiError:
            try:
                self._ui.close_menus()
            except InventoryUiError as exc:
                log(f"[STORAGE] menu close after session: {exc}")
            self._wait(0.5)
            raise

    def items_to_storage(self) -> None:
        """AHK ``ItemsToStorage`` — dump only, single open/close."""
        self.storage_session(dump=True, restock=False)

    def get_fly_wings(self) -> None:
        """AHK ``GetFlyWings`` — restock only, single open/close."""
        self.storage_session(dump=False, restock=True)

    def _abandon_fly_wings(self, reason: str) -> None:
        """Close menus, disable GetFlyWings for this hunt."""
        ctx = self._ctx
        log = ctx.logger.behavior
        creamy = ctx.config.creamy_tp_button.strip()
        alt = (
            f"Creamy TP ({creamy!r})"
            if ctx.config.creamy_tp_scan_code > 0 and creamy
            else "mob teleport key"
        )
        log(
            f"[STORAGE] fly wings unavailable ({reason}) — "
            "close panels, disable fly-wing restock for session, "
            f"teleport key → {alt}"
        )
        try:
            self._ui.close_menus()
        except InventoryUiError as exc:
            log(f"[STORAGE] menu close after wing abandon: {exc}")
        ctx.mark_fly_wings_exhausted()
