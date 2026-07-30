"""Deposit inventory to storage when weight is high; restock fly wings.

Faithful port of AHK HexBots ``ItemsToStorage`` / ``GetFlyWings``:
image templates under ``assets/UI``, Alt+E inventory toggle, Alt+RMB deposit.

Weight and HP are read from ``PlayerVitals`` which is published by the UI
(either from status-panel OCR or process-memory polls).  The worker must
never re-OCR or re-read process memory.
"""

from __future__ import annotations

import ctypes
import time
import traceback
from ctypes import wintypes

from pybot.game_state import PlayerVitals
from pybot.recognition.ui.inventory import (
    InventoryUiError,
    find_storage_wing,
    find_template,
    find_wings_in_use_grid,
    is_inventory_open,
    is_storage_open,
    require_inventory_panel,
    require_template,
    slot_contains_template,
    slot_looks_empty,
)
from pybot.runtime.constants import (
    FLY_WING_WEIGHT,
    STORAGE_CURSOR_CLEAR_S,
    STORAGE_INV_COLS,
    STORAGE_INV_ROWS,
    STORAGE_MENU_POLL_S,
    STORAGE_MENU_TIMEOUT_S,
    STORAGE_UI_SETTLE_S,
    STORAGE_WEIGHT_MODIFIER_MIN,
    STORAGE_WEIGHT_POLL_INTERVAL_S,
    STORAGE_WING_AIM_SETTLE_S,
)
from pybot.runtime.input.input_backend import InputBackend
from pybot.runtime.input.scan_codes import key_name_to_scan_code
from pybot.runtime.teleport import TeleportController
from pybot.runtime.workers.worker_contexts import ItemsToStorageWorkerContext

user32 = ctypes.windll.user32


def _cursor_pos() -> tuple[int, int]:
    pt = wintypes.POINT()
    if not user32.GetCursorPos(ctypes.byref(pt)):
        raise RuntimeError("GetCursorPos failed")
    return int(pt.x), int(pt.y)


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

    # ── Capture / cursor helpers ──────────────────────────────────────

    def _capture_client(self):
        frame = self._ctx.capture.capture_client()
        if frame is None or frame.size == 0:
            raise InventoryUiError("client capture failed")
        return frame

    def _client_origin(self) -> tuple[int, int]:
        client = self._ctx.capture.get_client_rect_screen()
        if client is None:
            raise InventoryUiError("client rect unavailable")
        return int(client[0]), int(client[1])

    def _cursor_in_client(self) -> tuple[int, int]:
        sx, sy = _cursor_pos()
        ox, oy = self._client_origin()
        return sx - ox, sy - oy

    def _move_to_template(
        self,
        name: str,
        x_offset: int = 0,
        y_offset: int = 0,
        *,
        frame=None,
        clear_cursor: bool = True,
    ) -> None:
        """AHK ``MoveCursorToImage``: find template, move, sleep 200ms.

        ``clear_cursor`` (default True) moves off UI before matching so the
        cursor cannot cover the template. Set False while LMB is held (drag).
        """
        if clear_cursor:
            def find() -> tuple[int, int]:
                return require_template(self._capture_client(), name)

            loc = self._recognize(f"template {name}", find)
        else:
            if frame is None:
                frame = self._capture_client()
            loc = require_template(frame, name)
        ox, oy = self._client_origin()
        self._input.move_mouse(ox + loc[0] + x_offset, oy + loc[1] + y_offset)
        time.sleep(0.2)

    def _wait_for_inventory_panel(self):
        """Poll until inventory is open; return ``(panel, frame)``."""
        frame = self._wait_menu_state(
            menu="inventory",
            want_open=True,
            label="inventory open",
        )
        return require_inventory_panel(frame), frame

    def _wait_menu_state(
        self,
        *,
        menu: str,
        want_open: bool,
        label: str,
        timeout_s: float = STORAGE_MENU_TIMEOUT_S,
    ):
        """Poll until inventory/storage matches *want_open*; return last frame."""
        if menu == "inventory":
            checker = is_inventory_open
        elif menu == "storage":
            checker = is_storage_open
        else:
            raise InventoryUiError(f"unknown menu: {menu}")

        self._cursor_off_screen()
        deadline = time.monotonic() + timeout_s
        last_frame = None
        while time.monotonic() < deadline:
            if self._ctx.is_stopped():
                raise InventoryUiError(f"stopped while waiting for {label}")
            last_frame = self._capture_client()
            if checker(last_frame) is want_open:
                self._ctx.logger.behavior(
                    f"[STORAGE] menu ok {label} "
                    f"inventory={is_inventory_open(last_frame)} "
                    f"storage={is_storage_open(last_frame)}"
                )
                return last_frame
            time.sleep(STORAGE_MENU_POLL_S)
        inv = is_inventory_open(last_frame) if last_frame is not None else False
        stor = is_storage_open(last_frame) if last_frame is not None else False
        raise InventoryUiError(
            f"menu validation failed: expected {label} "
            f"(inventory_open={inv} storage_open={stor})"
        )

    def _ensure_inventory_open(self):
        """Open inventory if closed; validate open. Return panel hit."""
        self._cursor_off_screen()
        frame = self._capture_client()
        if is_inventory_open(frame):
            self._ctx.logger.behavior("[STORAGE] inventory already open")
            return require_inventory_panel(frame)
        self._ctx.logger.behavior("[STORAGE] Alt+E open inventory")
        self._input.toggle_inventory()
        time.sleep(STORAGE_UI_SETTLE_S)
        panel, _frame = self._wait_for_inventory_panel()
        return panel

    def _ensure_storage_open(self) -> None:
        """Play Open Storage chain; validate storage is open."""
        steps = self._ctx.config.open_storage_steps
        if not steps:
            raise InventoryUiError("Open Storage keychain is not assigned")
        self._cursor_off_screen()
        frame = self._capture_client()
        if is_storage_open(frame):
            self._ctx.logger.behavior("[STORAGE] storage already open")
            return
        self._ctx.logger.behavior(
            "[STORAGE] open storage chain "
            + " → ".join(f"{k}/{d}ms" for k, _sc, d in steps)
        )
        if not self._input.play_key_chain(steps):
            raise InventoryUiError("Open Storage keychain failed")
        time.sleep(STORAGE_UI_SETTLE_S)
        self._wait_menu_state(
            menu="storage",
            want_open=True,
            label="storage open",
        )

    def _click_storage_close(self) -> None:
        """Click the storage window close control (no validation).
        To close the storage we need to double click (lmcx2).
        """
        self._ctx.logger.behavior("[STORAGE] double click storage close")
        self._move_to_template("close")
        time.sleep(0.2)
        # First click
        self._input.set_left_button(True)
        time.sleep(0.05)
        self._input.set_left_button(False)
        time.sleep(0.05)
        # Second click
        self._input.set_left_button(True)
        time.sleep(0.05)
        self._input.set_left_button(False)
        self._cursor_off_screen()
        time.sleep(STORAGE_UI_SETTLE_S)

    def _menus_are_open(self) -> bool:
        """True when inventory and/or storage is visible (cursor cleared first)."""
        self._cursor_off_screen()
        frame = self._capture_client()
        return is_inventory_open(frame) or is_storage_open(frame)

    def _close_menus_best_effort(self) -> None:
        """Force-close panels. Never raise."""
        try:
            if not self._menus_are_open():
                return
            self._ctx.logger.behavior("[STORAGE] force-closing menus")
            self._close_menus()
        except InventoryUiError as exc:
            self._ctx.logger.behavior(f"[STORAGE] force close: {exc}")
        except Exception as exc:
            self._ctx.logger.behavior(f"[STORAGE] force close error: {exc}")


    def _select_use_tab(self) -> None:
        """Click Use when ``use_img`` (unselected) is visible; else already on Use.

        ``use_img`` matches the inactive tab only. Raising on miss aborts the
        session after Use is already selected (common after a failed prior run).
        """
        log = self._ctx.logger.behavior

        def find() -> tuple[int, int] | None:
            return find_template(self._capture_client(), "use")

        self._cursor_off_screen()
        loc = find()
        if loc is None:
            self._cursor_off_screen()
            loc = find()
        if loc is None:
            log("[STORAGE] Use tab already active (use_img not visible)")
            return
        log("[STORAGE] click Use tab (use_img)")
        ox, oy = self._client_origin()
        self._input.move_mouse(ox + loc[0], oy + loc[1])
        time.sleep(0.2)
        time.sleep(0.1)
        self._input.left_click()
        self._cursor_off_screen()
        time.sleep(STORAGE_UI_SETTLE_S)

    def _close_menus(self) -> None:
        """Close storage and/or inventory until both are gone.

        Order does not matter: each pass closes whichever menu is still open.
        One off-screen retry if validation still fails.
        """

        def attempt() -> None:
            deadline = time.monotonic() + STORAGE_MENU_TIMEOUT_S
            while time.monotonic() < deadline:
                if self._ctx.is_stopped():
                    raise InventoryUiError("stopped while closing menus")
                self._cursor_off_screen()
                frame = self._capture_client()
                stor = is_storage_open(frame)
                inv = is_inventory_open(frame)
                if not stor and not inv:
                    self._ctx.logger.behavior(
                        "[STORAGE] menu ok both closed "
                        f"inventory={inv} storage={stor}"
                    )
                    return
                if stor:
                    try:
                        self._click_storage_close()
                    except InventoryUiError as exc:
                        self._ctx.logger.behavior(
                            f"[STORAGE] storage close click miss: {exc}"
                        )
                        self._cursor_off_screen()
                elif inv:
                    self._ctx.logger.behavior("[STORAGE] Alt+E close inventory")
                    self._input.toggle_inventory()
                    time.sleep(STORAGE_UI_SETTLE_S)
                time.sleep(STORAGE_MENU_POLL_S)
            self._cursor_off_screen()
            frame = self._capture_client()
            raise InventoryUiError(
                "menu validation failed: expected both closed "
                f"(inventory_open={is_inventory_open(frame)} "
                f"storage_open={is_storage_open(frame)})"
            )

        self._recognize("close menus", attempt)

    # ── AHK flows ─────────────────────────────────────────────────────

    def _open_storage(self) -> None:
        self._ensure_storage_open()

    def _cursor_off_screen(self) -> None:
        """Move cursor just outside the client so it cannot cover UI."""
        client = self._ctx.capture.get_client_rect_screen()
        if client is None:
            raise InventoryUiError("client rect unavailable")
        left, top, _w, _h = client
        x = max(0, int(left) - 2)
        y = max(0, int(top) - 2)
        self._ctx.logger.behavior(
            f"[STORAGE] cursor off-screen at ({x},{y})"
        )
        self._input.move_mouse(x, y)
        time.sleep(STORAGE_CURSOR_CLEAR_S)

    def _recognize(self, label: str, fn):
        """Run recognition with cursor off UI; one off-screen retry on miss."""
        self._cursor_off_screen()
        try:
            return fn()
        except InventoryUiError as exc:
            self._ctx.logger.behavior(
                f"[STORAGE] {label} failed ({exc}); off-screen retry"
            )
            self._cursor_off_screen()
            return fn()

    def _alt_rmb_deposit(self) -> None:
        """Deposit item under cursor via Alt+RMB (includes mandatory 100ms delay)."""
        self._input.alt_right_click()

    def _scan_use_grid_wings(self) -> list[tuple[int, int, int, int]]:
        def scan() -> list[tuple[int, int, int, int]]:
            frame = self._capture_client()
            panel = require_inventory_panel(frame)
            return find_wings_in_use_grid(frame, panel)

        return self._recognize("Use-grid wing scan", scan)

    def _deposit_wings_from_use_grid(self) -> int:
        """Find each Use-tab fly wing, aim bottom-left, Alt+RMB into storage."""
        log = self._ctx.logger.behavior
        ox, oy = self._client_origin()
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
            time.sleep(STORAGE_WING_AIM_SETTLE_S)
            log(
                f"[STORAGE] GetFlyWings Alt+RMB deposit "
                f"col={col} row={row}"
            )
            self._alt_rmb_deposit()
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
        ox, oy = self._client_origin()
        guard = STORAGE_INV_COLS * STORAGE_INV_ROWS

        # Determine the first slot's center and aim position
        frame_init = self._capture_client()
        panel_init = require_inventory_panel(frame_init)
        first_col, first_row = 0, 0
        first_cx, first_cy = panel_init.slot_center(first_col, first_row)
        first_ax, first_ay = panel_init.slot_aim(first_col, first_row)

        log(f"[STORAGE] Move to first slot of {tab_label} tab ({first_ax},{first_ay})")
        self._input.move_mouse(ox + first_ax, oy + first_ay)
        time.sleep(STORAGE_WING_AIM_SETTLE_S)

        for pass_i in range(guard):
            frame = self._capture_client()

            if slot_looks_empty(frame, first_cx, first_cy):
                log(f"[STORAGE] ItemsToStorage {tab_label} tab first slot empty — done")
                return

            log(f"[STORAGE] ItemsToStorage deposit {tab_label} item from first slot")
            self._alt_rmb_deposit()
            # Let the slot clear/next item move into the first slot before next loop
            time.sleep(STORAGE_UI_SETTLE_S)

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
            return find_template(self._capture_client(), name)

        self._cursor_off_screen()
        loc = find()
        if loc is None:
            self._cursor_off_screen()
            loc = find()
        if loc is None:
            log(f"[STORAGE] {name} tab already active ({name}_img not visible)")
            return
        log(f"[STORAGE] click {name} tab")
        ox, oy = self._client_origin()
        self._input.move_mouse(ox + loc[0], oy + loc[1])
        time.sleep(0.2)
        self._input.left_click()
        self._cursor_off_screen()
        time.sleep(STORAGE_UI_SETTLE_S)

    def _deposit_inventory_to_storage(self) -> None:
        """Deposit Use / Eqp / Etc tabs."""
        self._deposit_tab_grid(tab_label="Use")

        time.sleep(STORAGE_UI_SETTLE_S)
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
            self._select_use_tab()
        else:
            log("[STORAGE] GetFlyWings Use tab already selected")

        log("[STORAGE] GetFlyWings sleep 800ms")
        time.sleep(0.8)
        log(
            f"[STORAGE] GetFlyWings scan Use grid "
            f"{STORAGE_INV_COLS}x{STORAGE_INV_ROWS} for wings"
        )
        self._deposit_wings_from_use_grid()

        log("[STORAGE] GetFlyWings sleep 500ms")
        time.sleep(0.5)

        def find_storage() -> tuple[int, int]:
            frame = self._capture_client()
            panel = require_inventory_panel(frame)
            storage_wing = find_storage_wing(frame, panel)
            if storage_wing is None:
                raise InventoryUiError("no fly wings in storage")
            return storage_wing

        try:
            storage_wing = self._recognize("storage wing", find_storage)
        except InventoryUiError:
            self._abandon_fly_wings("no fly wings in storage")
            return False

        log(
            f"[STORAGE] GetFlyWings move storage wing at {storage_wing} "
            "(sleep 200ms)"
        )
        ox, oy = self._client_origin()
        self._input.move_mouse(ox + storage_wing[0], oy + storage_wing[1])
        time.sleep(0.2)
        log("[STORAGE] GetFlyWings sleep 100ms before LMB down")
        time.sleep(0.1)
        log("[STORAGE] GetFlyWings LMB down")
        inp.set_left_button(True)
        log("[STORAGE] GetFlyWings sleep 100ms")
        time.sleep(0.1)
        log("[STORAGE] GetFlyWings drag to etc +100,+20 (sleep 200ms)")
        # LMB held — do not clear cursor (would drag the stack off-screen).
        self._move_to_template("etc", 100, 20, clear_cursor=False)
        log("[STORAGE] GetFlyWings LMB up")
        inp.set_left_button(False)
        log("[STORAGE] GetFlyWings sleep 200ms before type")
        time.sleep(0.2)

        log(f"[STORAGE] GetFlyWings type_text {amount!r} (50ms hold + 50ms/digit)")
        if not inp.type_text(amount):
            raise InventoryUiError(f"failed to type fly wing amount {amount!r}")
        log("[STORAGE] GetFlyWings sleep 200ms before Enter")
        time.sleep(0.2)
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
            time.sleep(0.5)
            self._ensure_inventory_open()
            time.sleep(0.5)

            self._select_use_tab()
            self._open_storage()

            if dump:
                log("[STORAGE] deposit inventory tabs")
                self._deposit_inventory_to_storage()

            if restock:
                # After dump we are on Etc; restock-only keeps Use from above.
                self._restock_fly_wings_from_open_storage(ensure_use_tab=dump)

            time.sleep(0.1)
            self._close_menus()
            time.sleep(0.5)
        except InventoryUiError:
            try:
                self._close_menus()
            except InventoryUiError as exc:
                log(f"[STORAGE] menu close after session: {exc}")
            time.sleep(0.5)
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
            self._close_menus()
        except InventoryUiError as exc:
            log(f"[STORAGE] menu close after wing abandon: {exc}")
        ctx.mark_fly_wings_exhausted()
