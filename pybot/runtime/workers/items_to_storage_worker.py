"""Deposit inventory to storage when weight is high; restock fly wings.

Faithful port of AHK HexBots ``ItemsToStorage`` / ``GetFlyWings``:
image templates under ``assets/UI``, Alt+E inventory toggle, Alt+RMB deposit.
"""

from __future__ import annotations

import ctypes
import time
from ctypes import wintypes

from pybot.game_state import GameMemoryPoller
from pybot.config.clients import MemoryAddresses
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
from pybot.runtime.clear_area import HuntModeAreaReset, teleport_until_clear
from pybot.runtime.constants import (
    FLY_WING_WEIGHT,
    SIT_IDLE_BEFORE_SIT_S,
    STORAGE_ENTER_SCAN_CODE,
    STORAGE_INV_COLS,
    STORAGE_INV_ROWS,
    STORAGE_MENU_POLL_S,
    STORAGE_MENU_TIMEOUT_S,
    STORAGE_WEIGHT_MODIFIER_MIN,
    STORAGE_WEIGHT_POLL_INTERVAL_S,
    STORAGE_WING_AIM_SETTLE_S,
)
from pybot.runtime.input.input_backend import InputBackend
from pybot.runtime.input.scan_codes import key_name_to_scan_code
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
        memory: MemoryAddresses,
        hunt_mode: HuntModeAreaReset,
        *,
        poller: GameMemoryPoller | None = None,
    ) -> None:
        self._ctx = ctx
        self._input = input_backend
        self._memory = memory
        self._hunt_mode = hunt_mode
        self._poller = poller or GameMemoryPoller()
        self._last_fail_log = ""

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
                        "[STORAGE] teleport until clear before storage UI"
                    )
                    if not teleport_until_clear(
                        ctx, self._input, self._hunt_mode, log_tag="STORAGE"
                    ):
                        ctx.logger.behavior(
                            "[STORAGE] area clear aborted — skip storage session"
                        )
                        continue
                    ctx.logger.behavior(
                        f"[STORAGE] area clear — idle "
                        f"{SIT_IDLE_BEFORE_SIT_S:.0f}s before UI"
                    )
                    if not ctx.wait_unless_stopped(SIT_IDLE_BEFORE_SIT_S):
                        continue
                    self.storage_session(dump=dump, restock=need_wings)
                except InventoryUiError as exc:
                    ctx.logger.behavior(f"[STORAGE] UI miss: {exc}")
                except Exception:
                    import traceback

                    ctx.logger.behavior(
                        f"[STORAGE] cycle error:\n{traceback.format_exc()}"
                    )
                finally:
                    ctx.end_storage_ops()
                    ctx.discovery_wake.set()
            except Exception:
                import traceback

                ctx.logger.behavior(f"[STORAGE] tick error:\n{traceback.format_exc()}")

    def _read_weight(self) -> tuple[int, int] | None:
        """Return ``(weight, weight_max)`` or None when unavailable."""
        ctx = self._ctx
        snap = self._poller.read(ctx.config.hwnd, self._memory)
        if (
            not snap.ok
            or snap.weight is None
            or snap.weight_max is None
            or snap.weight_max <= 0
        ):
            reason = snap.error or "weight_unavailable"
            if reason != self._last_fail_log:
                self._last_fail_log = reason
                ctx.logger.behavior(f"[STORAGE] weight read failed: {reason}")
            return None
        self._last_fail_log = ""
        return int(snap.weight), int(snap.weight_max)

    def _weight_threshold(self, weight_max: int) -> float:
        modifier = int(self._ctx.config.weight_modifier)
        return weight_max * modifier / 100.0

    def _weight_over_threshold(self) -> bool:
        ctx = self._ctx
        if int(ctx.config.weight_modifier) < STORAGE_WEIGHT_MODIFIER_MIN:
            return False
        read = self._read_weight()
        if read is None:
            return False
        weight, weight_max = read
        return weight >= self._weight_threshold(weight_max)

    def _fly_wings_would_hit_threshold(self) -> bool:
        """True when restocking ``fly_wings_amount`` would reach the storage gate."""
        ctx = self._ctx
        if int(ctx.config.weight_modifier) < STORAGE_WEIGHT_MODIFIER_MIN:
            return False
        amount = int(ctx.config.fly_wings_amount)
        if amount <= 0:
            return False
        read = self._read_weight()
        if read is None:
            return False
        weight, weight_max = read
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
    ) -> None:
        """AHK ``MoveCursorToImage``: find template, move, sleep 200ms."""
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
        frame = self._capture_client()
        if is_inventory_open(frame):
            self._ctx.logger.behavior("[STORAGE] inventory already open")
            return require_inventory_panel(frame)
        self._ctx.logger.behavior("[STORAGE] Alt+E open inventory")
        self._input.toggle_inventory()
        panel, _frame = self._wait_for_inventory_panel()
        return panel

    def _ensure_storage_open(self) -> None:
        """Play Open Storage chain; validate storage is open."""
        steps = self._ctx.config.open_storage_steps
        if not steps:
            raise InventoryUiError("Open Storage keychain is not assigned")
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
        self._wait_menu_state(
            menu="storage",
            want_open=True,
            label="storage open",
        )

    def _click_storage_close(self, frame=None) -> None:
        """Click the storage window close control (no validation)."""
        if frame is None:
            frame = self._capture_client()
        self._ctx.logger.behavior("[STORAGE] click storage close")
        self._move_to_template("close", frame=frame)
        time.sleep(0.2)
        self._input.set_left_button(True)
        time.sleep(0.05)
        self._input.set_left_button(False)

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
                        self._click_storage_close(frame)
                    except InventoryUiError as exc:
                        self._ctx.logger.behavior(
                            f"[STORAGE] storage close click miss: {exc}"
                        )
                        self._cursor_off_screen()
                elif inv:
                    self._ctx.logger.behavior("[STORAGE] Alt+E close inventory")
                    self._input.toggle_inventory()
                time.sleep(STORAGE_MENU_POLL_S)
            frame = self._capture_client()
            raise InventoryUiError(
                "menu validation failed: expected both closed "
                f"(inventory_open={is_inventory_open(frame)} "
                f"storage_open={is_storage_open(frame)})"
            )

        self._recognize("close menus", attempt)

    def _image_on_screen(self, name: str) -> bool:
        """AHK ``CheckImageOnScreen``."""
        return find_template(self._capture_client(), name) is not None

    # ── AHK flows ─────────────────────────────────────────────────────

    def _open_storage(self) -> None:
        self._ensure_storage_open()

    def _move_to_first_inventory_cell(self) -> None:
        """Aim at Use-tab slot (0,0) bottom-left (icon stays uncovered)."""
        panel, _frame = self._wait_for_inventory_panel()
        ax, ay = panel.slot_aim(0, 0)
        ox, oy = self._client_origin()
        self._input.move_mouse(ox + ax, oy + ay)
        time.sleep(0.2)

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
        time.sleep(0.05)

    def _recognize(self, label: str, fn):
        """Run a recognition step; on miss, move off-screen once and retry."""
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
            self._cursor_off_screen()
        raise InventoryUiError(
            f"Use-tab wing deposit did not clear after {max_passes} passes"
        )

    def _deposit_use_tab_skipping_wings(self) -> None:
        """Deposit Use-tab non-wing items; skip fly wings."""
        log = self._ctx.logger.behavior
        ox, oy = self._client_origin()
        guard = STORAGE_INV_COLS * STORAGE_INV_ROWS
        for _ in range(guard):
            self._cursor_off_screen()

            def scan_target() -> tuple[int, int, int, int] | None:
                frame = self._capture_client()
                panel = require_inventory_panel(frame)
                for col, row, cx, cy in panel.iter_slot_centers():
                    if slot_looks_empty(frame, cx, cy):
                        continue
                    if slot_contains_template(frame, "wing", cx, cy):
                        log(
                            f"[STORAGE] ItemsToStorage skip fly wing "
                            f"col={col} row={row}"
                        )
                        continue
                    return col, row, *panel.slot_aim(col, row)
                return None

            target = self._recognize("Use-tab item scan", scan_target)
            if target is None:
                return
            col, row, ax, ay = target
            log(
                f"[STORAGE] ItemsToStorage deposit Use item "
                f"col={col} row={row} low-left ({ax},{ay})"
            )
            self._input.move_mouse(ox + ax, oy + ay)
            time.sleep(STORAGE_WING_AIM_SETTLE_S)
            self._alt_rmb_deposit()
        raise InventoryUiError("Use-tab deposit did not finish within grid size")

    def _deposit_inventory_to_storage(self) -> None:
        """Deposit Use (skip wings) / Eqp / Etc. Menus open; Use tab selected."""
        inp = self._input
        self._deposit_use_tab_skipping_wings()

        time.sleep(0.1)
        self._move_to_template("eqp")
        time.sleep(0.1)
        inp.left_click()
        time.sleep(0.05)
        self._move_to_first_inventory_cell()
        while not self._cursor_slot_empty():
            self._alt_rmb_deposit()

        self._move_to_template("etc")
        time.sleep(0.1)
        inp.left_click()
        time.sleep(0.1)
        self._move_to_first_inventory_cell()
        while not self._cursor_slot_empty():
            time.sleep(0.05)
            if self._image_on_screen("ok"):
                inp.key_tap(STORAGE_ENTER_SCAN_CODE, press_s=0.05, after_s=0.0)
                sx, sy = _cursor_pos()
                inp.move_mouse(sx + 40, sy)
            self._alt_rmb_deposit()

    def _restock_fly_wings_from_open_storage(self) -> bool:
        """Deposit Use-tab wings then pull amount from storage.

        Inventory + storage must already be open. Returns False if abandoned
        (menus already closed). Does not close menus on success.
        """
        inp = self._input
        ctx = self._ctx
        wings = int(ctx.config.fly_wings_amount)
        if wings <= 0:
            raise InventoryUiError(
                f"fly_wings_amount must be > 0 (got {wings})"
            )
        amount = str(wings)
        enter_sc = key_name_to_scan_code("enter")
        if enter_sc <= 0:
            raise InventoryUiError("enter scan code unresolved")

        log = ctx.logger.behavior
        log(f"[STORAGE] GetFlyWings restock amount={wings}")

        log("[STORAGE] GetFlyWings click Use tab (use_img)")
        self._move_to_template("use")
        time.sleep(0.1)
        inp.left_click()
        self._cursor_off_screen()

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
        self._move_to_template("etc", 100, 20)
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
        """One inventory/storage open: optional dump, optional wing restock, close."""
        if not dump and not restock:
            return
        log = self._ctx.logger.behavior
        time.sleep(0.5)
        self._ensure_inventory_open()
        time.sleep(0.5)

        self._move_to_template("use")
        time.sleep(0.1)
        self._input.left_click()
        self._open_storage()

        if dump:
            log("[STORAGE] deposit inventory tabs")
            self._deposit_inventory_to_storage()

        if restock:
            if not self._restock_fly_wings_from_open_storage():
                return

        time.sleep(0.1)
        self._close_menus()
        time.sleep(0.5)

    def items_to_storage(self) -> None:
        """AHK ``ItemsToStorage`` — dump only, single open/close."""
        self.storage_session(dump=True, restock=False)

    def get_fly_wings(self) -> None:
        """AHK ``GetFlyWings`` — restock only, single open/close."""
        self.storage_session(dump=False, restock=True)

    def _cursor_slot_empty(self) -> bool:
        """True when the icon under the cursor looks like an empty Use/Eqp/Etc slot."""
        frame = self._capture_client()
        cx, cy = self._cursor_in_client()
        return slot_looks_empty(frame, cx, cy)

    def _abandon_fly_wings(self, reason: str) -> None:
        """Close menus, disable GetFlyWings for this hunt, switch to Creamy TP."""
        ctx = self._ctx
        log = ctx.logger.behavior
        log(
            f"[STORAGE] fly wings unavailable ({reason}) — "
            "close panels, disable fly-wing restock for session, "
            f"teleport key → Creamy TP ({ctx.config.creamy_tp_button!r})"
        )
        try:
            self._close_menus()
        except InventoryUiError as exc:
            log(f"[STORAGE] menu close after wing abandon: {exc}")
        ctx.mark_fly_wings_exhausted()
