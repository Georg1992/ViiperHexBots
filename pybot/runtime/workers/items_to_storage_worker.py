"""Deposit inventory to storage when weight is high; restock fly wings.

Faithful port of AHK HexBots ``ItemsToStorage`` / ``GetFlyWings``:
image templates under ``assets/UI``, Alt+E inventory toggle, Alt+RMB deposit.
"""

from __future__ import annotations

import ctypes
import time
from ctypes import wintypes

from pybot.app.process_memory import GameMemoryPoller
from pybot.config.clients import MemoryAddresses
from pybot.recognition.ui.inventory import (
    CELL_SIZE_PX,
    InventoryUiError,
    cell_contains_template,
    find_template,
    require_template,
)
from pybot.runtime.constants import (
    STORAGE_CELL1_OFFSET_X,
    STORAGE_CELL1_OFFSET_Y,
    STORAGE_ENTER_SCAN_CODE,
    STORAGE_INV_COLS,
    STORAGE_INV_ROWS,
    STORAGE_WEIGHT_MODIFIER_MIN,
    STORAGE_WEIGHT_POLL_INTERVAL_S,
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
    """When weight ≥ WeightModifier%, pause hunt and run AHK storage flows."""

    def __init__(
        self,
        ctx: ItemsToStorageWorkerContext,
        input_backend: InputBackend,
        memory: MemoryAddresses,
        *,
        poller: GameMemoryPoller | None = None,
    ) -> None:
        self._ctx = ctx
        self._input = input_backend
        self._memory = memory
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
                need_wings = cfg.take_fly_wings and ctx.wingcount <= 0
                if not heavy and not need_wings:
                    ctx.stop_event.wait(STORAGE_WEIGHT_POLL_INTERVAL_S)
                    continue
                if not ctx.begin_exclusive_ops():
                    continue
                try:
                    if heavy:
                        ctx.logger.behavior("[STORAGE] weight high — ItemsToStorage")
                        self.items_to_storage()
                    if cfg.take_fly_wings and ctx.wingcount <= 0:
                        ctx.logger.behavior("[STORAGE] wingcount=0 — GetFlyWings")
                        self.get_fly_wings()
                except InventoryUiError as exc:
                    ctx.logger.behavior(f"[STORAGE] UI miss: {exc}")
                except Exception:
                    import traceback

                    ctx.logger.behavior(
                        f"[STORAGE] cycle error:\n{traceback.format_exc()}"
                    )
                finally:
                    ctx.end_exclusive_ops()
                    ctx.discovery_wake.set()
            except Exception:
                import traceback

                ctx.logger.behavior(f"[STORAGE] tick error:\n{traceback.format_exc()}")

    def _weight_over_threshold(self) -> bool:
        ctx = self._ctx
        modifier = int(ctx.config.weight_modifier)
        if modifier < STORAGE_WEIGHT_MODIFIER_MIN:
            return False
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
            return False
        self._last_fail_log = ""
        return snap.weight >= (snap.weight_max * modifier / 100.0)

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

    def _image_on_screen(self, name: str) -> bool:
        """AHK ``CheckImageOnScreen``."""
        return find_template(self._capture_client(), name) is not None

    def _check_inventory_cell(self, name: str, ignore_wing: bool = True) -> bool:
        """AHK ``CheckInventoryCell``."""
        frame = self._capture_client()
        cx, cy = self._cursor_in_client()
        found = cell_contains_template(frame, name, cx, cy)
        if found and name == "wing" and not ignore_wing:
            next_x = cx + CELL_SIZE_PX
            next_y = cy
            client = self._ctx.capture.get_client_rect_screen()
            if client is None:
                raise InventoryUiError("client rect unavailable")
            _left, _top, client_w, _client_h = client
            max_right = client_w - CELL_SIZE_PX // 2
            if next_x > max_right:
                next_x = CELL_SIZE_PX // 2
                next_y += CELL_SIZE_PX
            ox, oy = self._client_origin()
            self._input.move_mouse(ox + next_x, oy + next_y)
        return found

    # ── AHK flows ─────────────────────────────────────────────────────

    def _open_storage(self) -> None:
        steps = self._ctx.config.open_storage_steps
        if not steps:
            raise InventoryUiError("Open Storage keychain is not assigned")
        if not self._input.play_key_chain(steps):
            raise InventoryUiError("Open Storage keychain failed")

    def _move_to_first_inventory_cell(self) -> None:
        """Aim at the center of the first inventory cell (not the left edge)."""
        self._move_to_template(
            "cell1", STORAGE_CELL1_OFFSET_X, STORAGE_CELL1_OFFSET_Y
        )

    def _alt_rmb_deposit(self) -> None:
        """Deposit item under cursor via Alt+RMB (includes mandatory 100ms delay)."""
        self._input.alt_right_click()

    def _deposit_wings_from_use_grid(self) -> int:
        """Scan every Use-tab inventory slot; Alt+RMB each fly-wing stack.

        Grid is ``STORAGE_INV_COLS`` × ``STORAGE_INV_ROWS`` from cell1 center.
        Rescans until a full pass finds no wings (items may shift after deposit).
        Returns how many wing stacks were deposited.
        """
        log = self._ctx.logger.behavior
        ox, oy = self._client_origin()
        self._move_to_first_inventory_cell()
        base_cx, base_cy = self._cursor_in_client()
        deposited = 0
        max_passes = STORAGE_INV_COLS * STORAGE_INV_ROWS
        for pass_i in range(max_passes):
            found_this_pass = 0
            for row in range(STORAGE_INV_ROWS):
                for col in range(STORAGE_INV_COLS):
                    cx = base_cx + col * CELL_SIZE_PX
                    cy = base_cy + row * CELL_SIZE_PX
                    self._input.move_mouse(ox + cx, oy + cy)
                    time.sleep(0.05)
                    if not self._check_inventory_cell("wing"):
                        continue
                    log(
                        f"[STORAGE] GetFlyWings Use wing at col={col} row={row} "
                        f"— Alt+RMB deposit"
                    )
                    self._alt_rmb_deposit()
                    found_this_pass += 1
                    deposited += 1
            if found_this_pass == 0:
                log(
                    f"[STORAGE] GetFlyWings Use grid scan done "
                    f"passes={pass_i + 1} deposited={deposited}"
                )
                return deposited
        raise InventoryUiError(
            f"Use-tab wing deposit did not clear after {max_passes} passes"
        )

    def items_to_storage(self) -> None:
        """AHK ``ItemsToStorage`` — same order and sleeps."""
        inp = self._input
        time.sleep(0.5)
        inp.toggle_inventory()
        time.sleep(0.5)

        self._move_to_template("use")
        time.sleep(0.1)
        inp.left_click()
        self._open_storage()
        self._move_to_first_inventory_cell()

        while not self._check_inventory_cell("empty_cell"):
            self._check_inventory_cell("wing", ignore_wing=False)
            self._alt_rmb_deposit()

        time.sleep(0.1)
        self._move_to_template("eqp")
        time.sleep(0.1)
        inp.left_click()
        time.sleep(0.05)
        self._move_to_first_inventory_cell()

        while not self._check_inventory_cell("empty_cell"):
            self._alt_rmb_deposit()

        self._move_to_template("etc")
        time.sleep(0.1)
        inp.left_click()
        time.sleep(0.1)
        self._move_to_first_inventory_cell()

        while not self._check_inventory_cell("empty_cell"):
            time.sleep(0.05)
            if self._image_on_screen("ok"):
                inp.key_tap(STORAGE_ENTER_SCAN_CODE, press_s=0.05, after_s=0.0)
                sx, sy = _cursor_pos()
                inp.move_mouse(sx + 40, sy)
            self._alt_rmb_deposit()

        time.sleep(0.1)
        self._move_to_template("close", 10, 10)
        time.sleep(0.1)
        inp.left_click()
        inp.toggle_inventory()
        time.sleep(0.5)

    def get_fly_wings(self) -> None:
        """AHK ``GetFlyWings`` — same order and sleeps.

        Types ``fly_wings_amount`` from the GUI (synced into runtime config
        on Start) into the quantity dialog, then Enter.
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
        log(f"[STORAGE] GetFlyWings start amount={wings}")

        log("[STORAGE] GetFlyWings sleep 100ms")
        time.sleep(0.1)
        log("[STORAGE] GetFlyWings Alt+E open inventory (+500ms inside)")
        inp.toggle_inventory()
        log("[STORAGE] GetFlyWings click Use tab (use_img)")
        self._move_to_template("use")
        time.sleep(0.1)
        inp.left_click()
        log("[STORAGE] GetFlyWings move cell1 (+20,+20 then sleep 200ms)")
        self._move_to_first_inventory_cell()

        steps = ctx.config.open_storage_steps
        log(
            "[STORAGE] GetFlyWings open storage chain "
            + " → ".join(f"{k}/{d}ms" for k, _sc, d in steps)
        )
        self._open_storage()

        log("[STORAGE] GetFlyWings sleep 800ms")
        time.sleep(0.8)

        log(
            f"[STORAGE] GetFlyWings scan Use grid "
            f"{STORAGE_INV_COLS}x{STORAGE_INV_ROWS} for wings"
        )
        self._deposit_wings_from_use_grid()

        log("[STORAGE] GetFlyWings sleep 500ms")
        time.sleep(0.5)
        log("[STORAGE] GetFlyWings move wing template (sleep 200ms)")
        self._move_to_template("wing")
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

        log("[STORAGE] GetFlyWings Alt+E close inventory (+500ms inside)")
        inp.toggle_inventory()
        log("[STORAGE] GetFlyWings move close (sleep 200ms)")
        self._move_to_template("close")
        log("[STORAGE] GetFlyWings sleep 200ms before close click")
        time.sleep(0.2)
        log("[STORAGE] GetFlyWings LMB down close")
        inp.set_left_button(True)
        log("[STORAGE] GetFlyWings sleep 50ms")
        time.sleep(0.05)
        log("[STORAGE] GetFlyWings LMB up close")
        inp.set_left_button(False)
        ctx.wingcount = wings
        log(f"[STORAGE] GetFlyWings done wingcount={wings}; sleep 200ms")
        time.sleep(0.2)
