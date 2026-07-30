"""Inventory UI automation helpers for storage operations.

Extracted from ItemsToStorageWorker so the worker owns only orchestration
(run loop, session logic) and delegates low-level UI interactions here.
"""

from __future__ import annotations

import ctypes
import time
from ctypes import wintypes

from pybot.recognition.ui.inventory import (
    InventoryUiError,
    find_template,
    is_inventory_open,
    is_storage_open,
    require_inventory_panel,
    require_template,
)
from pybot.runtime.input.input_backend import InputBackend

user32 = ctypes.windll.user32


def cursor_pos() -> tuple[int, int]:
    pt = wintypes.POINT()
    if not user32.GetCursorPos(ctypes.byref(pt)):
        raise RuntimeError("GetCursorPos failed")
    return int(pt.x), int(pt.y)


class InventoryAutomation:
    """Low-level UI interaction for inventory/storage operations.

    Wraps cursor movement, template matching, menu polling, and keyboard/
    mouse automation so ItemsToStorageWorker can focus on orchestration.
    """

    def __init__(
        self,
        ctx,
        input_backend: InputBackend,
    ) -> None:
        self._ctx = ctx
        self._input = input_backend

    # ── Capture / cursor helpers ─────────────────────────────────

    def capture_client(self):
        frame = self._ctx.capture.capture_client()
        if frame is None or frame.size == 0:
            raise InventoryUiError("client capture failed")
        return frame

    def client_origin(self) -> tuple[int, int]:
        client = self._ctx.capture.get_client_rect_screen()
        if client is None:
            raise InventoryUiError("client rect unavailable")
        return int(client[0]), int(client[1])

    def cursor_in_client(self) -> tuple[int, int]:
        sx, sy = cursor_pos()
        ox, oy = self.client_origin()
        return sx - ox, sy - oy

    def cursor_off_screen(self) -> None:
        """Move cursor just outside the client so it cannot cover UI."""
        client = self._ctx.capture.get_client_rect_screen()
        if client is None:
            raise InventoryUiError("client rect unavailable")
        left, top, _w, _h = client
        x = max(0, int(left) - 2)
        y = max(0, int(top) - 2)
        self._input.move_mouse(x, y)
        time.sleep(0.3)

    # ── Template matching with retry ─────────────────────────────

    def move_to_template(
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
                return require_template(self.capture_client(), name)

            loc = self.recognize(f"template {name}", find)
        else:
            if frame is None:
                frame = self.capture_client()
            loc = require_template(frame, name)
        ox, oy = self.client_origin()
        self._input.move_mouse(ox + loc[0] + x_offset, oy + loc[1] + y_offset)
        time.sleep(0.2)

    def recognize(self, label: str, fn):
        """Run recognition with cursor off UI; one off-screen retry on miss."""
        self.cursor_off_screen()
        try:
            return fn()
        except InventoryUiError as exc:
            self.cursor_off_screen()
            return fn()

    # ── Alt+RMB deposit ──────────────────────────────────────────

    def alt_rmb_deposit(self) -> None:
        """Deposit item under cursor via Alt+RMB."""
        self._input.alt_right_click()

    # ── Menu state polling ───────────────────────────────────────

    def wait_menu_state(
        self,
        *,
        menu: str,
        want_open: bool,
        label: str,
        timeout_s: float = 5.0,
        poll_s: float = 0.2,
    ):
        """Poll until inventory/storage matches *want_open*; return last frame."""
        if menu == "inventory":
            checker = is_inventory_open
        elif menu == "storage":
            checker = is_storage_open
        else:
            raise InventoryUiError(f"unknown menu: {menu}")

        self.cursor_off_screen()
        deadline = time.monotonic() + timeout_s
        last_frame = None
        while time.monotonic() < deadline:
            if self._ctx.is_stopped():
                raise InventoryUiError(f"stopped while waiting for {label}")
            last_frame = self.capture_client()
            if checker(last_frame) is want_open:
                self._ctx.logger.behavior(
                    f"[{label}] menu ok inventory={is_inventory_open(last_frame)} "
                    f"storage={is_storage_open(last_frame)}"
                )
                return last_frame
            time.sleep(poll_s)
        inv = is_inventory_open(last_frame) if last_frame is not None else False
        stor = is_storage_open(last_frame) if last_frame is not None else False
        raise InventoryUiError(
            f"menu validation failed: expected {label} "
            f"(inventory_open={inv} storage_open={stor})"
        )

    def wait_for_inventory_panel(self):
        """Poll until inventory is open; return ``(panel, frame)``."""
        frame = self.wait_menu_state(
            menu="inventory", want_open=True, label="inventory open",
        )
        return require_inventory_panel(frame), frame

    # ── Menu open/close ──────────────────────────────────────────

    def ensure_inventory_open(self):
        """Open inventory if closed; validate open. Return panel hit."""
        self.cursor_off_screen()
        frame = self.capture_client()
        if is_inventory_open(frame):
            self._ctx.logger.behavior("[STORAGE] inventory already open")
            return require_inventory_panel(frame)
        self._ctx.logger.behavior("[STORAGE] Alt+E open inventory")
        self._input.toggle_inventory()
        time.sleep(0.5)
        panel, _frame = self.wait_for_inventory_panel()
        return panel

    def ensure_storage_open(self) -> None:
        """Play Open Storage chain; validate storage is open."""
        steps = self._ctx.config.open_storage_steps
        if not steps:
            raise InventoryUiError("Open Storage keychain is not assigned")
        self.cursor_off_screen()
        frame = self.capture_client()
        if is_storage_open(frame):
            self._ctx.logger.behavior("[STORAGE] storage already open")
            return
        self._ctx.logger.behavior(
            "[STORAGE] open storage chain "
            + " → ".join(f"{k}/{d}ms" for k, _sc, d in steps)
        )
        if not self._input.play_key_chain(steps):
            raise InventoryUiError("Open Storage keychain failed")
        time.sleep(0.5)
        self.wait_menu_state(
            menu="storage", want_open=True, label="storage open",
        )

    def click_storage_close(self) -> None:
        """Click the storage window close control (double click)."""
        self.move_to_template("close")
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
        self.cursor_off_screen()
        time.sleep(0.5)

    def menus_are_open(self) -> bool:
        """True when inventory and/or storage is visible (cursor cleared first)."""
        self.cursor_off_screen()
        frame = self.capture_client()
        return is_inventory_open(frame) or is_storage_open(frame)

    def close_menus(self) -> None:
        """Close storage and/or inventory until both are gone.

        Order does not matter: each pass closes whichever menu is still open.
        One off-screen retry if validation still fails.
        """
        timeout_s = 5.0
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self._ctx.is_stopped():
                raise InventoryUiError("stopped while closing menus")
            self.cursor_off_screen()
            frame = self.capture_client()
            stor = is_storage_open(frame)
            inv = is_inventory_open(frame)
            if not stor and not inv:
                return
            if stor:
                try:
                    self.click_storage_close()
                except InventoryUiError:
                    self.cursor_off_screen()
            elif inv:
                self._input.toggle_inventory()
                time.sleep(0.5)
            time.sleep(0.2)
        self.cursor_off_screen()
        frame = self.capture_client()
        raise InventoryUiError(
            "menu validation failed: expected both closed "
            f"(inventory_open={is_inventory_open(frame)} "
            f"storage_open={is_storage_open(frame)})"
        )

    def close_menus_best_effort(self) -> None:
        """Force-close panels. Never raise."""
        try:
            if not self.menus_are_open():
                return
            self.close_menus()
        except InventoryUiError:
            pass
        except Exception:
            pass

    # ── Tab selection ────────────────────────────────────────────

    def select_use_tab(self) -> None:
        """Click Use when ``use_img`` (unselected) is visible; else already on Use."""
        def find() -> tuple[int, int] | None:
            return find_template(self.capture_client(), "use")

        self.cursor_off_screen()
        loc = find()
        if loc is None:
            self.cursor_off_screen()
            loc = find()
        if loc is None:
            self._ctx.logger.behavior("[STORAGE] Use tab already active")
            return
        self._ctx.logger.behavior("[STORAGE] click Use tab")
        ox, oy = self.client_origin()
        self._input.move_mouse(ox + loc[0], oy + loc[1])
        time.sleep(0.2)
        time.sleep(0.1)
        self._input.left_click()
        self.cursor_off_screen()
        time.sleep(0.5)
