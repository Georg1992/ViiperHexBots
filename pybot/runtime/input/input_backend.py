"""Input abstraction for hunt actions."""

from __future__ import annotations

import time
from typing import Protocol


class InputBackend(Protocol):
    def move_mouse(self, x: int, y: int) -> bool: ...

    def skill_click(self, scan_code: int) -> bool: ...

    def teleport_key(self, scan_code: int) -> bool: ...

    def left_click(self) -> bool: ...

    def right_click(self) -> bool: ...

    def set_left_button(self, down: bool) -> bool: ...

    def alt_right_click(self) -> bool: ...

    def alt_right_clicks(self, times: int = 1) -> bool: ...

    def key_tap(
        self,
        scan_code: int,
        *,
        press_s: float = 0.05,
        after_s: float = 0.30,
    ) -> bool: ...

    def type_text(self, text: str) -> bool: ...

    def toggle_inventory(self) -> bool: ...

    def play_key_chain(
        self, steps: tuple[tuple[str, int, int], ...]
    ) -> bool: ...

    def shutdown(self) -> None: ...


class ShadowInputBackend:
    """No-op input for shadow mode.

    Precondition guards (e.g. ``scan_code <= 0``) mirror ViiperBackend
    so that subtypes are LSP-substitutable — callers see the same
    rejection behaviour regardless of backend.
    """

    def move_mouse(self, x: int, y: int) -> bool:
        return True

    def skill_click(self, scan_code: int) -> bool:
        if scan_code <= 0:
            return False
        return True

    def teleport_key(self, scan_code: int) -> bool:
        if scan_code <= 0:
            return False
        return True

    def left_click(self) -> bool:
        return True

    def right_click(self) -> bool:
        return True

    def set_left_button(self, down: bool) -> bool:
        return True

    def alt_right_click(self) -> bool:
        """Alt+RMB once, then always wait ``ALT_MOUSE_CLICK_DELAY_S``."""
        return True

    def alt_right_clicks(self, times: int = 1) -> bool:
        if times <= 0:
            return False
        for _ in range(times):
            if not self.alt_right_click():
                return False
        return True

    def key_tap(
        self,
        scan_code: int,
        *,
        press_s: float = 0.05,
        after_s: float = 0.30,
    ) -> bool:
        del press_s, after_s
        if scan_code <= 0:
            return False
        return True

    def type_text(self, text: str) -> bool:
        if not text:
            return False
        return True

    def toggle_inventory(self) -> bool:
        return True

    def play_key_chain(
        self, steps: tuple[tuple[str, int, int], ...]
    ) -> bool:
        """Play ``(button, scan_code, delay_ms)`` steps; delay is after each tap."""
        if not steps:
            return False
        for _button, scan_code, delay_ms in steps:
            if scan_code <= 0:
                return False
            if not self.key_tap(scan_code, press_s=0.05, after_s=0.0):
                return False
            if delay_ms > 0:
                time.sleep(delay_ms / 1000.0)
        return True

    def shutdown(self) -> None:
        return None
