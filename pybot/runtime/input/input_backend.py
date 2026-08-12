"""Input abstraction for hunt actions."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Protocol



def perform_if_allowed(
    backend: "InputBackend",
    allowed: Callable[[], bool],
    action: Callable[[], bool],
    *,
    lifecycle=None,
) -> bool:
    """Run one input action only when its lifecycle gate still admits it.

    Runtime contexts evaluate the predicate under the existing session
    ownership lock; lightweight callers use the same check-before-action
    contract without adding synchronization. The action itself owns its
    normal input timing and cancellation behavior.
    """
    # A runtime context can make the lifecycle check and action one atomic
    # ownership boundary. Inspect the type rather than the instance so
    # MagicMock/lightweight doubles cannot fabricate a lifecycle method and
    # silently alter the admission contract.
    admit = getattr(type(lifecycle), "perform_input_if_allowed", None)
    if lifecycle is not None and callable(admit):
        return lifecycle.perform_input_if_allowed(allowed, action)

    if not allowed():
        return False
    result = action()
    # Existing worker contracts reject only an explicit False. Lightweight
    # test/custom backends historically return None for successful input.
    return result is not False


class InputBackend(Protocol):
    def move_mouse(self, x: int, y: int) -> bool: ...

    def move_and_double_click(self, x: int, y: int) -> bool: ...

    def skill_click_at(
        self,
        scan_code: int,
        x: int,
        y: int,
        *,
        move_delay_s: float = 0.05,
    ) -> bool: ...

    def teleport_key(self, scan_code: int) -> bool: ...

    def left_click(self) -> bool: ...

    def set_left_button(self, down: bool) -> bool: ...

    def alt_right_click(self) -> bool: ...

    def key_tap(
        self,
        scan_code: int,
        *,
        press_s: float = 0.05,
        after_s: float = 0.30,
    ) -> bool: ...

    def toggle_key(self, scan_code: int) -> bool: ...

    def cleanup_toggle_key(self, scan_code: int) -> bool: ...

    def type_text(self, text: str) -> bool: ...

    def toggle_inventory(self) -> bool: ...

    def play_key_chain(
        self, steps: tuple[tuple[str, int, int], ...]
    ) -> bool: ...

    def begin_session(self, timeout_s: float | None = None) -> bool: ...

    def cancel_pending(self) -> None: ...

    def wait_interruptible(self, seconds: float) -> bool: ...

    def shutdown(self) -> bool: ...


class ShadowInputBackend:
    """No-op input for shadow mode.

    Precondition guards (e.g. ``scan_code <= 0``) mirror ViiperBackend
    so that subtypes are LSP-substitutable — callers see the same
    rejection behaviour regardless of backend.
    """

    def move_mouse(self, x: int, y: int) -> bool:
        return True

    def move_and_double_click(self, x: int, y: int) -> bool:
        del x, y
        return self.left_click() and self.left_click()

    def skill_click_at(
        self,
        scan_code: int,
        x: int,
        y: int,
        *,
        move_delay_s: float = 0.05,
    ) -> bool:
        if scan_code <= 0:
            return False
        del x, y, move_delay_s
        return True

    def teleport_key(self, scan_code: int) -> bool:
        if scan_code <= 0:
            return False
        return True

    def left_click(self) -> bool:
        return True

    def set_left_button(self, down: bool) -> bool:
        return True

    def alt_right_click(self) -> bool:
        """Alt+RMB once, then always wait ``ALT_MOUSE_CLICK_DELAY_S``."""
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

    def toggle_key(self, scan_code: int) -> bool:
        """Send one toggle key without a second post-key delay."""
        return self.key_tap(scan_code, after_s=0.0)

    def cleanup_toggle_key(self, scan_code: int) -> bool:
        """Send a shutdown-only toggle after normal input is cancelled."""
        return self.toggle_key(scan_code)

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

    def begin_session(self, timeout_s: float | None = None) -> bool:
        del timeout_s
        return True

    def cancel_pending(self) -> None:
        return None

    def wait_interruptible(self, seconds: float) -> bool:
        """Shadow waits are no-ops but retain the cancellation contract."""
        del seconds
        return True

    def shutdown(self) -> bool:
        return True
