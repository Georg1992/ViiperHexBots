"""Input abstraction for hunt actions."""

from __future__ import annotations

from typing import Protocol


class InputBackend(Protocol):
    def move_mouse(self, x: int, y: int) -> bool: ...

    def skill_click(self, scan_code: int) -> bool: ...

    def teleport_key(self, scan_code: int) -> bool: ...

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

    def shutdown(self) -> None:
        return None
