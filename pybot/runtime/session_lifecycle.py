"""Focused session lifecycle capability for runtime workers.

The full HuntRuntimeContext remains the compatibility façade, while this
protocol gives workers a narrow, explicit boundary for mutually exclusive
sit/storage/heal sessions. Implementations are structural, so existing test
contexts and lightweight doubles remain usable when they provide the methods.
"""

from __future__ import annotations

from typing import Protocol


class SessionLifecycle(Protocol):
    """Acquire and release one mutually exclusive runtime session."""

    def begin_sit_ops(self) -> bool: ...
    def try_begin_sit_ops(self) -> bool: ...
    def end_sit_ops(self) -> None: ...

    def begin_storage_ops(self) -> bool: ...
    def end_storage_ops(self) -> None: ...

    def begin_heal_ops(self) -> bool: ...
    def end_heal_ops(self) -> None: ...


__all__ = ["SessionLifecycle"]
