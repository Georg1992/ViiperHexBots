"""Hunt overlay port — runtime workers depend on this interface only."""

from __future__ import annotations

from typing import Protocol


class HuntOverlay(Protocol):
    def set_scan_living(self, count: int) -> None: ...

    def set_track_stats(self, track_count: int, alive_count: int) -> None: ...

    def set_track_positions(self, positions: list[tuple[int, int]]) -> None: ...

    def set_search_roi(self, x: int, y: int, w: int, h: int) -> None: ...

    def increment_attacks(self) -> None: ...

    def increment_teleports(self) -> None: ...


class NullOverlay:
    def set_scan_living(self, count: int) -> None:
        return None

    def set_track_stats(self, track_count: int, alive_count: int) -> None:
        return None

    def set_track_positions(self, positions: list[tuple[int, int]]) -> None:
        return None

    def set_search_roi(self, x: int, y: int, w: int, h: int) -> None:
        return None

    def increment_attacks(self) -> None:
        return None

    def increment_teleports(self) -> None:
        return None
