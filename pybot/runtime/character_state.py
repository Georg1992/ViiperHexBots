"""CharacterState — dedicated store for character visual state.

Published by the CharacterStateMonitor worker and consumed by other
workers (attack loop, sit worker, danger detector) for decisions that
depend on character position and surrounding mobs.

Designed as a lightweight dataclass with thread-safe read/write so it
can be shared between the monitor thread and worker threads.

Note: pose (sit/stand) is not tracked here — it is verified on-demand
by the sit worker after pressing sit/stand keys (see ``character_pose``).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass


@dataclass
class _CharacterStateData:
    """Raw state snapshot written by CharacterStateMonitor each tick."""

    char_x: int = 0
    char_y: int = 0
    is_surrounded: bool = False
    surrounded_reason: str = ""
    nearby_mob_count: int = 0  # tracked (hunted) mobs near character
    nearby_any_mobs_count: int = 0  # ANY sprite blobs near character (visual det.)
    # Tick of the last frame capture (monotonic_ms) — consumers can age-out.
    tick_ms: int = 0


class CharacterState:
    """Thread-safe store for the latest character visual state.

    One instance per hunt session, shared between the monitor worker
    (writer) and worker threads (readers).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data = _CharacterStateData()

    # ── Writer (called by CharacterStateMonitor) ──────────────────

    def publish(
        self,
        *,
        char_x: int,
        char_y: int,
        is_surrounded: bool,
        surrounded_reason: str,
        nearby_mob_count: int,
        nearby_any_mobs_count: int = 0,
        tick_ms: int,
    ) -> None:
        with self._lock:
            self._data = _CharacterStateData(
                char_x=char_x,
                char_y=char_y,
                is_surrounded=is_surrounded,
                surrounded_reason=surrounded_reason,
                nearby_mob_count=nearby_mob_count,
                nearby_any_mobs_count=nearby_any_mobs_count,
                tick_ms=tick_ms,
            )

    def clear_area_threat(self) -> None:
        """Clear surround/nearby flags after teleport area reset.

        Prevents a stale ``is_surrounded`` from the previous screen from
        looking like danger on the new screen before the next charstate tick.
        """
        with self._lock:
            self._data = _CharacterStateData(
                char_x=self._data.char_x,
                char_y=self._data.char_y,
                is_surrounded=False,
                surrounded_reason="",
                nearby_mob_count=0,
                nearby_any_mobs_count=0,
                tick_ms=self._data.tick_ms,
            )

    # ── Readers ──────────────────────────────────────────────────

    @property
    def char_x(self) -> int:
        with self._lock:
            return self._data.char_x

    @property
    def char_y(self) -> int:
        with self._lock:
            return self._data.char_y

    @property
    def is_surrounded(self) -> bool:
        with self._lock:
            return self._data.is_surrounded

    @property
    def surrounded_reason(self) -> str:
        with self._lock:
            return self._data.surrounded_reason

    @property
    def nearby_mob_count(self) -> int:
        with self._lock:
            return self._data.nearby_mob_count

    @property
    def nearby_any_mobs_count(self) -> int:
        with self._lock:
            return self._data.nearby_any_mobs_count

    @property
    def tick_ms(self) -> int:
        with self._lock:
            return self._data.tick_ms

    def snapshot(self) -> _CharacterStateData:
        """Atomic read of all fields — returns a frozen dataclass copy."""
        with self._lock:
            return _CharacterStateData(
                char_x=self._data.char_x,
                char_y=self._data.char_y,
                is_surrounded=self._data.is_surrounded,
                surrounded_reason=self._data.surrounded_reason,
                nearby_mob_count=self._data.nearby_mob_count,
                nearby_any_mobs_count=self._data.nearby_any_mobs_count,
                tick_ms=self._data.tick_ms,
            )

    @property
    def char_pos(self) -> tuple[int, int]:
        return self.char_x, self.char_y


# ── Standalone surround detection ─────────────────────────────────

SURROUND_RADIUS_PX = 200


def is_surrounded_by_tracks(
    char_x: int,
    char_y: int,
    all_mobs: list[tuple[int, int]],
    radius_px: int = SURROUND_RADIUS_PX,
) -> tuple[bool, str]:
    """Check whether mobs box the character in from opposite sides.

    Returns ``(in_danger, reason)`` where *reason* describes which
    axes are blocked (``"left+right"``, ``"above+below"``, or ``""``).
    Only mobs within *radius_px* of the character are considered.

    A single nearby tracked mob never counts as surrounded — at least two
    distinct nearby positions on opposite sides of one axis are required.
    """
    if len(all_mobs) < 2:
        return False, ""

    # Filter to nearby mobs only
    nearby = []
    for mx, my in all_mobs:
        dx = mx - char_x
        dy = my - char_y
        if (dx * dx + dy * dy) <= radius_px * radius_px:
            nearby.append((mx, my))

    # One mob near the character is normal combat — not surrounded.
    if len(nearby) < 2:
        return False, ""

    left = any(mx < char_x for mx, _my in nearby)
    right = any(mx > char_x for mx, _my in nearby)
    above = any(my < char_y for _mx, my in nearby)
    below = any(my > char_y for _mx, my in nearby)

    if left and right:
        return True, "left+right"
    if above and below:
        return True, "above+below"
    return False, ""
