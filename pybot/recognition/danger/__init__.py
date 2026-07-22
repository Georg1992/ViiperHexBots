"""Reusable danger assessment for hunt / sit / future callers.

Signals (extensible behind ``assess_danger``):
- ``near_objects`` — foreign sprites very near the player
- ``hp_drop`` — current HP lower than the previous sample (vision HP)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pybot.recognition.danger.near_objects import (
    NearObjectsResult,
    count_near_objects,
)


@dataclass(frozen=True)
class DangerReport:
    """Aggregate danger result for a single assessment."""

    in_danger: bool
    reasons: tuple[str, ...]
    near_object_count: int = 0

    @property
    def hp_dropped(self) -> bool:
        return any(reason.startswith("hp_drop:") for reason in self.reasons)

    @property
    def has_near_objects(self) -> bool:
        return self.near_object_count > 0


def assess_danger(
    frame_bgr: np.ndarray,
    *,
    cell_size_px: int = 64,
    hp: int | None = None,
    previous_hp: int | None = None,
) -> DangerReport:
    """Assess whether the player is in immediate danger.

    * ``near_objects`` — non-player sprites inside a small radius of the character
    * ``hp_drop`` — *hp* is set and strictly below *previous_hp*
    """
    near = count_near_objects(frame_bgr, cell_size_px=cell_size_px)
    reasons: list[str] = []
    if near.count > 0:
        reasons.append(f"near_objects:{near.count}")
    if (
        hp is not None
        and previous_hp is not None
        and hp < previous_hp
    ):
        reasons.append(f"hp_drop:{previous_hp}->{hp}")
    return DangerReport(
        in_danger=bool(reasons),
        reasons=tuple(reasons),
        near_object_count=near.count,
    )


__all__ = [
    "DangerReport",
    "NearObjectsResult",
    "assess_danger",
    "count_near_objects",
]
