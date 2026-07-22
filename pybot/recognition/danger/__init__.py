"""Reusable danger assessment for hunt / sit / future callers.

For now the only signal is foreign sprites very near the player
(``near_objects``). More signals can be added behind ``assess_danger``
without changing callers.
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
    """Aggregate danger result for a single client frame."""

    in_danger: bool
    reasons: tuple[str, ...]
    near_object_count: int = 0


def assess_danger(
    frame_bgr: np.ndarray,
    *,
    cell_size_px: int = 64,
) -> DangerReport:
    """Assess whether the player is in immediate danger.

    Current rule: any non-player blocking/moving sprite inside a small
    radius around the character (see ``near_objects``).
    """
    near = count_near_objects(frame_bgr, cell_size_px=cell_size_px)
    reasons: list[str] = []
    if near.count > 0:
        reasons.append(f"near_objects:{near.count}")
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
