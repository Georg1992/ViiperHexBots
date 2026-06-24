"""Mob state recognition for known tracked mobs (not discovery)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

    from detector import SimpleMobDetector


def evaluate_track_state_direct(
    detector: SimpleMobDetector,
    frame_bgr: np.ndarray,
    mob_name: str,
    track_id: int,
    x: int,
    y: int,
    *,
    offset_x: int = 0,
    offset_y: int = 0,
) -> dict:
    return detector.evaluate_track_state_direct(
        frame_bgr,
        mob_name,
        track_id,
        x,
        y,
        offset_x=offset_x,
        offset_y=offset_y,
    )


def evaluate_track_states(
    detector: SimpleMobDetector,
    frame_bgr: np.ndarray,
    mob_name: str,
    tracks: list[dict],
    *,
    offset_x: int = 0,
    offset_y: int = 0,
) -> list[dict]:
    return detector.evaluate_track_states(
        frame_bgr,
        mob_name,
        tracks,
        offset_x=offset_x,
        offset_y=offset_y,
    )
