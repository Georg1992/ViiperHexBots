"""Local track coordinate follower — not discovery, not state/death confirm."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tracking.local_tracker import LocalTrackResult, track_local  # noqa: F401

if TYPE_CHECKING:
    import numpy as np

    from detector import SimpleMobDetector


def follow_track_local(
    detector: SimpleMobDetector,
    frame_bgr: np.ndarray,
    mob_name: str,
    track: dict,
    *,
    offset_x: int = 0,
    offset_y: int = 0,
    search_radius_px: int | None = None,
) -> LocalTrackResult:
    return track_local(
        detector,
        frame_bgr,
        mob_name,
        track,
        offset_x=offset_x,
        offset_y=offset_y,
        search_radius_px=search_radius_px,
    )
