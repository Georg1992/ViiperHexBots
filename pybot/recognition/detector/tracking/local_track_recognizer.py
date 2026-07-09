"""Thin wrapper around ``track_local`` for the recognition CLI and tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pybot.recognition.detector.tracking.local_tracker import LocalTrackResult, track_local

if TYPE_CHECKING:
    import numpy as np

    from pybot.recognition.detector.detector import MobDetector


def follow_track_local(
    detector: MobDetector,
    frame_bgr: np.ndarray,
    mob_name: str,
    track: dict,
    *,
    offset_x: int = 0,
    offset_y: int = 0,
    search_radius_px: int | None = None,
    death_detection_enabled: bool = False,
) -> LocalTrackResult:
    return track_local(
        detector,
        frame_bgr,
        mob_name,
        track,
        offset_x=offset_x,
        offset_y=offset_y,
        search_radius_px=search_radius_px,
        death_detection_enabled=death_detection_enabled,
    )
