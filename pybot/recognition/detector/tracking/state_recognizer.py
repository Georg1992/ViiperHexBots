"""Mob state recognition for known tracked mobs (not discovery)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pybot.recognition.detector.detector import STATE_PROFILE_DIRECT, STATE_PROFILE_FULL, StateSearchProfile

if TYPE_CHECKING:
    import numpy as np

    from pybot.recognition.detector.detector import MobDetector


def evaluate_track_state(
    detector: MobDetector,
    frame_bgr: np.ndarray,
    mob_name: str,
    track_id: int,
    x: int,
    y: int,
    *,
    offset_x: int = 0,
    offset_y: int = 0,
    scale_hint: float | None = None,
    profile: StateSearchProfile = STATE_PROFILE_FULL,
) -> dict:
    return detector.evaluate_track_state(
        frame_bgr,
        mob_name,
        track_id,
        x,
        y,
        offset_x=offset_x,
        offset_y=offset_y,
        scale_hint=scale_hint,
        profile=profile,
    )


def evaluate_track_state_direct(
    detector: MobDetector,
    frame_bgr: np.ndarray,
    mob_name: str,
    track_id: int,
    x: int,
    y: int,
    *,
    offset_x: int = 0,
    offset_y: int = 0,
    scale_hint: float | None = None,
) -> dict:
    return evaluate_track_state(
        detector,
        frame_bgr,
        mob_name,
        track_id,
        x,
        y,
        offset_x=offset_x,
        offset_y=offset_y,
        scale_hint=scale_hint,
        profile=STATE_PROFILE_DIRECT,
    )


def evaluate_track_states(
    detector: MobDetector,
    frame_bgr: np.ndarray,
    mob_name: str,
    tracks: list[dict],
    *,
    offset_x: int = 0,
    offset_y: int = 0,
    profile: StateSearchProfile = STATE_PROFILE_FULL,
) -> list[dict]:
    return detector.evaluate_track_states(
        frame_bgr,
        mob_name,
        tracks,
        offset_x=offset_x,
        offset_y=offset_y,
        profile=profile,
    )
