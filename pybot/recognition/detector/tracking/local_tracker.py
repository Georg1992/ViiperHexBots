"""Local coordinate follower for already-discovered tracks.

Tracks by looking for any pixel whose BGR exactly matches the mob's sprite
palette within a tight search window, then picks the nearest exact match.
No distance-based heatmap, no silhouette gate — tracking is pure follow,
discovery handles all confirmation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pybot.recognition.detector.detector import MobDetector
    from pybot.recognition.detector.descriptors.descriptor import MobDescriptor


@dataclass(frozen=True)
class LocalTrackResult:
    track_id: int
    found: bool
    x: int
    y: int
    confidence: float
    miss_reason: str
    # Number of exact palette-match pixels in the search window.
    # Used by apply_tracking() to detect palette decay (corpse fading).
    match_count: int = 0


def _exact_palette_map(
    crop_bgr: np.ndarray,
    palette_bgr: list[tuple[int, int, int]],
) -> np.ndarray:
    """Binary map: 1 where pixel BGR exactly matches any palette color, 0 elsewhere."""
    if not palette_bgr or crop_bgr.size == 0:
        return np.zeros(crop_bgr.shape[:2], dtype=bool)
    pixels = crop_bgr.reshape(-1, 3)
    palette = np.asarray(palette_bgr, dtype=np.uint8)  # (C, 3)
    matches = np.any(np.all(pixels[:, None, :] == palette[None, :, :], axis=2), axis=1)
    return matches.reshape(crop_bgr.shape[:2])


def track_local(
    detector: MobDetector,
    frame_bgr: np.ndarray,
    mob_name: str,
    track: dict,
    *,
    offset_x: int = 0,
    offset_y: int = 0,
    search_radius_px: int | None = None,
) -> LocalTrackResult:
    """Follow one known track near its last known position.

    Looks for exact BGR palette matches in a tight search window and picks
    the nearest match to the predicted position. No silhouette gate — that's
    discovery's job.
    """
    track_id = int(track["trackId"])
    cx = int(track["x"])
    cy = int(track["y"])
    scale = float(track.get("scale", 1.0))
    moving = bool(track.get("moving", False))
    vel_x = float(track.get("velX", 0.0))
    vel_y = float(track.get("velY", 0.0))
    discovery_obs_tick = int(track.get("discoveryObsTick", 0))
    discovery_obs_x = int(track.get("discoveryObsX", 0))
    discovery_obs_y = int(track.get("discoveryObsY", 0))
    lost_count = int(track.get("lostCount", 0))
    track_palette = track.get("trackPaletteBgr", None)

    if search_radius_px is not None:
        radius = int(search_radius_px)
    elif moving or lost_count > 0:
        radius = int(detector.local_track_moving_search_radius_px)
    else:
        radius = int(detector.local_track_search_radius_px)

    # Predicted position: coast velocity when moving
    predicted_x = int(round(cx + vel_x)) if moving else cx
    predicted_y = int(round(cy + vel_y)) if moving else cy
    search_x, search_y = predicted_x, predicted_y

    # Use discovery soft prior when local follow is drifted or missing
    if discovery_obs_tick > 0:
        dedup_radius = int(detector.config["trackDedupRadiusPx"])
        drift_sq = (cx - discovery_obs_x) ** 2 + (cy - discovery_obs_y) ** 2
        half_dedup_sq = (dedup_radius // 2) ** 2
        if lost_count > 0 or drift_sq > half_dedup_sq:
            search_x = discovery_obs_x
            search_y = discovery_obs_y

    screen_cx = cx + offset_x
    screen_cy = cy + offset_y

    if not track_palette:
        # No sampled palette → can't confirm exact colors → miss.
        return _miss(track_id, screen_cx, screen_cy, reason="no_track_palette")

    descriptor = detector.ensure_descriptor(mob_name)
    frame_h, frame_w = frame_bgr.shape[:2]

    # Crop a tight search window around the predicted position
    margin_x = int(round(descriptor.avg_width * scale * 0.6))
    margin_y = int(round(descriptor.avg_height * scale * 0.6))
    pad = radius + max(margin_x, margin_y)
    x0 = max(0, search_x - pad)
    y0 = max(0, search_y - pad)
    x1 = min(frame_w, search_x + pad + 1)
    y1 = min(frame_h, search_y + pad + 1)
    if x1 <= x0 or y1 <= y0:
        return _miss(track_id, screen_cx, screen_cy, reason="out_of_bounds")

    crop_bgr = frame_bgr[y0:y1, x0:x1]

    # Exact BGR palette match against the track's sampled palette
    # (sampled from the actual frame at track creation time).
    exact_map = _exact_palette_map(crop_bgr, track_palette)
    if not np.any(exact_map):
        return _miss(track_id, screen_cx, screen_cy, reason="no_exact_palette_match")

    # Find the nearest exact match within the circular search window
    anchor_x = search_x - x0
    anchor_y = search_y - y0
    yy, xx = np.ogrid[: exact_map.shape[0], : exact_map.shape[1]]
    dist_sq = (xx - anchor_x) ** 2 + (yy - anchor_y) ** 2
    window = dist_sq <= (radius * radius)
    candidates = window & exact_map

    if not np.any(candidates):
        return _miss(track_id, screen_cx, screen_cy, reason="no_match_nearby")

    candidate_dists = np.where(candidates, dist_sq, np.inf)
    nearest_idx = int(candidate_dists.argmin())
    nearest_y, nearest_x = np.unravel_index(nearest_idx, candidate_dists.shape)
    hit_x = int(nearest_x + x0) + offset_x
    hit_y = int(nearest_y + y0) + offset_y

    match_count = int(candidates.sum())

    return LocalTrackResult(
        track_id=track_id, found=True, x=hit_x, y=hit_y,
        confidence=1.0, miss_reason="", match_count=match_count,
    )


def _miss(
    track_id: int,
    x: int,
    y: int,
    reason: str,
) -> LocalTrackResult:
    return LocalTrackResult(
        track_id=track_id, found=False, x=x, y=y,
        confidence=0.0, miss_reason=reason,
    )
