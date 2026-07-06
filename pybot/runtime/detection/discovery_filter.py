"""Discovery candidate filter"""

from __future__ import annotations

from pybot.runtime.capture.window_roi import HuntRoi, player_ignore_box, point_inside_ignore
from pybot.runtime.detection.detector_session import RawDetection


def filter_scan_candidates(
    candidates: list[RawDetection],
    roi: HuntRoi,
    cell_size_px: int,
) -> list[RawDetection]:
    ignore_x, ignore_y, ignore_w, ignore_h = player_ignore_box(roi, cell_size_px)
    filtered: list[RawDetection] = []
    for candidate in candidates:
        if not candidate.living or candidate.dead:
            continue
        if point_inside_ignore(
            candidate.x,
            candidate.y,
            ignore_x,
            ignore_y,
            ignore_w,
            ignore_h,
        ):
            continue
        filtered.append(candidate)
    return filtered
