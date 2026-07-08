"""Discovery candidate filter"""

from __future__ import annotations

from pybot.runtime.capture.window_roi import HuntRoi
from pybot.runtime.detection.detector_session import RawDetection


def filter_scan_candidates(
    candidates: list[RawDetection],
    roi: HuntRoi,
    cell_size_px: int,
) -> list[RawDetection]:
    del roi, cell_size_px
    return [candidate for candidate in candidates if candidate.living]
