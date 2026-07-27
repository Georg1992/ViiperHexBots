"""Discovery candidate filter"""

from __future__ import annotations

from pybot.runtime.detection.detector_session import RawDetection


def filter_scan_candidates(
    candidates: list[RawDetection],
) -> list[RawDetection]:
    return [candidate for candidate in candidates if candidate.living]
