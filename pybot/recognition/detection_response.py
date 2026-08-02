"""Pure transformations used by the mob-detection CLI and IPC server.

This module deliberately contains no capture, detector, filesystem, or Tk
logic. Keeping response shaping here makes the CLI orchestration thinner while
preserving its existing public helper names through imports in ``cli.py``.
"""

from __future__ import annotations


def apply_scale_calibration(
    config: dict,
    scale_range: tuple[float, float] | None,
    enforce_size_gate: bool,
) -> dict:
    """Return a calibrated detector config without mutating the input."""
    calibrated = dict(config)
    if scale_range is not None:
        low, high = scale_range
        mid = (low + high) / 2.0
        calibrated["scales"] = [low, mid, high]
        calibrated["centerScales"] = [low, mid, high]
    calibrated["enforceObjectSizeGate"] = enforce_size_gate
    return calibrated


def parse_request_scale_range(value) -> tuple[float, float] | None:
    """Normalize a two-item IPC scale range; invalid shapes remain unset."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) == 2:
        low, high = float(value[0]), float(value[1])
        return (min(low, high), max(low, high))
    return None


def candidate_to_json(candidate, screen_offset: tuple[int, int]) -> dict:
    """Serialize one detector candidate and translate it to screen coordinates."""
    ox, oy = screen_offset
    payload = candidate.to_dict()
    x, y, w, h = candidate.bbox
    payload.update(
        {
            "x": x + ox,
            "y": y + oy,
            "width": w,
            "height": h,
            "centerX": candidate.center_x + ox,
            "centerY": candidate.center_y + oy,
            "confidence": round(candidate.final_score, 4),
            "living": candidate.accepted,
        }
    )
    return payload


def build_detect_response(
    result,
    screen_offset: tuple[int, int],
    *,
    pipeline: str,
    session_id: str = "",
    scale_range: tuple[float, float] | None = None,
    enforce_size_gate: bool = False,
) -> dict:
    """Build the stable JSON payload shared by CLI and persistent server."""
    candidates = [candidate_to_json(candidate, screen_offset) for candidate in result.accepted]
    if pipeline == "scan":
        candidates = [item for item in candidates if item.get("living")]
    return {
        "ok": True,
        "pipeline": pipeline,
        "sessionId": session_id,
        "scaleCalibration": {
            "status": "locked" if scale_range else "discovering",
            "range": list(scale_range) if scale_range else None,
            "sizeGate": bool(enforce_size_gate),
        },
        "candidateCount": len(result.candidates),
        "acceptedCount": len(result.accepted),
        "elapsedS": round(result.elapsed_s, 4),
        "candidates": candidates,
    }


__all__ = [
    "apply_scale_calibration",
    "build_detect_response",
    "candidate_to_json",
    "parse_request_scale_range",
]
