"""Debug output for the heatmap mob detector."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from pybot.recognition.detector.detector import DetectionResult


def save_debug_bundle(
    output_root: Path,
    image_name: str,
    frame_bgr: np.ndarray,
    result: DetectionResult,
) -> Path:
    label = Path(image_name).stem
    out_dir = output_root / result.mob_name / label
    out_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(out_dir / "input.png"), frame_bgr)
    cv2.imwrite(str(out_dir / "candidate_centers.png"), _candidate_centers(frame_bgr, result))
    cv2.imwrite(str(out_dir / "detected_overlay.png"), _detected_overlay(frame_bgr, result))
    (out_dir / "candidates.json").write_text(
        json.dumps([candidate.to_dict() for candidate in result.candidates], indent=2),
        encoding="utf-8",
    )
    (out_dir / "timing.json").write_text(json.dumps(result.timing, indent=2), encoding="utf-8")
    return out_dir


def _candidate_centers(frame_bgr: np.ndarray, result: DetectionResult) -> np.ndarray:
    canvas = frame_bgr.copy()
    for candidate in result.candidates:
        color = (0, 255, 0) if candidate.accepted else (0, 165, 255)
        cv2.circle(canvas, (candidate.center_x, candidate.center_y), 6, color, 2)
        cv2.putText(
            canvas,
            f"{candidate.final_score:.2f}",
            (candidate.center_x + 7, candidate.center_y - 7),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
        )
    return canvas


def _detected_overlay(frame_bgr: np.ndarray, result: DetectionResult) -> np.ndarray:
    canvas = frame_bgr.copy()
    for candidate in result.candidates:
        x, y, w, h = candidate.bbox
        color = (0, 255, 0) if candidate.accepted else (0, 0, 255)
        thickness = 2 if candidate.accepted else 1
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, thickness)
        cv2.circle(canvas, (candidate.center_x, candidate.center_y), 3, color, -1)
    return canvas
