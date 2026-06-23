"""Debug output for the simple heatmap detector."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from detector import SimpleDetectionResult


def _heatmap_png(heatmap: np.ndarray) -> np.ndarray:
    normalized = np.zeros_like(heatmap, dtype=np.uint8)
    if heatmap.size and float(heatmap.max()) > 0:
        normalized = np.clip(heatmap / float(heatmap.max()) * 255.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(normalized, cv2.COLORMAP_JET)


def save_simple_debug_bundle(
    output_root: Path,
    image_name: str,
    frame_bgr: np.ndarray,
    result: SimpleDetectionResult,
) -> Path:
    label = Path(image_name).stem
    out_dir = output_root / result.mob_name / label
    out_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(out_dir / "input.png"), frame_bgr)
    cv2.imwrite(str(out_dir / "body_palette_heatmap.png"), _heatmap_png(result.heatmaps.body_palette))
    cv2.imwrite(str(out_dir / "accent_heatmap.png"), _heatmap_png(result.heatmaps.accent))
    cv2.imwrite(str(out_dir / "rare_color_heatmap.png"), _heatmap_png(result.heatmaps.rare_color))
    cv2.imwrite(str(out_dir / "local_pattern_heatmap.png"), _heatmap_png(result.heatmaps.local_pattern))
    cv2.imwrite(str(out_dir / "final_center_heatmap.png"), _heatmap_png(result.heatmaps.final_center))
    cv2.imwrite(str(out_dir / "candidate_centers.png"), _candidate_centers(frame_bgr, result))
    cv2.imwrite(str(out_dir / "detected_overlay.png"), _detected_overlay(frame_bgr, result))
    (out_dir / "candidates.json").write_text(
        json.dumps([candidate.to_dict() for candidate in result.candidates], indent=2),
        encoding="utf-8",
    )
    (out_dir / "timing.json").write_text(json.dumps(result.timing, indent=2), encoding="utf-8")
    return out_dir


def _candidate_centers(frame_bgr: np.ndarray, result: SimpleDetectionResult) -> np.ndarray:
    canvas = frame_bgr.copy()
    for candidate in result.candidates:
        color = (0, 255, 0) if candidate.accepted else (0, 165, 255)
        cv2.circle(canvas, (candidate.center_x, candidate.center_y), 6, color, 2)
        cv2.putText(
            canvas,
            f"{candidate.final_score:.2f}",
            (candidate.center_x + 7, candidate.center_y - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    return canvas


def _detected_overlay(frame_bgr: np.ndarray, result: SimpleDetectionResult) -> np.ndarray:
    canvas = frame_bgr.copy()
    for candidate in result.candidates:
        x, y, w, h = candidate.bbox
        color = (0, 255, 0) if candidate.accepted else (0, 0, 255)
        thickness = 2 if candidate.accepted else 1
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, thickness)
        cv2.circle(canvas, (candidate.center_x, candidate.center_y), 3, color, -1)
    return canvas


def save_summary_contact_sheet(output_path: Path, image_paths: list[Path]) -> None:
    tiles = []
    for path in image_paths:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        thumb = cv2.resize(img, (320, 180), interpolation=cv2.INTER_AREA)
        cv2.putText(thumb, path.parent.name, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        tiles.append(thumb)
    if not tiles:
        return
    cols = min(3, len(tiles))
    rows = int(np.ceil(len(tiles) / cols))
    sheet = np.zeros((rows * 180, cols * 320, 3), dtype=np.uint8)
    for idx, tile in enumerate(tiles):
        row, col = divmod(idx, cols)
        sheet[row * 180 : (row + 1) * 180, col * 320 : (col + 1) * 320] = tile
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), sheet)
