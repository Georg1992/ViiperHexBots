"""3-pane visual validation: heatmap | frame+boxes | silhouette comparison.

Uses MobDetector.detect() only — no duplicated pipeline logic.

Pane 1 — Heatmap:      jet-colormap sprite palette heatmap.
Pane 2 — Frame+boxes:  original frame with green = accepted, red = rejected.
Pane 3 — Silhouettes:  reference silhouette (from descriptor) vs extracted
                        silhouette for each blob, with similarity score.

Output:  _heatmap_viz/{mob}/{fixture_name}_viz.png
"""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from pybot.paths import PROJECT_ROOT
from pybot.recognition.detector.descriptors.descriptor import MobDescriptor
from pybot.recognition.detector.detector import (
    DetectionResult,
    MobDetector,
    SilhouetteCheck,
    load_detector_config,
)
from pybot.recognition.fixtures import MOB_FIXTURE_SUITES, fixture_search_frame

OUT_DIR = Path("_heatmap_viz")

_SIL_SCALE = 10  # 16×10 = 160 px per silhouette cell
_SIL_SIZE = 16 * _SIL_SCALE  # 160 px
_SIL_OVERLAY_PX = 40  # extracted silhouette thumbnail on the frame


def _candidate_sil_size(check_count: int) -> int:
    if check_count > 12:
        return _SIL_SIZE // 4
    if check_count > 6:
        return _SIL_SIZE // 2
    return _SIL_SIZE


def _ref_sil_size(ref_count: int) -> int:
    return _SIL_SIZE if ref_count <= 2 else _SIL_SIZE // 2


def _silhouette_panel_height(
    ref_count: int,
    check_count: int,
    min_height: int,
) -> int:
    ref_size = _ref_sil_size(ref_count)
    cand_size = _candidate_sil_size(check_count)
    refs_h = 10 + ref_count * (20 + ref_size + 8) + 4
    rows_h = check_count * (18 + cand_size + 8)
    return max(min_height, refs_h + rows_h + 20)


def _paste_image(
    canvas: np.ndarray,
    image: np.ndarray,
    x: int,
    y: int,
) -> None:
    h, w = image.shape[:2]
    fh, fw = canvas.shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(fw, x + w)
    y2 = min(fh, y + h)
    if x2 <= x1 or y2 <= y1:
        return
    src_x1 = x1 - x
    src_y1 = y1 - y
    canvas[y1:y2, x1:x2] = image[
        src_y1:src_y1 + (y2 - y1),
        src_x1:src_x1 + (x2 - x1),
    ]


def pad_to_height(image: np.ndarray, height: int) -> np.ndarray:
    if image.shape[0] >= height:
        return image
    pad = np.zeros((height - image.shape[0], image.shape[1], 3), dtype=np.uint8)
    pad[:] = (20, 20, 20)
    return np.vstack([image, pad])


def heatmap_to_color(heatmap: np.ndarray) -> np.ndarray:
    vis = (np.clip(heatmap, 0.0, 1.0) * 255).astype(np.uint8)
    return cv2.applyColorMap(vis, cv2.COLORMAP_JET)


def annotate_heatmap_pane(pane_heat: np.ndarray, result: DetectionResult) -> None:
    """Label the heatmap pane."""
    cv2.putText(
        pane_heat,
        "HEATMAP",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )


def render_silhouette_grid(mask_avg: list[float], mask_stable: list[bool],
                           size: int) -> np.ndarray:
    """Render a 16×16 silhouette as a size×size RGB image.

    Stable cells: white if occupied, black if empty.
    Unstable cells: gray with a soft outline.
    """
    avg = np.array(mask_avg).reshape(16, 16)
    stable = np.array(mask_stable).reshape(16, 16)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    cell = size // 16
    for y in range(16):
        for x in range(16):
            if stable[y, x]:
                val = int(np.clip(avg[y, x] * 255, 0, 255))
                color = (val, val, val) if val > 40 else (0, 0, 0)
            else:
                v = int(np.clip(avg[y, x] * 80, 0, 80))
                color = (v, v, v)
            cy, cx = y * cell, x * cell
            canvas[cy:cy + cell, cx:cx + cell] = color
    # Draw grid lines
    for i in range(17):
        cv2.line(canvas, (i * cell, 0), (i * cell, size), (60, 60, 60), 1)
        cv2.line(canvas, (0, i * cell), (size, i * cell), (60, 60, 60), 1)
    return canvas


def _candidate_pixels_outside_ref(
    check: SilhouetteCheck,
    descriptor: MobDescriptor,
) -> int | None:
    if check.candidate_mask is None:
        return None
    gate_masks = descriptor.silhouette_masks or []
    if not gate_masks or check.matched_mask_index >= len(gate_masks):
        return None
    mask = gate_masks[check.matched_mask_index]
    cand = np.array(check.candidate_mask, dtype=np.float32).reshape(16, 16)
    ref = np.array(mask.avg_mask, dtype=np.float32).reshape(16, 16)
    stable = np.array(mask.stable_mask, dtype=bool).reshape(16, 16)
    ref_bin = (ref >= 0.5) & stable
    cand_bin = cand >= 0.5
    return int(np.sum(cand_bin & ~ref_bin))


def allocate_silhouette_panel(
    descriptor: MobDescriptor,
    silhouette_checks: list[SilhouetteCheck],
    panel_width: int,
    panel_height: int,
) -> np.ndarray:
    """Build the silhouette comparison panel from production gate results."""
    gate_masks = descriptor.silhouette_masks or []
    check_count = len(silhouette_checks)
    panel_height = _silhouette_panel_height(len(gate_masks), check_count, panel_height)
    panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
    panel[:] = (20, 20, 20)

    if not gate_masks or not any(
        mask.stable_mask and any(mask.stable_mask) for mask in gate_masks
    ):
        cv2.putText(panel, "NO SILHOUETTE", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
        return panel

    ref_size = _ref_sil_size(len(gate_masks))
    cand_size = _candidate_sil_size(check_count)

    y_offset = 10
    for mask_idx, mask in enumerate(gate_masks):
        cv2.putText(
            panel, f"REF {mask_idx}", (10, y_offset + 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1,
        )
        y_offset += 20
        ref_img = render_silhouette_grid(mask.avg_mask, mask.stable_mask, ref_size)
        panel[y_offset:y_offset + ref_size, 10:10 + ref_size] = ref_img
        y_offset += ref_size + 8
    y_offset += 4

    for idx, check in enumerate(silhouette_checks):
        passed = check.passed
        sim = check.similarity
        heat_score = check.heat_score

        border_color = (0, 200, 0) if passed else (0, 0, 200)
        status = "PASS" if passed else "FAIL"
        ref_tag = f"ref={check.matched_mask_index}"
        if check.mask_similarities:
            score_bits = "/".join(f"{score:.2f}" for score in check.mask_similarities)
            ref_tag = f"{ref_tag} [{score_bits}]"
        extra_px = _candidate_pixels_outside_ref(check, descriptor)
        extra_tag = f"  out={extra_px}" if extra_px is not None else ""
        label = f"BLB{idx}: {heat_score:.3f}  sim={sim:.2f}  {status}  {ref_tag}{extra_tag}"
        cv2.putText(panel, label, (10, y_offset + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, border_color, 1)
        y_offset += 18

        if check.candidate_mask is None:
            cv2.putText(panel, "(no candidate)", (10, y_offset + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)
            y_offset += 24
            continue

        cand_img = render_silhouette_grid(
            check.candidate_mask,
            [True] * 256,
            cand_size,
        )
        cv2.rectangle(cand_img, (0, 0), (cand_size - 1, cand_size - 1), border_color, 3)
        panel[y_offset:y_offset + cand_size, 10:10 + cand_size] = cand_img
        y_offset += cand_size + 8

    return panel


def draw_extracted_silhouette_on_frame(
    overlay: np.ndarray,
    check: SilhouetteCheck,
    idx: int,
) -> None:
    """Draw the production-extracted silhouette thumbnail on the frame."""
    if check.candidate_mask is None:
        return

    sil_px = _SIL_OVERLAY_PX
    bx, by, bw, bh = check.bbox
    border_color = (0, 220, 0) if check.passed else (0, 80, 255)

    mini = render_silhouette_grid(check.candidate_mask, [True] * 256, sil_px)
    cv2.rectangle(mini, (0, 0), (sil_px - 1, sil_px - 1), border_color, 2)

    fh, fw = overlay.shape[:2]
    x = bx + bw - sil_px
    y = by - sil_px - 14
    if y < 0:
        y = by + bh + 4
    if x + sil_px > fw:
        x = max(0, bx)
    if y + sil_px > fh:
        y = max(0, by - sil_px - 14)

    _paste_image(overlay, mini, x, y)
    cv2.putText(
        overlay, f"{idx}", (x, max(y - 4, 12)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.4, border_color, 1,
    )


def format_timing_ms(timing: dict[str, float]) -> str:
    """Compact single-line timing summary for discovery scan stages."""
    order = (
        "descriptor",
        "hsv",
        "spriteHeatmap",
        "accentHeatmap",
        "blobCenters",
        "blobFilters",
        "silhouetteGate",
        "nms",
    )
    parts = [f"{key}={timing[key] * 1000:.0f}ms" for key in order if key in timing]
    total_ms = timing.get("total", 0.0) * 1000
    return "  ".join(parts) + f"  total={total_ms:.0f}ms"


def draw_timing_overlay(pane: np.ndarray, timing: dict[str, float], y0: int = 100) -> None:
    """Stacked timing bars for each discovery stage."""
    order = (
        ("spriteHeatmap", (0, 200, 255)),
        ("silhouetteGate", (0, 220, 0)),
        ("blobCenters", (255, 180, 0)),
        ("accentHeatmap", (200, 120, 255)),
        ("blobFilters", (180, 180, 180)),
        ("hsv", (120, 120, 120)),
        ("descriptor", (80, 80, 80)),
        ("nms", (255, 255, 255)),
    )
    total = max(timing.get("total", 0.0), 1e-9)
    bar_max_w = min(280, pane.shape[1] - 20)
    line_h = 14
    cv2.putText(
        pane, "TIMING", (10, y0),
        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1,
    )
    y = y0 + 16
    for key, color in order:
        if key not in timing:
            continue
        sec = timing[key]
        if sec < 1e-6:
            continue
        ms = sec * 1000
        bar_w = max(2, int(bar_max_w * sec / total))
        cv2.rectangle(pane, (10, y - 9), (10 + bar_w, y + 2), color, -1)
        cv2.putText(
            pane, f"{key} {ms:.0f}ms", (16 + bar_max_w, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1,
        )
        y += line_h


def draw_detection_overlay(
    frame: np.ndarray,
    result: DetectionResult,
) -> np.ndarray:
    """Draw silhouette-checked blobs and final accepted detections on the frame."""
    overlay = frame.copy()
    silhouette_checks = result.silhouette_checks
    accepted_centers = {(c.center_x, c.center_y) for c in result.accepted}

    for idx, check in enumerate(silhouette_checks):
        cx, cy = check.center_x, check.center_y
        bx, by, bw, bh = check.bbox
        nms_accepted = check.passed and (cx, cy) in accepted_centers

        if nms_accepted:
            color = (0, 220, 0)
            thickness = 3
        elif check.passed:
            color = (0, 180, 255)  # cyan: passed silhouette, NMS suppressed
            thickness = 2
        else:
            color = (0, 120, 255)  # orange: reached silhouette gate, failed
            thickness = 2

        cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), color, thickness)
        cv2.circle(overlay, (cx, cy), 5, color, -1)
        tag = f"{idx}:" + ("ACC" if nms_accepted else ("SIL" if check.passed else "FAIL"))
        cv2.putText(
            overlay, tag, (bx, max(by - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1,
        )
        draw_extracted_silhouette_on_frame(overlay, check, idx)

    n_sil = len(silhouette_checks)
    n_sil_pass = sum(1 for c in silhouette_checks if c.passed)
    n_acc = len(result.accepted)
    cv2.putText(
        overlay,
        f"Sil:{n_sil} Pass:{n_sil_pass} Acc:{n_acc}  {result.elapsed_s * 1000:.0f}ms",
        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2,
    )
    cv2.putText(
        overlay,
        "green=accepted  cyan=sil-pass  orange=sil-fail  #=BLB index",
        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1,
    )
    draw_timing_overlay(overlay, result.timing, y0=75)
    return overlay


def main():
    config = load_detector_config()
    detector = MobDetector(PROJECT_ROOT, config)
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Generating fresh visualizations at {generated_at}")

    count = 0
    timing_totals: dict[str, float] = {}
    timing_runs = 0
    for suite in MOB_FIXTURE_SUITES:
        mob_name = suite.mob_name
        try:
            detector.ensure_descriptor(mob_name)
        except (FileNotFoundError, RuntimeError) as exc:
            print(f"  SKIP {suite.folder:15s}  {exc}")
            continue

        for image in suite.images():
            frame = cv2.imread(str(image.path))
            if frame is None:
                continue

            frame = fixture_search_frame(frame)
            result = detector.detect(frame, mob_name)

            pane_heat = heatmap_to_color(result.sprite_heatmap)
            annotate_heatmap_pane(pane_heat, result)

            pane_overlay = draw_detection_overlay(frame, result)

            panel_w = 350
            panel_h = frame.shape[0]
            pane_sil = allocate_silhouette_panel(
                result.descriptor,
                result.silhouette_checks,
                panel_w,
                panel_h,
            )
            cv2.putText(pane_sil, "SILHOUETTES", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 2)

            combined_height = max(pane_heat.shape[0], pane_overlay.shape[0], pane_sil.shape[0])
            pane_heat = pad_to_height(pane_heat, combined_height)
            pane_overlay = pad_to_height(pane_overlay, combined_height)
            pane_sil = pad_to_height(pane_sil, combined_height)

            combined = np.hstack([pane_heat, pane_overlay, pane_sil])

            out_path = OUT_DIR / mob_name / f"{image.file_name.replace('.png', '')}_viz.png"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), combined)
            count += 1

            n_acc = len(result.accepted)
            expected = image.expected_count
            ok = "OK" if n_acc == expected else "FAIL"
            n_sil = len(result.silhouette_checks)
            print(
                f"  {mob_name:15s} {image.file_name.replace('.png', ''):20s}  "
                f"expect={expected} got={n_acc}  sil={n_sil}  {ok}"
            )
            print(f"    {format_timing_ms(result.timing)}")
            timing_runs += 1
            for key, sec in result.timing.items():
                timing_totals[key] = timing_totals.get(key, 0.0) + sec

    print(f"\nDone — {count} visualizations in {OUT_DIR.resolve()}/")
    print(f"Generated at {generated_at}")
    if timing_runs:
        print(f"\nAverage discovery timing over {timing_runs} frames:")
        order = (
            "descriptor", "hsv", "spriteHeatmap", "accentHeatmap",
            "blobCenters", "blobFilters", "silhouetteGate", "nms", "total",
        )
        avg_total = timing_totals.get("total", 0.0) / timing_runs
        for key in order:
            if key not in timing_totals:
                continue
            avg_ms = timing_totals[key] / timing_runs * 1000
            pct = (timing_totals[key] / timing_totals["total"] * 100) if timing_totals.get("total") else 0
            bar = "#" * max(1, int(pct / 2))
            print(f"  {key:16s} {avg_ms:6.0f}ms  ({pct:4.1f}%)  {bar}")
        print(f"  {'TOTAL':16s} {avg_total * 1000:6.0f}ms")


if __name__ == "__main__":
    main()
