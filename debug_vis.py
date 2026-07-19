"""Debug visualization for the discovery pipeline.

Outputs under _debug_vis/:
  pipeline.txt                 — discovery pipeline structure (text)
  {mob}/descriptor.png         — descriptor fields used for that mob
  {mob}/{fixture}_viz.png      — heatmap | frame+boxes | silhouettes
"""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from pybot.paths import PROJECT_ROOT
from pybot.recognition.detector.descriptors.descriptor import (
    ColorCluster,
    MobDescriptor,
)
from pybot.recognition.detector.detector import (
    DetectionResult,
    MobDetector,
    SilhouetteCheck,
    load_detector_config,
)
from pybot.recognition.detector.discovery_pipeline import (
    assert_discovery_pipeline_matches_source,
    format_discovery_pipeline_text,
)
from pybot.recognition.fixtures import MOB_FIXTURE_SUITES, fixture_search_frame

OUT_DIR = Path("_debug_vis")

_SIL_SCALE = 10
_SIL_SIZE = 16 * _SIL_SCALE
_SWATCH = 28


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


def pad_to_height(image: np.ndarray, height: int) -> np.ndarray:
    if image.shape[0] >= height:
        return image
    pad = np.zeros((height - image.shape[0], image.shape[1], 3), dtype=np.uint8)
    pad[:] = (20, 20, 20)
    return np.vstack([image, pad])


# Jet full-scale maps to this absolute heat. Same scale every frame so
# brightness is comparable across fixtures / palette sizes (not /frame-max).
_HEATMAP_VIZ_ABS_SCALE = 1.0


def heatmap_to_color(heatmap: np.ndarray) -> np.ndarray:
    """Colorize sprite heatmap on a fixed absolute scale."""
    vis = (
        np.clip(heatmap / np.float32(_HEATMAP_VIZ_ABS_SCALE), 0.0, 1.0) * 255
    ).astype(np.uint8)
    return cv2.applyColorMap(vis, cv2.COLORMAP_JET)


def annotate_heatmap_pane(pane_heat: np.ndarray, result: DetectionResult) -> None:
    peak = float(result.sprite_heatmap.max()) if result.sprite_heatmap.size else 0.0
    cv2.putText(
        pane_heat, "HEATMAP", (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2,
    )
    cv2.putText(
        pane_heat,
        f"abs/{_HEATMAP_VIZ_ABS_SCALE:g}  peak={peak:.3f}",
        (10, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (220, 220, 220),
        1,
    )


def render_silhouette_grid(
    mask_avg: list[float],
    mask_stable: list[bool],
    size: int,
) -> np.ndarray:
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
    gate_masks = descriptor.silhouette_masks
    if check.matched_mask_index >= len(gate_masks):
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
    gate_masks = descriptor.silhouette_masks
    check_count = len(silhouette_checks)
    panel_height = _silhouette_panel_height(len(gate_masks), check_count, panel_height)
    panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
    panel[:] = (20, 20, 20)

    if not any(mask.stable_mask and any(mask.stable_mask) for mask in gate_masks):
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
        border_color = (0, 200, 0) if check.passed else (0, 0, 200)
        status = "PASS" if check.passed else "FAIL"
        ref_tag = f"ref={check.matched_mask_index}"
        if check.mask_similarities:
            score_bits = "/".join(f"{score:.2f}" for score in check.mask_similarities)
            ref_tag = f"{ref_tag} [{score_bits}]"
        extra_px = _candidate_pixels_outside_ref(check, descriptor)
        extra_tag = f"  out={extra_px}" if extra_px is not None else ""
        label = (
            f"BLB{idx}: {check.heat_score:.3f}  sim={check.similarity:.2f}  "
            f"{status}  {ref_tag}{extra_tag}"
        )
        cv2.putText(panel, label, (10, y_offset + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, border_color, 1)
        y_offset += 18

        if check.candidate_mask is None:
            cv2.putText(panel, "(no candidate)", (10, y_offset + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)
            y_offset += 24
            continue

        cand_img = render_silhouette_grid(
            check.candidate_mask, [True] * 256, cand_size,
        )
        cv2.rectangle(cand_img, (0, 0), (cand_size - 1, cand_size - 1), border_color, 3)
        panel[y_offset:y_offset + cand_size, 10:10 + cand_size] = cand_img
        y_offset += cand_size + 8

    return panel


def format_timing_ms(timing: dict[str, float]) -> str:
    order = (
        "descriptor", "spriteHeatmap", "blobCenters",
        "blobFilters", "silhouetteGate",
    )
    parts = [f"{key}={timing[key] * 1000:.0f}ms" for key in order if key in timing]
    total_ms = timing.get("total", 0.0) * 1000
    return "  ".join(parts) + f"  total={total_ms:.0f}ms"


def draw_timing_overlay(pane: np.ndarray, timing: dict[str, float], y0: int = 100) -> None:
    order = (
        ("spriteHeatmap", (0, 200, 255)),
        ("silhouetteGate", (0, 220, 0)),
        ("blobCenters", (255, 180, 0)),
        ("blobFilters", (180, 180, 180)),
        ("descriptor", (80, 80, 80)),
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


def draw_detection_overlay(frame: np.ndarray, result: DetectionResult) -> np.ndarray:
    overlay = frame.copy()
    silhouette_checks = result.silhouette_checks
    accepted_centers = {(c.center_x, c.center_y) for c in result.accepted}

    for idx, check in enumerate(silhouette_checks):
        cx, cy = check.center_x, check.center_y
        is_accepted = check.passed and (cx, cy) in accepted_centers

        if is_accepted:
            color = (0, 220, 0)
            thickness = 3
        elif check.passed:
            color = (0, 180, 255)
            thickness = 2
        else:
            color = (0, 120, 255)
            thickness = 2

        # Single box = exact palette-CC crop fed into silhouette check.
        crop = check.extract_bbox
        if crop is None:
            continue
        bx, by, bw, bh = crop
        cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), color, thickness)
        cv2.circle(overlay, (cx, cy), 5, color, -1)
        tag = f"{idx}:" + ("ACC" if is_accepted else ("SIL" if check.passed else "FAIL"))
        cv2.putText(
            overlay, tag, (bx, max(by - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1,
        )
        cv2.putText(
            overlay, f"{bw}x{bh}", (bx, min(by + bh + 14, overlay.shape[0] - 4)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1,
        )

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
        "green=accepted  cyan=sil-pass  orange=sil-fail  box=sil-crop",
        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1,
    )
    draw_timing_overlay(overlay, result.timing, y0=75)
    return overlay


def _bgr_swatch_row(
    colors: list[tuple[int, int, int]],
    weights: list[float] | None = None,
    cell: int = _SWATCH,
) -> np.ndarray:
    if not colors:
        blank = np.full((cell + 18, 120, 3), 40, dtype=np.uint8)
        cv2.putText(
            blank, "(empty)", (8, cell // 2 + 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140, 140, 140), 1,
        )
        return blank
    n = len(colors)
    row = np.full((cell + 18, n * (cell + 4) + 4, 3), 28, dtype=np.uint8)
    peak = max(weights) if weights else 1.0
    for i, bgr in enumerate(colors):
        x0 = 4 + i * (cell + 4)
        color = tuple(int(v) for v in bgr)
        cv2.rectangle(row, (x0, 2), (x0 + cell - 1, 2 + cell - 1), color, -1)
        cv2.rectangle(row, (x0, 2), (x0 + cell - 1, 2 + cell - 1), (200, 200, 200), 1)
        if weights is not None and i < len(weights):
            bar_h = max(1, int(round(14 * (weights[i] / max(peak, 1e-9)))))
            cv2.rectangle(
                row,
                (x0, cell + 16 - bar_h),
                (x0 + cell - 1, cell + 15),
                (180, 180, 80),
                -1,
            )
    return row


def _cluster_swatch_row(clusters: list[ColorCluster], cell: int = _SWATCH) -> np.ndarray:
    colors = [tuple(int(v) for v in c.bgr) for c in clusters]
    weights = [float(c.fraction) for c in clusters]
    return _bgr_swatch_row(colors, weights, cell=cell)


def _text_block(lines: list[str], width: int = 420, line_h: int = 18) -> np.ndarray:
    height = 12 + line_h * max(1, len(lines))
    canvas = np.full((height, width, 3), 28, dtype=np.uint8)
    y = 16
    for line in lines:
        cv2.putText(
            canvas, line, (8, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (210, 210, 210), 1, cv2.LINE_AA,
        )
        y += line_h
    return canvas


def render_descriptor_info(descriptor: MobDescriptor) -> np.ndarray:
    header = _text_block([
        f"{descriptor.mob_name}  v{descriptor.version}",
        f"size avg={descriptor.avg_width}x{descriptor.avg_height}",
        (
            f"matchPalette={len(descriptor.match_palette_bgr)}  "
            f"body={len(descriptor.body_palette)}  "
            f"accent={len(descriptor.accent_colors)}  "
            f"silMasks={len(descriptor.silhouette_masks)}"
        ),
    ], width=720)

    sections: list[tuple[str, np.ndarray]] = [
        ("MATCH PALETTE (weight bars)", _bgr_swatch_row(
            [tuple(int(v) for v in c) for c in descriptor.match_palette_bgr],
            list(descriptor.match_palette_weights),
        )),
        (
            f"PALETTE GROUPS ({len(descriptor.match_palette_groups)})",
            _text_block([
                f"g{i}: {group}"
                for i, group in enumerate(descriptor.match_palette_groups)
            ], width=720),
        ),
        ("DOMINANT", _cluster_swatch_row([descriptor.dominant_color])),
        ("SUPPORTING", _cluster_swatch_row(descriptor.supporting_colors)),
        ("ACCENT CLUSTERS", _cluster_swatch_row(descriptor.accent_colors)),
        (
            "STRUCTURAL PIXELS",
            _bgr_swatch_row([tuple(int(v) for v in p) for p in descriptor.dominant_pixels_bgr]),
        ),
        (
            "ACCENT PIXELS",
            _bgr_swatch_row([tuple(int(v) for v in p) for p in descriptor.accent_pixels_bgr]),
        ),
    ]

    sil_row_imgs: list[np.ndarray] = []
    for idx, mask in enumerate(descriptor.silhouette_masks):
        sil = render_silhouette_grid(mask.avg_mask, mask.stable_mask, _SIL_SIZE)
        labeled = np.full((_SIL_SIZE + 22, _SIL_SIZE, 3), 28, dtype=np.uint8)
        labeled[22:] = sil
        cv2.putText(
            labeled, f"SIL {idx}", (4, 14),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1,
        )
        sil_row_imgs.append(labeled)
    sections.append(("SILHOUETTE REFS", np.hstack(sil_row_imgs)))

    rows: list[np.ndarray] = [header]
    max_w = header.shape[1]
    for title, img in sections:
        block_w = max(img.shape[1], 200)
        title_bar = np.full((22, block_w, 3), 36, dtype=np.uint8)
        cv2.putText(
            title_bar, title, (6, 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 220, 255), 1, cv2.LINE_AA,
        )
        if img.shape[1] < block_w:
            pad = np.full((img.shape[0], block_w - img.shape[1], 3), 28, dtype=np.uint8)
            img = np.hstack([img, pad])
        block = np.vstack([title_bar, img])
        max_w = max(max_w, block.shape[1])
        rows.append(block)

    padded: list[np.ndarray] = []
    for row in rows:
        if row.shape[1] < max_w:
            pad = np.full((row.shape[0], max_w - row.shape[1], 3), 28, dtype=np.uint8)
            row = np.hstack([row, pad])
        padded.append(row)
    return np.vstack(padded)


def write_pipeline_structure(path: Path) -> None:
    assert_discovery_pipeline_matches_source()
    path.write_text(format_discovery_pipeline_text(), encoding="utf-8")


def main() -> None:
    config = load_detector_config()
    detector = MobDetector(PROJECT_ROOT, config)
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Generating debug viz at {generated_at}")

    pipeline_path = OUT_DIR / "pipeline.txt"
    write_pipeline_structure(pipeline_path)
    print(f"  wrote {pipeline_path}")

    viz_count = 0
    descriptor_count = 0
    timing_totals: dict[str, float] = {}
    timing_runs = 0

    for suite in MOB_FIXTURE_SUITES:
        mob_name = suite.mob_name
        try:
            descriptor = detector.ensure_descriptor(mob_name)
        except (FileNotFoundError, RuntimeError) as exc:
            print(f"  SKIP {suite.folder:15s}  {exc}")
            continue

        mob_dir = OUT_DIR / mob_name
        mob_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(mob_dir / "descriptor.png"), render_descriptor_info(descriptor))
        descriptor_count += 1
        print(f"  {mob_name:15s} wrote descriptor.png")

        for image in suite.images():
            frame = cv2.imread(str(image.path))
            if frame is None:
                continue

            frame = fixture_search_frame(frame)
            result = detector.detect(frame, mob_name)

            pane_heat = heatmap_to_color(result.sprite_heatmap)
            annotate_heatmap_pane(pane_heat, result)
            pane_overlay = draw_detection_overlay(frame, result)
            pane_sil = allocate_silhouette_panel(
                result.descriptor,
                result.silhouette_checks,
                350,
                frame.shape[0],
            )
            cv2.putText(pane_sil, "SILHOUETTES", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 2)

            combined_height = max(
                pane_heat.shape[0], pane_overlay.shape[0], pane_sil.shape[0],
            )
            combined = np.hstack([
                pad_to_height(pane_heat, combined_height),
                pad_to_height(pane_overlay, combined_height),
                pad_to_height(pane_sil, combined_height),
            ])

            stem = image.file_name.replace(".png", "")
            cv2.imwrite(str(mob_dir / f"{stem}_viz.png"), combined)
            viz_count += 1

            n_acc = len(result.accepted)
            expected = image.expected_count
            ok = "OK" if n_acc == expected else "FAIL"
            print(
                f"  {mob_name:15s} {stem:20s}  "
                f"expect={expected} got={n_acc}  "
                f"sil={len(result.silhouette_checks)}  {ok}"
            )
            print(f"    {format_timing_ms(result.timing)}")
            timing_runs += 1
            for key, sec in result.timing.items():
                timing_totals[key] = timing_totals.get(key, 0.0) + sec

    print(
        f"\nDone — {viz_count} viz, {descriptor_count} descriptors, "
        f"1 pipeline in {OUT_DIR.resolve()}/"
    )
    if timing_runs:
        print(f"\nAverage discovery timing over {timing_runs} frames:")
        order = (
            "descriptor", "spriteHeatmap", "blobCenters",
            "blobFilters", "silhouetteGate", "total",
        )
        avg_total = timing_totals.get("total", 0.0) / timing_runs
        for key in order:
            if key not in timing_totals:
                continue
            avg_ms = timing_totals[key] / timing_runs * 1000
            pct = (
                timing_totals[key] / timing_totals["total"] * 100
                if timing_totals.get("total") else 0
            )
            bar = "#" * max(1, int(pct / 2))
            print(f"  {key:16s} {avg_ms:6.0f}ms  ({pct:4.1f}%)  {bar}")
        print(f"  {'TOTAL':16s} {avg_total * 1000:6.0f}ms")


if __name__ == "__main__":
    main()
