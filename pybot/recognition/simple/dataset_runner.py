"""Fixture runner for the simple heatmap detector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from pybot.paths import PROJECT_ROOT, RECOGNITION_DIR
from pybot.recognition.simple.debug_renderer import save_simple_debug_bundle, save_summary_contact_sheet
from pybot.recognition.simple.descriptors.descriptor_builder import SimpleDescriptorBuilder
from pybot.recognition.simple.detector import SimpleMobDetector, load_simple_config


def _load_ground_truth(image_dir: Path, entry: dict) -> list[dict]:
    file_name = entry["file"]
    json_path = image_dir / f"{Path(file_name).stem}.json"
    if json_path.exists():
        data = json.loads(json_path.read_text(encoding="utf-8"))
        return data["horns"]
    if int(entry["expectHorns"]) == 0:
        return []
    raise FileNotFoundError(f"missing ground truth JSON for positive fixture: {json_path}")


def _manifest_entries(fixtures_dir: Path) -> list[dict]:
    manifest_path = fixtures_dir / "manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    return list(data["images"])


def _match_counts(accepted, ground_truth: list[dict]) -> tuple[int, int, int]:
    matched_candidates: set[int] = set()
    tp = 0
    for gt in ground_truth:
        gx, gy = int(gt["centerX"]), int(gt["centerY"])
        radius = int(gt["radius"])
        radius_sq = radius * radius
        best_idx = None
        best_score = -1.0
        for idx, candidate in enumerate(accepted):
            if idx in matched_candidates:
                continue
            dist_sq = (candidate.center_x - gx) ** 2 + (candidate.center_y - gy) ** 2
            if dist_sq <= radius_sq and candidate.final_score > best_score:
                best_idx = idx
                best_score = candidate.final_score
        if best_idx is not None:
            matched_candidates.add(best_idx)
            tp += 1
    fp = len(accepted) - tp
    fn = len(ground_truth) - tp
    return tp, fp, fn


def run_fixtures(mob_name: str, fixtures_dir: Path, *, debug: bool, rebuild_descriptor: bool) -> dict:
    if rebuild_descriptor:
        SimpleDescriptorBuilder(PROJECT_ROOT).build(mob_name, force=True)
    config = load_simple_config()
    detector = SimpleMobDetector(PROJECT_ROOT, config)
    image_dir = fixtures_dir / "game-screenshots"
    manifest = json.loads((fixtures_dir / "manifest.json").read_text(encoding="utf-8"))
    debug_root = PROJECT_ROOT / config["debugOutputDir"]
    summary = {
        "mobName": mob_name,
        "pipeline": "simple",
        "images": [],
        "totals": {"tp": 0, "fp": 0, "fn": 0},
    }
    overlay_paths: list[Path] = []
    for entry in _manifest_entries(fixtures_dir):
        file_name = entry["file"]
        image_path = image_dir / file_name
        frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        result = detector.detect(frame, mob_name)
        gt = _load_ground_truth(image_dir, entry)
        tp, fp, fn = _match_counts(result.accepted, gt)
        summary["totals"]["tp"] += tp
        summary["totals"]["fp"] += fp
        summary["totals"]["fn"] += fn
        best_scores = [round(c.final_score, 4) for c in sorted(result.candidates, key=lambda c: c.final_score, reverse=True)[:5]]
        row = {
            "file": file_name,
            "expected": len(gt),
            "accepted": len(result.accepted),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "bestScores": best_scores,
            "elapsedS": round(result.elapsed_s, 4),
        }
        summary["images"].append(row)
        print(
            f"{file_name}: expected={row['expected']} accepted={row['accepted']} "
            f"TP={tp} FP={fp} FN={fn} best={best_scores} time={row['elapsedS']}s"
        )
        if debug:
            out_dir = save_simple_debug_bundle(debug_root, file_name, frame, result)
            overlay_paths.append(out_dir / "detected_overlay.png")
    if debug:
        summary_dir = debug_root / mob_name
        summary_dir.mkdir(parents=True, exist_ok=True)
        (summary_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        save_summary_contact_sheet(summary_dir / "summary_contact_sheet.png", overlay_paths)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run simple detector fixture suite")
    parser.add_argument("--mob", required=True)
    parser.add_argument("--fixtures", default=str(RECOGNITION_DIR / "test-fixtures"))
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--rebuild-descriptor", action="store_true")
    args = parser.parse_args(argv)
    run_fixtures(args.mob.lower(), Path(args.fixtures), debug=args.debug, rebuild_descriptor=args.rebuild_descriptor)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
