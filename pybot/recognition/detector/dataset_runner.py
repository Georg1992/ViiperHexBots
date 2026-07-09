"""Fixture runner for the heatmap mob detector."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2

from pybot.paths import PROJECT_ROOT, RECOGNITION_DIR
from pybot.recognition.detector.debug_renderer import save_debug_bundle, save_summary_contact_sheet
from pybot.recognition.detector.descriptors.descriptor_builder import DescriptorBuilder
from pybot.recognition.detector.detector import MobDetector, load_detector_config
from pybot.recognition.fixtures import MOB_FIXTURE_SUITES, MobFixtureImage, MobFixtureSuite


def _manifest_entries(fixtures_dir: Path, mob_name: str) -> list[MobFixtureImage]:
    mob_name = mob_name.lower()
    for suite in MOB_FIXTURE_SUITES:
        if suite.mob_name == mob_name:
            return suite.images()
    manifest_path = fixtures_dir / "manifest.json"
    if not manifest_path.is_file():
        return []
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    pattern = re.compile(str(data["filenamePattern"]), re.IGNORECASE)
    image_dir = fixtures_dir / str(data.get("folder", data.get("imageDir", ".")))
    entries: list[MobFixtureImage] = []
    for path in sorted(image_dir.glob("*.png")):
        match = pattern.match(path.name)
        if match is None:
            continue
        entries.append(
            MobFixtureImage(
                file_name=path.name,
                path=path,
                expected_count=int(match.group(1)),
                gray_world="_Gray" in path.stem,
            )
        )
    return entries


def run_fixtures(mob_name: str, fixtures_dir: Path, *, debug: bool, rebuild_descriptor: bool) -> dict:
    if rebuild_descriptor:
        DescriptorBuilder(PROJECT_ROOT).build(mob_name, force=True)
    config = load_detector_config()
    detector = MobDetector(PROJECT_ROOT, config)
    debug_root = PROJECT_ROOT / config["debugOutputDir"]
    summary = {
        "mobName": mob_name,
        "pipeline": "heatmap",
        "images": [],
        "totals": {"expected": 0, "accepted": 0, "matches": 0, "misses": 0, "extras": 0},
    }
    overlay_paths: list[Path] = []
    for image in _manifest_entries(fixtures_dir, mob_name):
        frame = cv2.imread(str(image.path), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        result = detector.detect(frame, mob_name)
        accepted = len(result.accepted)
        expected = image.expected_count
        matches = min(expected, accepted)
        misses = max(0, expected - accepted)
        extras = max(0, accepted - expected)
        summary["totals"]["expected"] += expected
        summary["totals"]["accepted"] += accepted
        summary["totals"]["matches"] += matches
        summary["totals"]["misses"] += misses
        summary["totals"]["extras"] += extras
        best_scores = [
            round(candidate.final_score, 4)
            for candidate in sorted(result.candidates, key=lambda item: item.final_score, reverse=True)[:5]
        ]
        row = {
            "file": image.file_name,
            "grayWorld": image.gray_world,
            "expected": expected,
            "accepted": accepted,
            "matches": matches,
            "misses": misses,
            "extras": extras,
            "bestScores": best_scores,
            "elapsedS": round(result.elapsed_s, 4),
        }
        summary["images"].append(row)
        print(
            f"{image.file_name}: expected={expected} accepted={accepted} "
            f"misses={misses} extras={extras} best={best_scores} time={row['elapsedS']}s"
        )
        if debug:
            out_dir = save_debug_bundle(debug_root, image.file_name, frame, result)
            overlay_paths.append(out_dir / "detected_overlay.png")
    if debug:
        summary_dir = debug_root / mob_name
        summary_dir.mkdir(parents=True, exist_ok=True)
        (summary_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        save_summary_contact_sheet(summary_dir / "summary_contact_sheet.png", overlay_paths)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run mob detector fixture suite")
    parser.add_argument("--mob", required=True)
    parser.add_argument("--fixtures", default=str(RECOGNITION_DIR / "test-fixtures"))
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--rebuild-descriptor", action="store_true")
    args = parser.parse_args(argv)
    run_fixtures(args.mob.lower(), Path(args.fixtures), debug=args.debug, rebuild_descriptor=args.rebuild_descriptor)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
