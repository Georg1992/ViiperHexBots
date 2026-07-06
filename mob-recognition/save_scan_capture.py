"""Save an annotated hunt-ROI screenshot for discovery scan diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from capture import capture_region  # noqa: E402


def _parse_horns(value: str) -> list[dict]:
    if not value:
        return []
    data = json.loads(value)
    if not isinstance(data, list):
        raise ValueError("horns must be a JSON array")
    return data


def _draw_label(img, lines: list[str]) -> None:
    y = 22
    for line in lines:
        cv2.putText(
            img,
            line,
            (8, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            line,
            (8, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += 22


def _write_png(path: Path, frame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", frame)
    if not ok:
        raise SystemExit(f"failed to encode {path}")
    path.write_bytes(encoded.tobytes())


def _log_error(path: Path | None, message: str) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(message.rstrip() + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Save annotated discovery scan capture")
    parser.add_argument("--roi", required=True, help="x,y,width,height screen ROI")
    parser.add_argument("--out", required=True, type=Path, help="output PNG path")
    parser.add_argument("--horns", default="[]", help="JSON list of {x,y,confidence,scale}")
    parser.add_argument("--horns-file", type=Path, help="path to JSON file with horn list")
    parser.add_argument("--duration-ms", type=int, default=0)
    parser.add_argument("--scan-index", type=int, default=0)
    parser.add_argument("--raw-count", type=int, default=0)
    parser.add_argument("--living-count", type=int, default=0)
    parser.add_argument("--timestamp", default="")
    parser.add_argument("--error-log", type=Path, help="append capture errors to this file")
    args = parser.parse_args()

    try:
        parts = [int(p.strip()) for p in args.roi.split(",")]
        if len(parts) != 4:
            raise SystemExit("roi must be x,y,width,height")
        x, y, w, h = parts
        if args.horns_file is not None:
            horns = _parse_horns(args.horns_file.read_text(encoding="utf-8-sig"))
        else:
            horns = _parse_horns(args.horns)

        frame = capture_region(x, y, w, h)

        lines = [
            f"scan #{args.scan_index:02d}  horns={args.living_count}  raw={args.raw_count}",
            f"roi origin=({x},{y})  size={w}x{h}",
            f"scan time={args.duration_ms}ms",
        ]
        if args.timestamp:
            lines.append(f"captured={args.timestamp}")
        if horns:
            coord_bits = []
            for index, horn in enumerate(horns, start=1):
                coord_bits.append(
                    f"#{index} ({int(horn.get('x', 0))},{int(horn.get('y', 0))})"
                )
            lines.append("coords: " + "  ".join(coord_bits))
        _draw_label(frame, lines)

        _write_png(args.out, frame)
        return 0
    except Exception as exc:
        _log_error(args.error_log, str(exc))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
