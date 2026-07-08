"""One-shot detection test — captures a frame, runs detection, prints details, saves screenshots.

Usage:
    py scripts/capture_detect.py --mob horn
    py scripts/capture_detect.py --hwnd 123456 --mob horn
    py scripts/capture_detect.py --fixture path/to/screenshot.png
"""

from __future__ import annotations

import argparse
import time
from ctypes import wintypes
from pathlib import Path

import ctypes
import cv2
import numpy as np

from pybot.paths import PROJECT_ROOT
from pybot.recognition.capture import capture_region
from pybot.recognition.detector.detector import MobDetector, load_detector_config


def describe_frame(frame: np.ndarray, label: str) -> None:
    """Print frame stats."""
    h, w = frame.shape[:2]
    print(f"  {label}: {w}x{h}x{frame.shape[2]} dtype={frame.dtype} "
          f"mean={frame.mean():.1f} min={frame.min()} max={frame.max()}")


def save_diagnostic(frame: np.ndarray, path: Path, result) -> None:
    """Save annotated screenshot with detection marks."""
    annotated = frame.copy()
    for c in result.candidates:
        color = (0, 255, 0) if c.accepted else (0, 0, 255)
        cv2.drawMarker(annotated, (c.center_x, c.center_y), color,
                       cv2.MARKER_CROSS, 12, 2)
        if not c.accepted and c.rejection_reason:
            cv2.putText(annotated, c.rejection_reason,
                        (c.center_x + 10, c.center_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    cv2.imwrite(str(path), annotated)
    print(f"  Saved: {path.name}")


def run_detection(frame: np.ndarray, detector: MobDetector,
                  mob_name: str) -> None:
    """Run full detection pipeline and print detailed results."""
    t0 = time.perf_counter()
    result = detector.detect(frame, mob_name)
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"\nDetection: {len(result.candidates)} candidates, "
          f"{len(result.accepted)} accepted in {elapsed:.0f}ms")

    for i, c in enumerate(result.candidates):
        status = "ACC" if c.accepted else "REJ"
        reject = f" reason={c.rejection_reason}" if not c.accepted and c.rejection_reason else ""
        scores = (f"body={c.body_palette_score:.4f} "
                  f"accent={c.accent_score:.4f} "
                  f"size={c.size_score:.4f} "
                  f"heat={c.heatmap_score:.4f} "
                  f"final={c.final_score:.4f}")
        print(f"  [{status}] center=({c.center_x},{c.center_y}) "
              f"{scores}{reject}")


def main() -> int:
    parser = argparse.ArgumentParser(description="One-shot detection test")
    parser.add_argument("--hwnd", type=int, default=0)
    parser.add_argument("--mob", type=str, default="horn")
    parser.add_argument("--fixture", type=str, default="",
                        help="Screenshot path instead of live capture")
    parser.add_argument("--save", type=str, default="",
                        help="Save annotated screenshot to this path")
    args = parser.parse_args()

    config = load_detector_config()
    detector = MobDetector(PROJECT_ROOT, config)
    mob_name = args.mob.lower()

    try:
        detector.ensure_descriptor(mob_name)
        print(f"Descriptor loaded for '{mob_name}'")
    except FileNotFoundError:
        print(f"ERROR: Descriptor for '{mob_name}' not found.")
        return 1

    # ── Fixture mode ────────────────────────────────────────────────
    if args.fixture:
        path = Path(args.fixture)
        if not path.exists():
            print(f"ERROR: Fixture not found: {path}")
            return 1
        frame = cv2.imread(str(path))
        if frame is None:
            print(f"ERROR: Could not read: {path}")
            return 1
        describe_frame(frame, path.name)
        run_detection(frame, detector, mob_name)
        if args.save:
            result = detector.detect(frame, mob_name)
            save_diagnostic(frame, Path(args.save), result)
        return 0

    # ── Live capture mode ───────────────────────────────────────────
    user32 = ctypes.windll.user32

    if args.hwnd:
        hwnd = args.hwnd
    else:
        from pybot.app.win32_util import enum_game_windows
        windows = enum_game_windows()
        if not windows:
            print("ERROR: No game windows found.")
            return 1
        print("Found game windows:")
        for idx, entry in enumerate(windows, 1):
            print(f"  [{idx}] {entry.title} (hwnd={entry.hwnd})")
        choice = input("Select window number (or 0 for first): ").strip()
        hwnd = windows[0].hwnd
        if choice and choice != "0":
            try:
                hwnd = windows[int(choice) - 1].hwnd
            except (IndexError, ValueError):
                pass

    print(f"\nTarget: hwnd={hwnd} mob={mob_name}")

    # Build ROI
    from pybot.runtime.capture.window_roi import hunt_roi_from_client_rect
    rect = wintypes.RECT()
    if not user32.GetClientRect(hwnd, ctypes.byref(rect)):
        print("ERROR: GetClientRect failed")
        return 1
    origin = wintypes.POINT(0, 0)
    if not user32.ClientToScreen(hwnd, ctypes.byref(origin)):
        print("ERROR: ClientToScreen failed")
        return 1
    cw = rect.right - rect.left
    ch = rect.bottom - rect.top
    roi = hunt_roi_from_client_rect(
        origin.x, origin.y, cw, ch,
        search_range_cells=16, cell_size_px=64,
    )
    if roi is None:
        print("ERROR: Could not build ROI")
        return 1
    print(f"ROI: {roi.x},{roi.y} {roi.w}x{roi.h}")

    # Capture
    frame = capture_region(roi.x, roi.y, roi.w, roi.h)
    describe_frame(frame, "Capture")

    # Detect
    run_detection(frame, detector, mob_name)

    # Save diagnostic
    if args.save:
        result = detector.detect(frame, mob_name)
        save_diagnostic(frame, Path(args.save), result)
    else:
        # Auto-save to temp
        save_dir = PROJECT_ROOT / "logs" / "detect_debug"
        save_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        save_path = save_dir / f"{mob_name}_{ts}.png"
        result = detector.detect(frame, mob_name)
        save_diagnostic(frame, save_path, result)

        # Also save raw unmodified frame for offline analysis
        raw_path = save_dir / f"{mob_name}_{ts}_raw.png"
        cv2.imwrite(str(raw_path), frame)
        print(f"  Saved raw: {raw_path.name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
