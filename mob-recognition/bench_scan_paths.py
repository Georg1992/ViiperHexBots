"""One-off benchmark for hunt scan audit. Run: py -3 mob-recognition/bench_scan_paths.py"""
from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
MOB_REC = Path(__file__).resolve().parent
SIMPLE = MOB_REC / "simple"
sys.path[:0] = [str(MOB_REC), str(SIMPLE)]

from detector import SimpleMobDetector, load_simple_config  # noqa: E402


def playfield_roi(frame):
    height, width = frame.shape[:2]
    return frame[
        int(height * 0.08) : int(height * 0.92),
        int(width * 0.03) : int(width * 0.97),
    ]


def bench(label: str, fn, runs: int = 3) -> None:
    times: list[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    print(
        f"{label}: min={min(times):.3f}s avg={statistics.mean(times):.3f}s max={max(times):.3f}s"
    )


def main() -> None:
    config = load_simple_config()
    detector = SimpleMobDetector(ROOT, config)
    frame = cv2.imread(str(MOB_REC / "test-fixtures" / "game-screenshots" / "333.png"))
    roi = playfield_roi(frame)
    watch3 = [(200, 180), (300, 220), (400, 260)]

    bench("discovery_scan", lambda: detector.detect(roi, "horn"))
    bench("watch_3_points", lambda: detector.detect(roi, "horn", watch_points=watch3, watch_only=True))
    bench("watch_6_points", lambda: detector.detect(roi, "horn", watch_points=watch3 * 2, watch_only=True))

    descriptor = detector.ensure_descriptor("horn")
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    start = time.perf_counter()
    for x, y in watch3:
        detector._evaluate_center(roi, hsv, descriptor, x, y, 1.0, watch_point=True)
    elapsed = time.perf_counter() - start
    print(f"watch_eval_3_points_no_heatmap: {elapsed:.3f}s")
    print("python_subprocess_overhead_estimate: ~0.4-0.8s import+capture+spawn per CLI call")


if __name__ == "__main__":
    main()
