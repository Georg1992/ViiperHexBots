"""Benchmark local track follower. Run: py -3 mob-recognition/bench_local_track.py"""
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

from cli import apply_scale_calibration  # noqa: E402
from detector import SimpleMobDetector, load_simple_config  # noqa: E402
from tracking.local_tracker import track_local  # noqa: E402


def playfield_roi(frame):
    height, width = frame.shape[:2]
    return frame[
        int(height * 0.08) : int(height * 0.92),
        int(width * 0.03) : int(width * 0.97),
    ]


def bench(label: str, fn, runs: int = 5) -> None:
    times: list[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    print(
        f"{label}: min={min(times):.3f}s avg={statistics.mean(times):.3f}s max={max(times):.3f}s"
    )


def main() -> None:
    config = apply_scale_calibration(load_simple_config(), (0.82, 0.98), True)
    detector = SimpleMobDetector(ROOT, config)
    detector.apply_runtime_config(config)
    frame = cv2.imread(str(MOB_REC / "test-fixtures" / "game-screenshots" / "333.png"))
    roi = playfield_roi(frame)
    discovery = detector.detect(roi, "horn")
    living = [c for c in discovery.accepted if not c.is_dead]

    def tracks(count: int) -> list[dict]:
        return [
            {
                "trackId": index + 1,
                "x": candidate.center_x,
                "y": candidate.center_y,
                "scale": candidate.candidate_scale,
            }
            for index, candidate in enumerate(living[:count])
        ]

    bench("local_track_1", lambda: track_local(detector, roi, "horn", tracks(1)[0]))
    count = len(living)
    if count >= 2:
        repeat_three = tracks(2) + [tracks(2)[0]]
        bench("local_track_3", lambda: [track_local(detector, roi, "horn", t) for t in repeat_three])
        repeat_six = repeat_three * 2
        bench("local_track_6", lambda: [track_local(detector, roi, "horn", t) for t in repeat_six])


if __name__ == "__main__":
    main()
