"""Benchmark local track follower."""

from __future__ import annotations

import statistics
import time
from pathlib import Path

import cv2

from pybot.paths import PROJECT_ROOT, RECOGNITION_DIR
from pybot.recognition.cli import apply_scale_calibration
from pybot.recognition.simple.detector import SimpleMobDetector, load_simple_config
from pybot.recognition.simple.tracking.local_tracker import track_local

FIXTURE = RECOGNITION_DIR / "test-fixtures" / "game-screenshots" / "333.png"


def main() -> None:
    frame = cv2.imread(str(FIXTURE), cv2.IMREAD_COLOR)
    if frame is None:
        raise SystemExit(f"fixture missing: {FIXTURE}")

    config = load_simple_config()
    calibrated = apply_scale_calibration(frame, config)
    detector = SimpleMobDetector(PROJECT_ROOT, calibrated)
    detector.apply_runtime_config(calibrated)

    result = detector.detect(frame, "horn")
    living = [candidate for candidate in result.accepted if candidate.accepted]
    if not living:
        raise SystemExit("fixture has no living horns")

    track = {
        "trackId": 1,
        "x": living[0].center_x,
        "y": living[0].center_y,
        "scale": living[0].candidate_scale,
    }

    timings: list[float] = []
    for _ in range(20):
        start = time.perf_counter()
        track_local(detector, frame, "horn", track)
        timings.append((time.perf_counter() - start) * 1000)

    print(f"local track ms: mean={statistics.mean(timings):.1f} p95={sorted(timings)[18]:.1f}")


if __name__ == "__main__":
    main()
