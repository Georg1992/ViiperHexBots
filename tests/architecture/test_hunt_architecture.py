"""Static checks that hunt behaviour modules stay detection-free.

Checks that behaviour modules (hunt_policy, hunt_mode, hunt_tracks)
don't import detection/capture modules directly — they must access
detector and capture only through the runtime context (ctx).
Worker modules and hunt_runtime.py, which orchestrate detection,
are verified to import the expected detection boundaries.
"""

from __future__ import annotations

import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYBOT_RUNTIME = PROJECT_ROOT / "pybot" / "runtime"

BEHAVIOUR_FILES = (
    PYBOT_RUNTIME / "hunt_policy.py",
    PYBOT_RUNTIME / "hunt_mode.py",
    PYBOT_RUNTIME / "hunt_tracks.py",
)

FORBIDDEN_IMPORTS = (
    "detector_session",
    "hunt_capture",
    "discovery_filter",
    "window_roi",
)


class HuntArchitectureTests(unittest.TestCase):
    def test_behaviour_modules_do_not_import_detection_directly(self) -> None:
        for path in BEHAVIOUR_FILES:
            text = path.read_text(encoding="utf-8", errors="replace")
            for symbol in FORBIDDEN_IMPORTS:
                self.assertNotIn(
                    symbol,
                    text,
                    f"{path.name} must not import {symbol} "
                    f"(use ctx.detector / ctx.capture instead)",
                )

    def test_worker_modules_import_detection_boundaries(self) -> None:
        discovery = (PYBOT_RUNTIME / "workers" / "discovery_worker.py").read_text(
            encoding="utf-8", errors="replace"
        )
        self.assertIn("discovery_filter", discovery)

        tracking = (PYBOT_RUNTIME / "workers" / "tracking_worker.py").read_text(
            encoding="utf-8", errors="replace"
        )
        self.assertIn("detector_session", tracking)

        attack = (PYBOT_RUNTIME / "workers" / "attack_loop.py").read_text(
            encoding="utf-8", errors="replace"
        )
        self.assertNotIn("discovery_filter", attack)
        self.assertNotIn("detector_session", attack)

    def test_hunt_runtime_orchestrates_detection(self) -> None:
        runtime = (PYBOT_RUNTIME / "hunt_runtime.py").read_text(
            encoding="utf-8", errors="replace"
        )
        self.assertIn("DetectorSession", runtime)
        self.assertIn("HuntWindowCapture", runtime)
        self.assertIn("TrackingWorker", runtime)
        self.assertIn("DiscoveryWorker", runtime)
        self.assertIn("AttackLoop", runtime)


if __name__ == "__main__":
    unittest.main()
