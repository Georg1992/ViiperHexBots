"""Shared runtime-test fixtures: fixture-frame detector, fake capture, helpers."""

from __future__ import annotations

import numpy as np

from pybot.runtime.capture.window_roi import HuntRoi
from pybot.paths import PROJECT_ROOT
from pybot.runtime.config import HuntRuntimeConfig
from pybot.runtime.detection.detector_session import DetectorSession


def playfield_roi(frame: np.ndarray) -> np.ndarray:
    """Crop a fixture frame to the playfield region (8%–92% H, 3%–97% W)."""
    height, width = frame.shape[:2]
    return frame[
        int(height * 0.08) : int(height * 0.92),
        int(width * 0.03) : int(width * 0.97),
    ]


class FixtureDetector(DetectorSession):
    """DetectorSession that always operates on a fixed fixture frame."""

    def __init__(self, frame: np.ndarray) -> None:
        super().__init__("horn", project_root=PROJECT_ROOT)
        self._fixture_frame = frame

    def discover(self, roi: HuntRoi):
        return self.discover_frame(self._fixture_frame, roi)


class FakeCapture:
    """Capture stub returning a fixed ROI; always valid."""

    def __init__(self, roi: HuntRoi) -> None:
        self._roi = roi

    def is_valid(self) -> bool:
        return True

    def get_hunt_roi(self) -> HuntRoi:
        return self._roi


def make_config(**overrides) -> HuntRuntimeConfig:
    """Build a HuntRuntimeConfig for shadow/integration tests."""
    base = dict(
        config_path=PROJECT_ROOT / "config.ini",
        hwnd=12345,
        mob_name="horn",
        hunt_mode="teleport",
        skill_delay_ms=500,
        skill_button="e",
        skill_scan_code=18,
        teleport_button="q",
        teleport_scan_code=16,
        search_range_cells=16,
        cell_size_px=64,
        discovery_interval_ms=3000,
        teleport_duration_ms=500,
        validation_enabled=False,
        control_file=None,
    )
    base.update(overrides)
    return HuntRuntimeConfig(**base)
