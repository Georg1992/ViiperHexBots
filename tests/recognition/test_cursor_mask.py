"""Tests for simplified opacity death gating during local tracking."""

from __future__ import annotations

import unittest

from pybot.recognition.detector.detector import load_detector_config
from pybot.recognition.detector.tracking.local_tracker import _track_old_enough


class OpacityProbeGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_detector_config()

    def test_young_track_skips_death_probe(self) -> None:
        self.assertFalse(
            _track_old_enough(
                self.config,
                created_tick=1_000,
                now_tick=1_500,
            )
        )

    def test_mature_track_allows_death_probe(self) -> None:
        self.assertTrue(
            _track_old_enough(
                self.config,
                created_tick=1_000,
                now_tick=2_000,
            )
        )


if __name__ == "__main__":
    unittest.main()
