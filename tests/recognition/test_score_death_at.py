"""Death silhouette scoring uses death masks, not living gate refs."""

from __future__ import annotations

import unittest

import numpy as np

from pybot.paths import PROJECT_ROOT
from pybot.recognition.detector.descriptors.descriptor_builder import DescriptorBuilder
from pybot.recognition.detector.detector import MobDetector, load_detector_config


class ScoreDeathAtTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.descriptor = DescriptorBuilder(PROJECT_ROOT).build("horn", force=True)
        cls.detector = MobDetector(PROJECT_ROOT, load_detector_config())

    def test_score_death_at_empty_frame_rejects(self) -> None:
        frame = np.zeros((128, 128, 3), dtype=np.uint8)
        accepted, _bbox, sim = self.detector.score_death_at(
            frame, self.descriptor, 64, 64, 1.0,
        )
        self.assertFalse(accepted)
        self.assertEqual(sim, 0.0)

    def test_score_death_at_without_masks_rejects(self) -> None:
        from dataclasses import replace

        frame = np.zeros((128, 128, 3), dtype=np.uint8)
        empty = replace(self.descriptor, death_silhouette_masks=[])
        accepted, _bbox, _sim = self.detector.score_death_at(
            frame, empty, 64, 64, 1.0,
        )
        self.assertFalse(accepted)


if __name__ == "__main__":
    unittest.main()
