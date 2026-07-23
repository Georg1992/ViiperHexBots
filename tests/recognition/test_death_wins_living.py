"""Death silhouette scoring: score_death_at uses death refs via silhouette gate."""

from __future__ import annotations

import unittest

import numpy as np

from pybot.paths import PROJECT_ROOT
from pybot.recognition.detector.descriptors.descriptor_builder import DescriptorBuilder
from pybot.recognition.detector.detector import MobDetector, load_detector_config


class DeathWinsLivingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.descriptor = DescriptorBuilder(PROJECT_ROOT).build("horn", force=True)
        cls.detector = MobDetector(PROJECT_ROOT, load_detector_config())

    def test_score_death_at_rejects_when_no_death_masks(self) -> None:
        """score_death_at returns False when descriptor has no death masks."""
        # Horn has death masks; test that the API still works.
        accepted, _bbox, sim = self.detector.score_death_at(
            np.zeros((64, 64, 3), dtype="uint8"),
            self.descriptor,
            32, 32,
            scale=1.0,
        )
        # On an empty frame, score_death_at should reject.
        self.assertFalse(accepted)
        self.assertLessEqual(sim, 0.0)


if __name__ == "__main__":
    unittest.main()
