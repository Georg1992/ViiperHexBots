"""Known-track death check: death silhouette must beat living similarity."""

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

    def test_death_candidate_scores_higher_against_death_refs(self) -> None:
        death_mask = self.descriptor.death_silhouette_masks[0]
        candidate = np.asarray(death_mask.avg_mask, dtype=np.float32).reshape(16, 16)
        death_sim, death_passed = self.detector._score_death_vs_living_extract(
            np.zeros((64, 64, 3), dtype=np.uint8),
            self.descriptor,
            (0, 0, 32, 32),
            comp_bbox=(0, 0, 32, 32),
            death_masks=list(self.descriptor.death_silhouette_masks),
            living_candidate=candidate,
        )
        living_refs = self.detector._descriptor_silhouette_references(
            self.descriptor.silhouette_masks,
        )
        from pybot.recognition.detector.descriptors.layout_utils import (
            best_silhouette_match,
        )

        living_sim, *_rest = best_silhouette_match(candidate, living_refs)
        self.assertTrue(death_passed)
        self.assertGreater(death_sim, float(living_sim))

    def test_living_candidate_does_not_let_death_win(self) -> None:
        living_mask = self.descriptor.silhouette_masks[0]
        candidate = np.asarray(living_mask.avg_mask, dtype=np.float32).reshape(16, 16)
        death_sim, _death_passed = self.detector._score_death_vs_living_extract(
            np.zeros((64, 64, 3), dtype=np.uint8),
            self.descriptor,
            (0, 0, 32, 32),
            comp_bbox=(0, 0, 32, 32),
            death_masks=list(self.descriptor.death_silhouette_masks),
            living_candidate=candidate,
        )
        living_refs = self.detector._descriptor_silhouette_references(
            self.descriptor.silhouette_masks,
        )
        from pybot.recognition.detector.descriptors.layout_utils import (
            best_silhouette_match,
        )

        living_sim, *_rest = best_silhouette_match(candidate, living_refs)
        self.assertGreaterEqual(float(living_sim), death_sim)


if __name__ == "__main__":
    unittest.main()
