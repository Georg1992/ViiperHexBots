"""Silhouette similarity metric tests."""

from __future__ import annotations

import unittest

import numpy as np

from pybot.recognition.detector.descriptors.layout_utils import silhouette_similarity


class SilhouetteSimilarityTests(unittest.TestCase):
    def test_viewport_fill_scores_low_against_tight_ref(self) -> None:
        reference = np.zeros((16, 16), dtype=np.float32)
        reference[4:12, 4:12] = 1.0
        stable_mask = np.zeros((16, 16), dtype=bool)
        stable_mask[4:12, 4:12] = True
        candidate = np.ones((16, 16), dtype=np.float32)

        score = silhouette_similarity(candidate, reference, stable_mask)

        self.assertLess(score, 0.35)

    def test_matching_shapes_score_high(self) -> None:
        reference = np.zeros((16, 16), dtype=np.float32)
        reference[5:11, 4:10] = 1.0
        stable_mask = reference >= 0.5
        candidate = reference.copy()

        score = silhouette_similarity(candidate, reference, stable_mask)

        self.assertGreaterEqual(score, 0.99)

    def test_extra_candidate_pixels_lower_score_than_subset_match(self) -> None:
        reference = np.zeros((16, 16), dtype=np.float32)
        reference[4:12, 4:12] = 1.0
        stable_mask = reference >= 0.5

        tight_candidate = reference.copy()
        bloated_candidate = np.zeros((16, 16), dtype=np.float32)
        bloated_candidate[2:14, 2:14] = 1.0

        tight_score = silhouette_similarity(tight_candidate, reference, stable_mask)
        bloated_score = silhouette_similarity(bloated_candidate, reference, stable_mask)

        self.assertGreater(tight_score, bloated_score)
        self.assertGreaterEqual(tight_score, 0.99)
        self.assertLess(bloated_score, 0.75)


if __name__ == "__main__":
    unittest.main()
