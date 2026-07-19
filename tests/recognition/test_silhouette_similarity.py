"""Silhouette similarity metric tests."""

from __future__ import annotations

import unittest

import numpy as np

from pybot.recognition.detector.descriptors.layout_utils import (
    candidate_silhouette,
    silhouette_similarity,
)


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

    def test_occupancy_mask_excludes_pixels_outside_component(self) -> None:
        region = np.zeros((8, 8, 3), dtype=np.uint8)
        region[2:6, 2:6] = (0, 200, 0)
        region[0, :] = (0, 200, 0)
        palette = np.asarray([(0, 200, 0)], dtype=np.float32)
        mask = np.zeros((8, 8), dtype=bool)
        mask[2:6, 2:6] = True

        unrestricted = candidate_silhouette(region, palette, 20.0, 4, 4)
        restricted = candidate_silhouette(
            region, palette, 20.0, 4, 4, occupancy_mask=mask,
        )

        self.assertGreater(float((restricted >= 0.5).sum()), 0.0)
        self.assertLess(float((restricted >= 0.5).sum()), float((unrestricted >= 0.5).sum()))

    def test_sparse_match_passes_while_outside_fill_fails_at_half(self) -> None:
        """Near-complete subset still clears the gate; viewport fill stays low."""
        reference = np.zeros((16, 16), dtype=np.float32)
        stable_mask = np.zeros((16, 16), dtype=bool)
        reference[3:12, 2:11] = 1.0
        stable_mask[3:12, 2:11] = True
        reference[6:9, 5:8] = 0.0
        stable_mask[6:9, 5:8] = False

        # Small trim only — massive structure misses must not clear 0.50.
        sparse_candidate = np.zeros((16, 16), dtype=np.float32)
        sparse_candidate[3:12, 2:11] = 1.0
        sparse_candidate[3, 2:11] = 0.0

        bloated_candidate = np.zeros((16, 16), dtype=np.float32)
        bloated_candidate[1:14, 1:14] = 1.0

        large_hole = reference.copy()
        large_hole[3:9, 2:11] = 0.0

        sparse_score = silhouette_similarity(sparse_candidate, reference, stable_mask)
        bloated_score = silhouette_similarity(bloated_candidate, reference, stable_mask)
        hole_score = silhouette_similarity(large_hole, reference, stable_mask)

        self.assertGreaterEqual(sparse_score, 0.50)
        self.assertLess(bloated_score, 0.50)
        self.assertLess(hole_score, 0.50)
        self.assertGreater(sparse_score, bloated_score)
        self.assertGreater(sparse_score, hole_score)


if __name__ == "__main__":
    unittest.main()
