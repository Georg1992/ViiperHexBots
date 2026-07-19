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

    def test_soft_gray_covers_ref_better_than_empty(self) -> None:
        reference = np.zeros((16, 16), dtype=np.float32)
        reference[4:12, 4:12] = 1.0
        stable_mask = reference >= 0.5

        empty = np.zeros((16, 16), dtype=np.float32)
        soft = np.zeros((16, 16), dtype=np.float32)
        soft[4:12, 4:12] = 0.35

        empty_score = silhouette_similarity(empty, reference, stable_mask)
        soft_score = silhouette_similarity(soft, reference, stable_mask)

        self.assertEqual(empty_score, 0.0)
        self.assertGreater(soft_score, 0.30)
        self.assertLess(soft_score, 0.40)
        self.assertGreater(soft_score, empty_score)

    def test_one_cell_shift_still_scores_high(self) -> None:
        reference = np.zeros((16, 16), dtype=np.float32)
        reference[5:11, 4:10] = 1.0
        stable_mask = reference >= 0.5
        shifted = np.zeros((16, 16), dtype=np.float32)
        shifted[5:11, 5:11] = 1.0  # +1 x

        score = silhouette_similarity(shifted, reference, stable_mask)
        self.assertGreaterEqual(score, 0.70)

    def test_sparse_match_beats_viewport_fill_and_thin_remnant(self) -> None:
        reference = np.zeros((16, 16), dtype=np.float32)
        stable_mask = np.zeros((16, 16), dtype=bool)
        reference[3:12, 2:11] = 1.0
        stable_mask[3:12, 2:11] = True
        reference[6:9, 5:8] = 0.0
        stable_mask[6:9, 5:8] = False

        sparse_candidate = np.zeros((16, 16), dtype=np.float32)
        sparse_candidate[3:12, 2:11] = 1.0
        sparse_candidate[3, 2:11] = 0.0

        bloated_candidate = np.zeros((16, 16), dtype=np.float32)
        bloated_candidate[1:14, 1:14] = 1.0

        large_hole = np.zeros((16, 16), dtype=np.float32)
        large_hole[10:12, 4:8] = 1.0

        sparse_score = silhouette_similarity(sparse_candidate, reference, stable_mask)
        bloated_score = silhouette_similarity(bloated_candidate, reference, stable_mask)
        hole_score = silhouette_similarity(large_hole, reference, stable_mask)

        self.assertGreaterEqual(sparse_score, 0.50)
        self.assertLess(bloated_score, 0.50)
        self.assertLess(hole_score, 0.35)
        self.assertGreater(sparse_score, bloated_score)
        self.assertGreater(sparse_score, hole_score)

    def test_scattered_extras_score_below_threshold(self) -> None:
        """Noise outside the ref must pull soft Jaccard under 0.5."""
        reference = np.zeros((16, 16), dtype=np.float32)
        reference[6:10, 6:10] = 1.0
        stable_mask = reference >= 0.5

        scattered = reference.copy()
        for y in range(0, 16, 2):
            for x in range(0, 16, 2):
                if reference[y, x] < 0.5:
                    scattered[y, x] = 1.0

        score = silhouette_similarity(scattered, reference, stable_mask)
        perfect = silhouette_similarity(reference, reference, stable_mask)
        self.assertLess(score, 0.50)
        self.assertLess(score, perfect)


if __name__ == "__main__":
    unittest.main()
