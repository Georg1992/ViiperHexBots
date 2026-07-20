"""Known-track discovery marks heatmap peaks for alive+dead silhouette checks."""

from __future__ import annotations

import unittest

from pybot.recognition.detector.detector import MobDetector


class MarkKnownBlobsTests(unittest.TestCase):
    def test_marks_nearest_blob_within_radius(self) -> None:
        blobs = [
            (10, 10, 1.0, (0, 0, 8, 8)),
            (100, 100, 1.0, (90, 90, 20, 20)),
        ]
        known = [(7, 105, 98, 1.0)]
        marked = MobDetector._mark_known_blobs(blobs, known, dedup_radius=20)
        self.assertEqual(marked, {1: (7, 105, 98, 1.0)})

    def test_first_scan_has_no_marks(self) -> None:
        blobs = [(10, 10, 1.0, (0, 0, 8, 8))]
        marked = MobDetector._mark_known_blobs(blobs, [], dedup_radius=90)
        self.assertEqual(marked, {})

    def test_one_track_claims_one_blob(self) -> None:
        blobs = [
            (50, 50, 1.0, (40, 40, 20, 20)),
            (55, 52, 0.9, (45, 42, 20, 20)),
        ]
        known = [(3, 50, 50, 1.0)]
        marked = MobDetector._mark_known_blobs(blobs, known, dedup_radius=90)
        self.assertEqual(len(marked), 1)
        self.assertIn(0, marked)
        self.assertEqual(marked[0][0], 3)


if __name__ == "__main__":
    unittest.main()
