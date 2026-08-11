"""Debug visualization status-color tests."""

from __future__ import annotations

import unittest

import numpy as np

from pybot.recognition.detector.detector import (
    DetectionCandidate,
    DetectionResult,
    SilhouetteCheck,
)
from scripts.debug_vis import draw_detection_overlay


class DebugVisualizationTests(unittest.TestCase):
    @staticmethod
    def _result(check: SilhouetteCheck, accepted: list[DetectionCandidate]) -> DetectionResult:
        return DetectionResult(
            mob_name="test",
            descriptor=None,  # draw_detection_overlay does not inspect the descriptor
            candidates=accepted,
            accepted=accepted,
            elapsed_s=0.0,
            timing={},
            sprite_heatmap=np.zeros((4, 4), dtype=np.float32),
            silhouette_checks=[check],
        )

    def test_refined_center_stays_green_when_candidate_is_accepted(self) -> None:
        check = SilhouetteCheck(
            center_x=20,
            center_y=20,
            heat_score=1.0,
            passed=True,
            similarity=1.0,
            candidate_id=7,
            extract_bbox=(5, 5, 10, 10),
            discovery_bbox=(4, 4, 12, 12),
        )
        accepted = DetectionCandidate(
            mob_name="test",
            center_x=23,
            center_y=22,
            bbox=(4, 4, 12, 12),
            final_score=1.0,
            heatmap_score=1.0,
            accepted=True,
            rejection_reason="",
            candidate_id=7,
        )

        overlay = draw_detection_overlay(
            np.zeros((40, 40, 3), dtype=np.uint8),
            self._result(check, [accepted]),
        )

        self.assertEqual(tuple(int(v) for v in overlay[5, 5]), (0, 220, 0))

    def test_passed_nonfinal_candidate_is_cyan(self) -> None:
        check = SilhouetteCheck(
            center_x=20,
            center_y=20,
            heat_score=1.0,
            passed=True,
            similarity=1.0,
            candidate_id=7,
            extract_bbox=(5, 5, 10, 10),
            discovery_bbox=(4, 4, 12, 12),
        )
        overlay = draw_detection_overlay(
            np.zeros((40, 40, 3), dtype=np.uint8),
            self._result(check, []),
        )
        self.assertEqual(tuple(int(v) for v in overlay[5, 5]), (255, 220, 0))

    def test_failed_pre_extraction_candidate_draws_discovery_box(self) -> None:
        check = SilhouetteCheck(
            center_x=20,
            center_y=20,
            heat_score=1.0,
            passed=False,
            similarity=0.0,
            candidate_id=7,
            discovery_bbox=(4, 4, 12, 12),
        )
        overlay = draw_detection_overlay(
            np.zeros((40, 40, 3), dtype=np.uint8),
            self._result(check, []),
        )
        self.assertEqual(tuple(int(v) for v in overlay[4, 4]), (0, 140, 255))

    def test_failed_candidate_is_orange(self) -> None:
        check = SilhouetteCheck(
            center_x=20,
            center_y=20,
            heat_score=1.0,
            passed=False,
            similarity=0.0,
            candidate_id=7,
            extract_bbox=(5, 5, 10, 10),
            discovery_bbox=(4, 4, 12, 12),
        )
        overlay = draw_detection_overlay(
            np.zeros((40, 40, 3), dtype=np.uint8),
            self._result(check, []),
        )
        self.assertEqual(tuple(int(v) for v in overlay[5, 5]), (0, 140, 255))


if __name__ == "__main__":
    unittest.main()
