"""Tests for pure detector response transformations."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from pybot.recognition.detection_response import (
    apply_scale_calibration,
    build_detect_response,
    candidate_to_json,
    parse_request_scale_range,
)


class Candidate:
    accepted = True
    bbox = (10, 20, 30, 40)
    center_x = 25
    center_y = 35
    final_score = 0.87654

    def to_dict(self) -> dict:
        return {"mobName": "horn", "accepted": self.accepted}


class DetectionResponseTests(unittest.TestCase):
    def test_calibration_does_not_mutate_config(self) -> None:
        config = {"scales": [1.0], "other": "kept"}
        calibrated = apply_scale_calibration(config, (0.9, 0.7), True)

        self.assertEqual(config, {"scales": [1.0], "other": "kept"})
        self.assertEqual(calibrated["scales"], [0.9, 0.8, 0.7])
        self.assertEqual(calibrated["centerScales"], [0.9, 0.8, 0.7])
        self.assertTrue(calibrated["enforceObjectSizeGate"])

    def test_parse_request_scale_range_normalizes_order(self) -> None:
        self.assertEqual(parse_request_scale_range([0.98, 0.82]), (0.82, 0.98))
        self.assertIsNone(parse_request_scale_range(None))
        self.assertIsNone(parse_request_scale_range([0.9]))

    def test_candidate_coordinates_include_screen_offset(self) -> None:
        payload = candidate_to_json(Candidate(), (100, 200))

        self.assertEqual(payload["x"], 110)
        self.assertEqual(payload["y"], 220)
        self.assertEqual(payload["centerX"], 125)
        self.assertEqual(payload["centerY"], 235)
        self.assertEqual(payload["confidence"], 0.8765)
        self.assertTrue(payload["living"])

    def test_scan_response_filters_nonliving_candidates(self) -> None:
        living = Candidate()
        dead = Candidate()
        dead.accepted = False
        result = SimpleNamespace(
            candidates=[living, dead],
            accepted=[living, dead],
            elapsed_s=0.12345,
        )

        response = build_detect_response(result, (0, 0), pipeline="scan")

        self.assertEqual(response["candidateCount"], 2)
        self.assertEqual(response["acceptedCount"], 2)
        self.assertEqual(len(response["candidates"]), 1)
        self.assertTrue(response["candidates"][0]["living"])


if __name__ == "__main__":
    unittest.main()
