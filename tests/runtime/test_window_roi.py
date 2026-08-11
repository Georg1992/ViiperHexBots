"""Tests for hunt ROI math — mirrors GetHuntSearchRegion."""

from __future__ import annotations

import unittest

import numpy as np

from pybot.runtime.capture.window_roi import (
    crop_frame_to_hunt_search_roi,
    hunt_roi_from_client_rect,
    search_box_size_px,
)


class WindowRoiTests(unittest.TestCase):
    def test_search_box_default_size(self) -> None:
        self.assertEqual(search_box_size_px(16, 64), 1024)

    def test_hunt_roi_centered_in_client(self) -> None:
        roi = hunt_roi_from_client_rect(
            100,
            50,
            1600,
            900,
            search_range_cells=16,
            cell_size_px=64,
        )
        assert roi is not None
        self.assertEqual(roi.w, 1024)
        self.assertEqual(roi.h, 1024)
        self.assertEqual(roi.x, 100 + (1600 // 2) - (1024 // 2))
        self.assertEqual(roi.y, 50 + 900 - 1024)

    def test_hunt_roi_clamps_when_client_smaller_than_search_box(self) -> None:
        roi = hunt_roi_from_client_rect(
            0,
            0,
            800,
            600,
            search_range_cells=16,
            cell_size_px=64,
        )
        assert roi is not None
        self.assertEqual(roi.x, 800 - 1024)
        self.assertEqual(roi.y, 600 - 1024)

    def test_crop_frame_to_hunt_search_roi_matches_client_math(self) -> None:
        frame_h, frame_w = 1079, 1919
        roi = hunt_roi_from_client_rect(
            0, 0, frame_w, frame_h,
            search_range_cells=16,
            cell_size_px=64,
        )
        assert roi is not None
        frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
        cropped = crop_frame_to_hunt_search_roi(frame, search_range_cells=16, cell_size_px=64)
        self.assertEqual(cropped.shape[0], 1024)
        self.assertEqual(cropped.shape[1], 1024)
        self.assertEqual(roi.x, (frame_w // 2) - 512)
        self.assertEqual(roi.y, (frame_h // 2) - 512)


if __name__ == "__main__":
    unittest.main()
