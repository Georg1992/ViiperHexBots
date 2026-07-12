"""Tests for hunt ROI math — mirrors GetHuntSearchRegion."""

from __future__ import annotations

import unittest

import numpy as np

from pybot.runtime.capture.window_roi import (
    crop_frame_to_hunt_search_roi,
    hunt_roi_from_client_rect,
    hunt_roi_from_frame_shape,
    player_ignore_box,
    point_inside_ignore,
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

    def test_player_ignore_centered_two_by_two_cells(self) -> None:
        roi = hunt_roi_from_client_rect(
            0,
            0,
            1600,
            900,
            search_range_cells=16,
            cell_size_px=64,
        )
        assert roi is not None
        ignore_x, ignore_y, ignore_w, ignore_h = player_ignore_box(roi, 64)
        self.assertEqual(ignore_w, 128)
        self.assertEqual(ignore_h, 128)
        self.assertEqual(ignore_x, roi.center_x - 64)
        self.assertEqual(ignore_y, roi.center_y - 64)
        self.assertTrue(point_inside_ignore(roi.center_x, roi.center_y, ignore_x, ignore_y, ignore_w, ignore_h))
        self.assertFalse(point_inside_ignore(roi.x, roi.y, ignore_x, ignore_y, ignore_w, ignore_h))


if __name__ == "__main__":
    unittest.main()
