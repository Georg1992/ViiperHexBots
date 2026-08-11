"""Background status-panel capture/OCR tests (no Tk root required)."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from pybot.recognition.ui.status_panel import StatusPanelValues

from pybot.app.status_panel_worker import (
    read_status_panel_snapshot,
)


class StatusPanelWorkerTests(unittest.TestCase):
    @patch("pybot.app.status_panel_worker.is_window_minimized", return_value=True)
    @patch("pybot.app.status_panel_worker.window_exists", return_value=True)
    def test_minimized_window_returns_without_capture(self, _exists, _minimized) -> None:
        with patch("pybot.app.status_panel_worker.ui_capture_region") as capture:
            result = read_status_panel_snapshot(123, None, refresh_max=True)
        self.assertEqual(result.state, "inactive")
        capture.assert_not_called()

    @patch("pybot.app.status_panel_worker.is_window_minimized", return_value=False)
    @patch("pybot.app.status_panel_worker.window_exists", return_value=True)
    @patch("pybot.app.status_panel_worker.client_rect_screen", return_value=(10, 20, 300, 200))
    @patch("pybot.app.status_panel_worker.ui_capture_region")
    @patch("pybot.app.status_panel_worker.find_status_panel", return_value=(4, 5))
    @patch("pybot.app.status_panel_worker.read_status_panel")
    def test_ocr_proceeds_without_foreground_requirement(
        self,
        _full_reader,
        _find,
        capture,
        _client,
        _minimized,
        _exists,
    ) -> None:
        """OCR must not depend on the game being the foreground window."""
        _full_reader.return_value = StatusPanelValues(
            hp=70,
            hp_max=100,
            sp=20,
            sp_max=30,
            weight=5,
            weight_max=100,
            panel_origin=(4, 5),
        )
        capture.return_value = np.zeros((200, 300, 3), dtype=np.uint8)

        result = read_status_panel_snapshot(123, None, refresh_max=True)

        self.assertEqual(result.state, "values")
        capture.assert_called()
        self.assertEqual(result.values.hp, 70)

    @patch("pybot.app.status_panel_worker.is_window_minimized", return_value=False)
    @patch("pybot.app.status_panel_worker.window_exists", return_value=True)
    @patch("pybot.app.status_panel_worker.client_rect_screen", return_value=(10, 20, 100, 100))
    @patch("pybot.app.status_panel_worker.ui_capture_region", return_value=None)
    def test_capture_failure_is_a_panel_missing_result(
        self, capture, _client, _minimized, _exists
    ) -> None:
        result = read_status_panel_snapshot(123, None, refresh_max=True)
        self.assertEqual(result.state, "panel_missing")
        capture.assert_called_once_with(10, 20, 100, 100)

    @patch("pybot.app.status_panel_worker.is_window_minimized", side_effect=AssertionError("must be skipped"))
    @patch("pybot.app.status_panel_worker.window_exists", side_effect=AssertionError("must be skipped"))
    @patch("pybot.app.status_panel_worker.client_rect_screen", side_effect=AssertionError("must be skipped"))
    @patch("pybot.app.status_panel_worker.ui_capture_region", return_value=None)
    def test_cached_client_rect_skips_all_block_prone_geometry_queries(
        self, capture, client_rect, _minimized, _exists
    ) -> None:
        result = read_status_panel_snapshot(
            123,
            None,
            refresh_max=True,
            client_hint=(10, 20, 100, 100),
        )
        self.assertEqual(result.state, "panel_missing")
        client_rect.assert_not_called()
        capture.assert_called_once_with(10, 20, 100, 100)

    @patch("pybot.app.status_panel_worker.find_status_panel", side_effect=AssertionError("hot path must not search"))
    @patch("pybot.recognition.ui.status_panel.verify_status_panel_at", side_effect=AssertionError("hot path must not verify header"))
    @patch("pybot.app.status_panel_worker.read_status_panel_fixed_rois")
    @patch("pybot.app.status_panel_worker.ui_capture_region")
    def test_confirmed_read_captures_only_fixed_value_band(
        self, capture, fixed_reader, _verify, _find
    ) -> None:
        confirmed = StatusPanelValues(
            hp=80,
            hp_max=100,
            sp=10,
            sp_max=100,
            weight=20,
            weight_max=100,
            panel_origin=(4, 5),
        )
        fixed_reader.return_value = confirmed
        capture.return_value = np.zeros((85, 138, 3), dtype=np.uint8)

        result = read_status_panel_snapshot(
            123,
            confirmed,
            refresh_max=False,
            client_hint=(10, 20, 300, 200),
        )

        self.assertEqual(result.state, "values")
        self.assertEqual(result.values.panel_origin, (4, 5))
        capture.assert_called_once_with(56, 70, 138, 85)
        fixed_reader.assert_called_once()

    @patch("pybot.app.status_panel_worker.find_status_panel", side_effect=AssertionError("ROI miss must not search immediately"))
    @patch("pybot.app.status_panel_worker.ui_capture_region", side_effect=AssertionError("invalid origin must not capture"))
    def test_invalid_confirmed_origin_reports_roi_miss(
        self, _capture, _find
    ) -> None:
        confirmed = StatusPanelValues(
            hp=80,
            hp_max=100,
            sp=10,
            sp_max=100,
            weight=20,
            weight_max=100,
            panel_origin=(500, 500),
        )
        result = read_status_panel_snapshot(
            123,
            confirmed,
            refresh_max=False,
            client_hint=(10, 20, 300, 200),
        )
        self.assertEqual(result.state, "roi_missing")

    @patch("pybot.app.status_panel_worker.is_window_minimized", return_value=False)
    @patch("pybot.app.status_panel_worker.window_exists", return_value=True)
    @patch("pybot.app.status_panel_worker.client_rect_screen", return_value=(10, 20, 300, 200))
    @patch("pybot.app.status_panel_worker.ui_capture_region")
    @patch("pybot.app.status_panel_worker.find_status_panel", return_value=(4, 5))
    @patch("pybot.app.status_panel_worker.read_status_panel", return_value=None)
    @patch("pybot.app.status_panel_worker.read_status_panel_hp", return_value=(40, 100))
    def test_full_read_keeps_hp_when_sp_or_weight_ocr_fails(
        self,
        _hp_reader,
        _full_reader,
        _find,
        capture,
        _client,
        _minimized,
        _exists,
    ) -> None:
        capture.return_value = np.zeros((200, 300, 3), dtype=np.uint8)

        result = read_status_panel_snapshot(123, None, refresh_max=True)

        self.assertEqual(result.state, "hp_only")
        self.assertEqual(result.hp, (40, 100))
        self.assertIsNone(result.values)

    @patch("pybot.app.status_panel_worker.is_window_minimized", return_value=False)
    @patch("pybot.app.status_panel_worker.window_exists", return_value=True)
    @patch("pybot.app.status_panel_worker.client_rect_screen", return_value=(10, 20, 300, 200))
    @patch("pybot.app.status_panel_worker.ui_capture_region")
    @patch("pybot.app.status_panel_worker.find_status_panel", return_value=(4, 5))
    @patch("pybot.app.status_panel_worker.read_status_panel", return_value=None)
    @patch("pybot.app.status_panel_worker.read_status_panel_hp", return_value=None)
    @patch("pybot.app.status_panel_worker.read_status_panel_sp", return_value=(10, 10))
    def test_full_read_publishes_sp_when_hp_and_weight_ocr_fail(
        self,
        _sp_reader,
        _hp_reader,
        _full_reader,
        _find,
        capture,
        _client,
        _minimized,
        _exists,
    ) -> None:
        capture.return_value = np.zeros((200, 300, 3), dtype=np.uint8)

        result = read_status_panel_snapshot(123, None, refresh_max=True)

        self.assertEqual(result.state, "sp_only")
        self.assertEqual(result.sp, (10, 10))
        self.assertIsNone(result.hp)

    @patch("pybot.app.status_panel_worker.is_window_minimized", return_value=False)
    @patch("pybot.app.status_panel_worker.window_exists", return_value=True)
    @patch("pybot.app.status_panel_worker.client_rect_screen", return_value=(10, 20, 300, 200))
    @patch("pybot.app.status_panel_worker.ui_capture_region")
    @patch("pybot.recognition.ui.status_panel.verify_status_panel_at", return_value=True)
    @patch("pybot.app.status_panel_worker.read_status_panel", return_value=None)
    @patch("pybot.app.status_panel_worker.read_status_panel_hp", return_value=None)
    @patch("pybot.app.status_panel_worker.read_status_panel_sp", return_value=(10, 10))
    def test_fast_poll_publishes_sp_when_hp_and_weight_ocr_fail(
        self,
        _sp_reader,
        _hp_reader,
        _fast_reader,
        _verify,
        capture,
        _client,
        _minimized,
        _exists,
    ) -> None:
        capture.return_value = np.zeros((143, 219, 3), dtype=np.uint8)
        confirmed = StatusPanelValues(
            hp=80,
            hp_max=100,
            sp=10,
            sp_max=10,
            weight=20,
            weight_max=100,
            panel_origin=(4, 5),
        )

        result = read_status_panel_snapshot(123, confirmed, refresh_max=False)

        self.assertEqual(result.state, "sp_only")
        self.assertEqual(result.sp, (10, 10))
        self.assertIsNone(result.hp)

    @patch("pybot.app.status_panel_worker.is_window_minimized", return_value=False)
    @patch("pybot.app.status_panel_worker.window_exists", return_value=True)
    @patch("pybot.app.status_panel_worker.client_rect_screen", return_value=(10, 20, 300, 200))
    @patch("pybot.app.status_panel_worker.ui_capture_region")
    @patch("pybot.recognition.ui.status_panel.verify_status_panel_at", return_value=True)
    @patch("pybot.app.status_panel_worker.read_status_panel", return_value=None)
    @patch("pybot.app.status_panel_worker.read_status_panel_hp", return_value=(35, 100))
    @patch("pybot.app.status_panel_worker.read_status_panel_sp", return_value=None)
    def test_fast_poll_keeps_hp_when_sp_or_weight_ocr_fails(
        self,
        _sp_reader,
        _hp_reader,
        _fast_reader,
        _verify,
        capture,
        _client,
        _minimized,
        _exists,
    ) -> None:
        capture.return_value = np.zeros((143, 219, 3), dtype=np.uint8)
        confirmed = StatusPanelValues(
            hp=80,
            hp_max=100,
            sp=10,
            sp_max=10,
            weight=20,
            weight_max=100,
            panel_origin=(4, 5),
        )

        result = read_status_panel_snapshot(123, confirmed, refresh_max=False)

        self.assertEqual(result.state, "hp_only")
        self.assertEqual(result.hp, (35, 100))
        self.assertIsNone(result.sp)
        self.assertIsNone(result.values)


if __name__ == "__main__":
    unittest.main()
