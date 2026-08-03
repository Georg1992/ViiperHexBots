"""Background status-panel capture/OCR tests (no Tk root required)."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from pybot.recognition.ui.status_panel import StatusPanelValues

from pybot.app.status_panel_worker import read_status_panel_snapshot


class StatusPanelWorkerTests(unittest.TestCase):
    @patch("pybot.app.status_panel_worker.is_window_active", return_value=False)
    @patch("pybot.app.status_panel_worker.window_exists", return_value=True)
    def test_inactive_window_returns_without_capture(self, _exists, _active) -> None:
        with patch("pybot.app.status_panel_worker.capture_region") as capture:
            result = read_status_panel_snapshot(123, None, refresh_max=True)
        self.assertEqual(result.state, "inactive")
        capture.assert_not_called()

    @patch("pybot.app.status_panel_worker.is_window_active", return_value=True)
    @patch("pybot.app.status_panel_worker.window_exists", return_value=True)
    @patch("pybot.app.status_panel_worker.client_rect_screen", return_value=(10, 20, 100, 100))
    @patch("pybot.app.status_panel_worker.capture_region", return_value=None)
    def test_capture_failure_is_a_panel_missing_result(
        self, capture, _client, _active, _exists
    ) -> None:
        result = read_status_panel_snapshot(123, None, refresh_max=True)
        self.assertEqual(result.state, "panel_missing")
        capture.assert_called_once_with(10, 20, 100, 100)

    @patch("pybot.app.status_panel_worker.is_window_active", return_value=True)
    @patch("pybot.app.status_panel_worker.window_exists", return_value=True)
    @patch("pybot.app.status_panel_worker.client_rect_screen", return_value=(10, 20, 300, 200))
    @patch("pybot.app.status_panel_worker.capture_region")
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
        _active,
        _exists,
    ) -> None:
        capture.return_value = np.zeros((200, 300, 3), dtype=np.uint8)

        result = read_status_panel_snapshot(123, None, refresh_max=True)

        self.assertEqual(result.state, "hp_only")
        self.assertEqual(result.hp, (40, 100))
        self.assertIsNone(result.values)

    @patch("pybot.app.status_panel_worker.is_window_active", return_value=True)
    @patch("pybot.app.status_panel_worker.window_exists", return_value=True)
    @patch("pybot.app.status_panel_worker.client_rect_screen", return_value=(10, 20, 300, 200))
    @patch("pybot.app.status_panel_worker.capture_region")
    @patch("pybot.app.status_panel_worker.verify_status_panel_at", return_value=True)
    @patch("pybot.app.status_panel_worker.read_status_panel", return_value=None)
    @patch("pybot.app.status_panel_worker.read_status_panel_hp", return_value=(35, 100))
    def test_fast_poll_keeps_hp_when_sp_or_weight_ocr_fails(
        self,
        _hp_reader,
        _fast_reader,
        _verify,
        capture,
        _client,
        _active,
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
        self.assertIsNone(result.values)


if __name__ == "__main__":
    unittest.main()
