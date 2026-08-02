"""Background status-panel capture/OCR tests (no Tk root required)."""

from __future__ import annotations

import unittest
from unittest.mock import patch

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


if __name__ == "__main__":
    unittest.main()
