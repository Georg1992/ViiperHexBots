"""Background status-panel capture/OCR tests (no Tk root required)."""

from __future__ import annotations

import threading
import unittest
from unittest.mock import patch

import numpy as np

from pybot.recognition.ui.status_panel import StatusPanelValues

from pybot.app.status_panel_worker import (
    StatusPanelReadResult,
    read_status_panel_snapshot,
    read_status_panel_snapshot_bounded,
)


class StatusPanelWorkerTests(unittest.TestCase):
    def test_bounded_read_does_not_overlap_previous_native_read(self) -> None:
        """A timed-out helper blocks overlap while a fresh retry returns fast."""
        started = threading.Event()
        release = threading.Event()
        finished = threading.Event()
        calls = 0
        calls_lock = threading.Lock()

        def blocked_first_read(*_args, **_kwargs):
            nonlocal calls
            with calls_lock:
                calls += 1
                call_number = calls
            if call_number == 1:
                started.set()
                release.wait(timeout=1.0)
                finished.set()
            return StatusPanelReadResult(hwnd=123, state="inactive")

        with patch(
            "pybot.app.status_panel_worker.read_status_panel_snapshot",
            side_effect=blocked_first_read,
        ):
            result_holder: list[StatusPanelReadResult] = []

            def invoke() -> None:
                result_holder.append(
                    read_status_panel_snapshot_bounded(
                        123,
                        None,
                        refresh_max=True,
                        timeout_s=0.01,
                    )
                )

            try:
                caller = threading.Thread(target=invoke, daemon=True)
                caller.start()
                self.assertTrue(started.wait(timeout=1.0))
                caller.join(timeout=1.0)
                self.assertFalse(caller.is_alive())
                self.assertEqual(result_holder[0].state, "read_timeout")

                # The first helper is still abandoned. A retry must not spawn
                # another native OCR operation while it owns the single-flight
                # lock; it returns immediately instead.
                second = read_status_panel_snapshot_bounded(
                    123,
                    None,
                    refresh_max=True,
                    timeout_s=0.1,
                )
                self.assertEqual(second.state, "read_timeout")
                self.assertIn("still in flight", second.error or "")
                self.assertEqual(calls, 1)
            finally:
                release.set()
            self.assertTrue(finished.wait(timeout=1.0))

            # Once the abandoned helper has actually exited, a later read may
            # acquire the guard and execute normally again.
            third = None
            for _ in range(20):
                candidate = read_status_panel_snapshot_bounded(
                    123,
                    None,
                    refresh_max=True,
                    timeout_s=0.1,
                )
                if candidate.state == "inactive":
                    third = candidate
                    break
            self.assertIsNotNone(third)
            self.assertEqual(calls, 2)

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

    @patch("pybot.app.status_panel_worker.is_window_minimized", side_effect=AssertionError("must be skipped"))
    @patch("pybot.app.status_panel_worker.window_exists", side_effect=AssertionError("must be skipped"))
    @patch("pybot.app.status_panel_worker.client_rect_screen", side_effect=AssertionError("must be skipped"))
    @patch("pybot.app.status_panel_worker.ui_capture_region", return_value=None)
    def test_bounded_cached_read_keeps_working_without_win32_probes(
        self, capture, client_rect, _minimized, _exists
    ) -> None:
        result = read_status_panel_snapshot_bounded(
            123,
            None,
            refresh_max=True,
            timeout_s=0.2,
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
