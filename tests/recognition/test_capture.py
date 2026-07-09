"""Tests for resilient screen capture."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from mss.exception import ScreenShotError

from pybot.recognition import capture as capture_mod


class CaptureRegionTests(unittest.TestCase):
    def tearDown(self) -> None:
        capture_mod.reset_capture_session()

    def test_returns_none_after_repeated_grab_failure(self) -> None:
        fake_sct = MagicMock()
        fake_sct.grab.side_effect = ScreenShotError("BitBlt", details={})

        with patch("pybot.recognition.capture.mss.MSS", return_value=fake_sct):
            capture_mod.reset_capture_session()
            frame = capture_mod.capture_region(10, 20, 64, 64)

        self.assertIsNone(frame)
        self.assertEqual(fake_sct.grab.call_count, 2)
        fake_sct.close.assert_called()

    def test_retries_once_after_grab_failure(self) -> None:
        shot = np.zeros((4, 4, 4), dtype=np.uint8)
        fake_sct = MagicMock()
        fake_sct.grab.side_effect = [ScreenShotError("BitBlt", details={}), shot]

        with patch("pybot.recognition.capture.mss.MSS", return_value=fake_sct):
            with patch("pybot.recognition.capture.cv2.cvtColor", side_effect=lambda frame, _: frame[:, :, :3]):
                capture_mod.reset_capture_session()
                frame = capture_mod.capture_region(10, 20, 4, 4)

        self.assertIsNotNone(frame)
        self.assertEqual(fake_sct.grab.call_count, 2)


if __name__ == "__main__":
    unittest.main()
