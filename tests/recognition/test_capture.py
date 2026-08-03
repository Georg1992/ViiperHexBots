"""Tests for resilient screen capture."""

from __future__ import annotations

import threading
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

    def test_reset_does_not_orphan_busy_capture_session(self) -> None:
        old_lock = capture_mod._capture_lock
        old_sct = MagicMock()
        old_lock.acquire()
        try:
            capture_mod._sct = old_sct
            capture_mod.reset_capture_session()

            # A busy session is retired, not discarded. It is closed when the
            # owner later releases/finishes its capture operation.
            self.assertIsNot(capture_mod._capture_lock, old_lock)
            self.assertIs(capture_mod._retired_sessions[id(old_lock)][1], old_sct)
            self.assertFalse(old_sct.close.called)
        finally:
            old_lock.release()
            capture_mod._close_retired_session(old_lock)

        old_sct.close.assert_called_once_with()
        self.assertNotIn(id(old_lock), capture_mod._retired_sessions)

    def test_waiting_capture_retries_after_lock_rotation(self) -> None:
        """A waiter must use the new session after reset retires the old one."""
        old_sct = MagicMock()
        new_sct = MagicMock()
        shot = np.zeros((4, 4, 4), dtype=np.uint8)
        old_grab_started = threading.Event()
        waiter_attempted_old_lock = threading.Event()
        release_old_grab = threading.Event()
        old_grab_timed_out = threading.Event()

        class SignalingLock:
            def __init__(self) -> None:
                self._lock = threading.Lock()
                self._calls = 0

            def acquire(self, timeout: float | None = None) -> bool:
                self._calls += 1
                if self._calls == 2:
                    waiter_attempted_old_lock.set()
                if timeout is None:
                    return self._lock.acquire()
                return self._lock.acquire(timeout=timeout)

            def release(self) -> None:
                self._lock.release()

            def locked(self) -> bool:
                return self._lock.locked()

        old_lock = SignalingLock()

        def old_grab(_monitor) -> np.ndarray:
            old_grab_started.set()
            if not release_old_grab.wait(timeout=2.0):
                old_grab_timed_out.set()
            return shot

        old_sct.grab.side_effect = old_grab
        new_sct.grab.return_value = shot
        capture_mod._capture_lock = old_lock
        capture_mod._sct = old_sct
        owner_result: list[np.ndarray | None] = []
        waiter_result: list[np.ndarray | None] = []
        waiter_started = threading.Event()

        def capture_owner() -> None:
            owner_result.append(capture_mod.capture_region(10, 20, 4, 4))

        def capture_waiter() -> None:
            waiter_started.set()
            waiter_result.append(capture_mod.capture_region(10, 20, 4, 4))

        with patch("pybot.recognition.capture.mss.MSS", return_value=new_sct):
            owner = threading.Thread(target=capture_owner)
            owner.start()
            self.assertTrue(old_grab_started.wait(timeout=2.0))

            waiter = threading.Thread(target=capture_waiter)
            waiter.start()
            self.assertTrue(waiter_started.wait(timeout=2.0))
            self.assertTrue(waiter_attempted_old_lock.wait(timeout=2.0))
            # The waiter is definitely blocked on the old lock while the real
            # owner is still inside grab(); reset must rotate that pair.
            # reset waits for the real in-flight owner, retires its session,
            # and rotates the pair used by future captures.
            capture_mod.reset_capture_session()
            release_old_grab.set()
            owner.join(timeout=2.0)
            waiter.join(timeout=2.0)

        self.assertFalse(owner.is_alive())
        self.assertFalse(waiter.is_alive())
        self.assertFalse(old_grab_timed_out.is_set())
        self.assertEqual(len(owner_result), 1)
        self.assertEqual(len(waiter_result), 1)
        self.assertIsNotNone(owner_result[0])
        self.assertIsNotNone(waiter_result[0])
        old_sct.grab.assert_called_once()
        new_sct.grab.assert_called_once()
        old_sct.close.assert_called_once_with()
        self.assertFalse(capture_mod._retired_sessions)

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
