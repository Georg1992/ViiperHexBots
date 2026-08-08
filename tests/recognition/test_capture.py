"""Tests for resilient screen capture."""

from __future__ import annotations

import threading
import time
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

    def test_hung_grab_times_out_without_freezing_the_caller(self) -> None:
        """A native grab that never returns must fail the frame, not wedge."""
        release = threading.Event()
        fake_sct = MagicMock()

        def hung_grab(_monitor):
            release.wait(timeout=5.0)
            return np.zeros((4, 4, 4), dtype=np.uint8)

        fake_sct.grab.side_effect = hung_grab
        try:
            with patch("pybot.recognition.capture.mss.MSS", return_value=fake_sct):
                with patch("pybot.recognition.capture._CAPTURE_GRAB_TIMEOUT_S", 0.2):
                    capture_mod.reset_capture_session()
                    start = time.monotonic()
                    frame = capture_mod.capture_region(10, 20, 4, 4)
                    elapsed = time.monotonic() - start
        finally:
            # Always release the worker so later tests are not queued behind
            # a stuck grab.
            release.set()

        self.assertIsNone(frame)
        self.assertLess(elapsed, 2.0)
        # A timed-out runtime grab is retained for the worker that owns it and
        # closed after the native call finally returns.
        for _ in range(100):
            if fake_sct.close.called:
                break
            time.sleep(0.01)
        fake_sct.close.assert_called_once_with()

    def test_consecutive_timeouts_retain_each_retired_session(self) -> None:
        """Each rotated runtime worker must close its own native session."""
        releases = [threading.Event(), threading.Event()]
        sessions = [MagicMock(), MagicMock()]
        shot = np.zeros((4, 4, 4), dtype=np.uint8)

        def hung_grab(index: int):
            def grab(_monitor):
                releases[index].wait(timeout=5.0)
                return shot

            return grab

        for index, session in enumerate(sessions):
            session.grab.side_effect = hung_grab(index)

        try:
            with patch(
                "pybot.recognition.capture.mss.MSS",
                side_effect=sessions,
            ):
                with patch("pybot.recognition.capture._CAPTURE_GRAB_TIMEOUT_S", 0.1):
                    self.assertIsNone(capture_mod.capture_region(10, 20, 4, 4))
                    self.assertIsNone(capture_mod.capture_region(10, 20, 4, 4))
                    self.assertEqual(len(capture_mod._retired_sessions), 2)
        finally:
            for release in releases:
                release.set()

        for _ in range(100):
            if all(session.close.called for session in sessions):
                break
            time.sleep(0.01)
        for session in sessions:
            session.close.assert_called_once_with()
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


class UiCaptureChannelTests(unittest.TestCase):
    """The UI capture channel must be fully isolated from the runtime pipeline."""

    def setUp(self) -> None:
        self.channel = capture_mod._UiCaptureChannel()

    def tearDown(self) -> None:
        # Stop this test's worker via the None sentinel so daemon threads do
        # not accumulate across tests. The module singleton is never touched.
        try:
            self.channel._queue.put_nowait(None)
        except Exception:
            pass

    def test_ui_channel_uses_its_own_session(self) -> None:
        """A UI capture must not touch the runtime session/queue."""
        shot = np.zeros((4, 4, 4), dtype=np.uint8)
        fake_sct = MagicMock()
        fake_sct.grab.return_value = shot
        channel = self.channel

        with patch("pybot.recognition.capture.mss.MSS", return_value=fake_sct):
            with patch(
                "pybot.recognition.capture.cv2.cvtColor",
                side_effect=lambda frame, _: frame[:, :, :3],
            ):
                frame = channel.capture(10, 20, 4, 4)

        self.assertIsNotNone(frame)
        # The runtime pipeline's global session stays untouched.
        self.assertIsNone(capture_mod._sct)
        fake_sct.grab.assert_called_once()

    def test_ui_channel_hung_grab_times_out(self) -> None:
        """A wedged UI grab fails the frame without freezing the caller."""
        release = threading.Event()
        fake_sct = MagicMock()

        def hung_grab(_monitor):
            release.wait(timeout=5.0)
            return np.zeros((4, 4, 4), dtype=np.uint8)

        fake_sct.grab.side_effect = hung_grab
        channel = self.channel
        try:
            with patch("pybot.recognition.capture.mss.MSS", return_value=fake_sct):
                with patch("pybot.recognition.capture._CAPTURE_GRAB_TIMEOUT_S", 0.2):
                    start = time.monotonic()
                    frame = channel.capture(10, 20, 4, 4)
                    elapsed = time.monotonic() - start
        finally:
            release.set()

        self.assertIsNone(frame)
        self.assertLess(elapsed, 2.0)

    def test_ui_channel_retries_once_after_grab_failure(self) -> None:
        shot = np.zeros((4, 4, 4), dtype=np.uint8)
        fake_sct = MagicMock()
        fake_sct.grab.side_effect = [ScreenShotError("BitBlt", details={}), shot]
        channel = self.channel

        with patch("pybot.recognition.capture.mss.MSS", return_value=fake_sct):
            with patch(
                "pybot.recognition.capture.cv2.cvtColor",
                side_effect=lambda frame, _: frame[:, :, :3],
            ):
                frame = channel.capture(10, 20, 4, 4)

        self.assertIsNotNone(frame)
        self.assertEqual(fake_sct.grab.call_count, 2)

    def test_ui_channel_does_not_replace_an_unresolved_native_grab(self) -> None:
        """Retries fail closed until the abandoned native grab actually exits."""
        release_old = threading.Event()
        old_sct = MagicMock()
        new_sct = MagicMock()
        shot = np.zeros((4, 4, 4), dtype=np.uint8)

        def hung_grab(_monitor):
            release_old.wait(timeout=5.0)
            return shot

        old_sct.grab.side_effect = hung_grab
        new_sct.grab.return_value = shot
        channel = self.channel
        try:
            with patch(
                "pybot.recognition.capture.mss.MSS",
                side_effect=[old_sct, new_sct],
            ):
                with patch(
                    "pybot.recognition.capture._CAPTURE_GRAB_TIMEOUT_S",
                    0.1,
                ):
                    first = channel.capture(10, 20, 4, 4)
                    self.assertIsNone(first)
                    # The old native worker is still inside grab(). Do not
                    # create a second mss session while it is unresolved.
                    second = channel.capture(10, 20, 4, 4)
                    self.assertIsNone(second)
        finally:
            release_old.set()

        # Once the original native call returns, the worker clears the
        # retired-queue barrier and the next read may create one replacement.
        for _ in range(100):
            if not channel._retired_queues:
                break
            time.sleep(0.01)
        with patch(
            "pybot.recognition.capture.mss.MSS",
            return_value=new_sct,
        ):
            third = channel.capture(10, 20, 4, 4)

        self.assertIsNotNone(third)
        old_sct.grab.assert_called_once()
        new_sct.grab.assert_called_once()

    def test_ui_channel_wedged_session_creation_times_out_and_recovers(self) -> None:
        """A blocked mss.MSS() must never hang the caller; the next read recovers.

        Session creation used to run on the caller thread, so a wedged GDI
        desktop (observed after a seated danger teleport) could pin every
        status-panel read for its full multi-second timeout. Creation now
        runs on the worker: the caller always returns within the grab bound
        and the next read starts a fresh worker with a fresh session.
        """
        release = threading.Event()
        old_sct = MagicMock()
        new_sct = MagicMock()
        shot = np.zeros((4, 4, 4), dtype=np.uint8)
        mss_calls = {"count": 0}

        def wedged_factory():
            mss_calls["count"] += 1
            if mss_calls["count"] == 1:
                release.wait(timeout=5.0)
                return old_sct
            return new_sct

        new_sct.grab.return_value = shot
        channel = self.channel
        try:
            with patch(
                "pybot.recognition.capture.mss.MSS",
                side_effect=wedged_factory,
            ):
                with patch(
                    "pybot.recognition.capture._CAPTURE_GRAB_TIMEOUT_S",
                    0.2,
                ):
                    start = time.monotonic()
                    first = channel.capture(10, 20, 4, 4)
                    elapsed = time.monotonic() - start
                    self.assertIsNone(first)
                    self.assertLess(elapsed, 2.0)
                    second = channel.capture(10, 20, 4, 4)
                    self.assertIsNone(second)
        finally:
            release.set()

        for _ in range(100):
            if not channel._retired_queues:
                break
            time.sleep(0.01)
        with patch(
            "pybot.recognition.capture.mss.MSS",
            return_value=new_sct,
        ):
            third = channel.capture(10, 20, 4, 4)

        self.assertIsNotNone(third)
        new_sct.grab.assert_called_once()
        # Once the wedge resolves, the abandoned worker drains its retired
        # queue and closes its wedged session — no permanent native-handle
        # leak across recovery.
        for _ in range(100):
            if old_sct.close.called:
                break
            time.sleep(0.02)
        old_sct.close.assert_called_once()

    def test_ui_channel_is_distinct_from_runtime_worker(self) -> None:
        """The UI channel and runtime channel must not share a grab queue."""
        self.assertIsNot(
            capture_mod._ui_capture_channel._queue,
            capture_mod._grab_queue,
        )
        self.assertIsNot(
            capture_mod._ui_capture_channel._session_lock,
            capture_mod._capture_lock,
        )

    def test_discovery_and_tracking_observers_have_independent_channels(self) -> None:
        """A wedged observer channel cannot serialize the other observer."""
        discovery = capture_mod._UiCaptureChannel()
        tracking = capture_mod._UiCaptureChannel()
        release = threading.Event()
        old_sct = MagicMock()
        new_sct = MagicMock()
        shot = np.zeros((4, 4, 4), dtype=np.uint8)

        def hung_grab(_monitor):
            release.wait(timeout=5.0)
            return shot

        old_sct.grab.side_effect = hung_grab
        new_sct.grab.return_value = shot
        try:
            with patch(
                "pybot.recognition.capture.mss.MSS",
                side_effect=[old_sct, new_sct],
            ):
                with patch(
                    "pybot.recognition.capture._CAPTURE_GRAB_TIMEOUT_S",
                    0.1,
                ):
                    self.assertIsNone(discovery.capture(0, 0, 4, 4))
                    self.assertIsNotNone(tracking.capture(0, 0, 4, 4))
        finally:
            release.set()
            for channel in (discovery, tracking):
                try:
                    channel._queue.put_nowait(None)
                except Exception:
                    pass

        old_sct.grab.assert_called_once()
        new_sct.grab.assert_called_once()


if __name__ == "__main__":
    unittest.main()
