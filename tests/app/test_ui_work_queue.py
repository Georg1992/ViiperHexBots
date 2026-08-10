"""Tests for the UI background work queue lifecycle."""

from __future__ import annotations

import threading
import unittest

from pybot.app.ui_work_queue import UiWorkQueue


class UiWorkQueueTests(unittest.TestCase):
    def test_close_drains_running_and_pending_tasks(self) -> None:
        started = threading.Event()
        release = threading.Event()
        completed: list[str] = []
        done = threading.Event()

        work = UiWorkQueue(name="test-ui-work")

        def first() -> None:
            started.set()
            release.wait(timeout=2.0)
            completed.append("first")

        def second() -> None:
            completed.append("second")
            done.set()

        self.assertTrue(work.submit(first))
        self.assertTrue(started.wait(timeout=2.0))
        self.assertTrue(work.submit(second))

        # The sentinel cannot fit while the pending task occupies the slot;
        # close must still let both accepted tasks finish.
        work.close()
        release.set()

        self.assertTrue(done.wait(timeout=2.0))
        self.assertEqual(completed, ["first", "second"])
        self.assertFalse(work.submit(lambda: None))

    def test_task_exception_is_forwarded_to_owner(self) -> None:
        errors: list[Exception] = []
        done = threading.Event()
        work = UiWorkQueue(name="test-ui-work-error", on_error=lambda exc: (errors.append(exc), done.set()))
        try:
            self.assertTrue(work.submit(lambda: (_ for _ in ()).throw(ValueError("bad task"))))
            self.assertTrue(done.wait(timeout=2.0))
            self.assertEqual(len(errors), 1)
            self.assertIsInstance(errors[0], ValueError)
            self.assertEqual(str(errors[0]), "bad task")
        finally:
            work.close()

    def test_discard_pending_keeps_running_task_and_drops_queued_task(self) -> None:
        started = threading.Event()
        release = threading.Event()
        work = UiWorkQueue(name="test-ui-work-discard")
        try:
            self.assertTrue(work.submit(lambda: (started.set(), release.wait(timeout=2.0))))
            self.assertTrue(started.wait(timeout=2.0))
            self.assertTrue(work.submit(lambda: None))
            work.discard_pending()
            self.assertEqual(work._pending_count, 1)
        finally:
            release.set()
            work.close()

    def test_close_is_idempotent_and_rejects_new_work(self) -> None:
        work = UiWorkQueue(name="test-ui-work-closed")
        work.close()
        work.close()
        self.assertFalse(work.submit(lambda: None))


if __name__ == "__main__":
    unittest.main()
