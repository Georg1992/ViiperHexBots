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

    def test_close_is_idempotent_and_rejects_new_work(self) -> None:
        work = UiWorkQueue(name="test-ui-work-closed")
        work.close()
        work.close()
        self.assertFalse(work.submit(lambda: None))


if __name__ == "__main__":
    unittest.main()
