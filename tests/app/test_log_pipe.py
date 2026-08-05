from __future__ import annotations

import unittest

from pybot.app.log_pipe import LogPipe
from pybot.app.session_log import AppSessionLog


class _Root:
    def after(self, *_args) -> None:
        return None


class LogPipePersistTests(unittest.TestCase):
    def test_persist_callback_receives_every_log_line(self) -> None:
        pipe = LogPipe(_Root())
        seen: list[str] = []
        pipe.set_persist_callback(seen.append)
        pipe.log("first")
        pipe.log("second")
        pipe._drain()
        self.assertEqual(seen, ["first", "second"])

    def test_persist_callback_not_called_when_unset(self) -> None:
        pipe = LogPipe(_Root())
        pipe.log("no sink")
        pipe._drain()  # must not raise

    def test_persist_callback_can_be_cleared(self) -> None:
        pipe = LogPipe(_Root())
        seen: list[str] = []
        pipe.set_persist_callback(seen.append)
        pipe.log("before")
        pipe._drain()
        pipe.set_persist_callback(None)
        pipe.log("after")
        pipe._drain()
        self.assertEqual(seen, ["before"])


class AppSessionLogGuardTests(unittest.TestCase):
    def test_write_system_is_noop_before_open(self) -> None:
        session = AppSessionLog()
        # Must not raise and must not create files before open().
        session.write_system("INFO", "ui", "dropped")
        self.assertIsNone(session._system_log)

    def test_write_system_is_noop_after_close(self) -> None:
        # Drive the open/closed flags directly to avoid creating a real
        # session directory under logs/sessions during tests.
        session = AppSessionLog()
        session._opened = True
        session._closed = True
        session.write_system("INFO", "ui", "dropped after close")
        self.assertTrue(session._queue.empty())
