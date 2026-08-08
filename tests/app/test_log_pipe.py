from __future__ import annotations

import unittest

from pybot.app.log_pipe import LogPipe
from pybot.app.session_log import AppSessionLog


class _Root:
    def after(self, *_args) -> None:
        return None


class LogPipePersistTests(unittest.TestCase):
    def test_persist_callback_receives_logs_without_tk_drain(self) -> None:
        pipe = LogPipe(_Root())
        seen: list[str] = []
        pipe.set_persist_callback(seen.append)
        pipe.log("first")
        pipe.log("second")
        self.assertEqual(seen, ["first", "second"])

    def test_log_does_not_enter_status_queue_or_need_visual_sinks(self) -> None:
        pipe = LogPipe(_Root())
        pipe.log("file only")
        self.assertTrue(pipe._queue.empty())

    def test_persist_callback_not_called_when_unset(self) -> None:
        pipe = LogPipe(_Root())
        pipe.log("no sink")
        pipe._drain()  # must not raise

    def test_persist_callback_can_be_cleared(self) -> None:
        pipe = LogPipe(_Root())
        seen: list[str] = []
        pipe.set_persist_callback(seen.append)
        pipe.log("before")
        pipe.set_persist_callback(None)
        pipe.log("after")
        self.assertEqual(seen, ["before"])

    def test_status_updates_still_use_tk_queue(self) -> None:
        class _Widget:
            def __init__(self) -> None:
                self.values: list[str] = []

            def configure(self, *, text: str) -> None:
                self.values.append(text)

        pipe = LogPipe(_Root())
        status = _Widget()
        hint = _Widget()
        pipe.set_status_widgets(status, hint)  # type: ignore[arg-type]
        pipe.status("Ready", "Launch the game")
        pipe._drain()
        self.assertEqual(status.values, ["Ready"])
        self.assertEqual(hint.values, ["Launch the game"])


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
