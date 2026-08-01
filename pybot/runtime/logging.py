"""Session logging for the Python hunt runtime.

Logging is fully asynchronous: callers only enqueue a record onto an
in-memory queue, and a dedicated daemon thread (``QueueListener``) does
the actual file and stdout writes. This guarantees the hunt runtime's
control threads can never block on I/O — e.g. a stalled console
(Windows "QuickEdit" selection) or a full stdout pipe stops the writer
thread only, never the bot.
"""

from __future__ import annotations

import logging
import itertools
import logging.handlers
import queue
import sys
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from pybot.paths import SESSIONS_DIR

LOGS_DIR = SESSIONS_DIR
_LOG_QUEUE_MAXSIZE = 2000
_LOGGER_STOP_TIMEOUT_S = 0.5
_logger_sequence = itertools.count()


class _DroppingQueueHandler(logging.handlers.QueueHandler):
    """Non-blocking handler that drops oldest records under backpressure."""

    def enqueue(self, record: logging.LogRecord) -> None:
        try:
            self.queue.put_nowait(record)
        except queue.Full:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.queue.put_nowait(record)
            except queue.Full:
                pass


class HuntLogger:
    def __init__(
        self,
        session_id: str | None = None,
        *,
        on_behavior: Callable[[str], None] | None = None,
        echo_stdout: bool = True,
    ) -> None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_id = session_id or stamp
        self.session_dir = LOGS_DIR / self.session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self._on_behavior = on_behavior
        self._echo_stdout = echo_stdout
        self._listener: logging.handlers.QueueListener | None = None
        self._closed = False

        # A fresh logger name gives each HuntLogger exclusive ownership of
        # its listener and handlers, even when a session ID is reused.
        logger_name = f"pybot.behavior.{self.session_id}.{next(_logger_sequence)}"
        self._behavior = logging.getLogger(logger_name)
        if not self._behavior.handlers:
            self._configure_async_logger(self._behavior, self.session_dir / "behavior.log")

    def _configure_async_logger(self, logger: logging.Logger, path: Path) -> None:
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        log_queue: queue.Queue[logging.LogRecord] = queue.Queue(
            maxsize=_LOG_QUEUE_MAXSIZE
        )
        logger.addHandler(_DroppingQueueHandler(log_queue))

        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S")
        )
        handlers: list[logging.Handler] = [file_handler]
        if self._echo_stdout:
            stream_handler = logging.StreamHandler(sys.stdout)
            stream_handler.setFormatter(logging.Formatter("%(message)s"))
            handlers.append(stream_handler)

        self._listener = logging.handlers.QueueListener(
            log_queue, *handlers
        )
        self._listener.start()

    def set_behavior_callback(self, callback: Callable[[str], None] | None) -> None:
        """Set the callback for behavior log lines (replaces direct _on_behavior access)."""
        self._on_behavior = callback

    def close(self) -> bool:
        """Stop the writer within a bounded time and detach owned handlers.

        A filesystem or console handler can block a logging thread. Never let
        that handler make the hunt shutdown wait forever; leave ownership in
        place so a later runtime shutdown retry can finish the close.
        """
        listener = self._listener
        if listener is not None:
            listener.enqueue_sentinel()
            thread = getattr(listener, "_thread", None)
            if thread is not None:
                thread.join(timeout=_LOGGER_STOP_TIMEOUT_S)
                if thread.is_alive():
                    return False
            self._listener = None
        if self._closed:
            return True
        self._closed = True
        for handler in list(self._behavior.handlers):
            self._behavior.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass
        # Logger objects are kept globally by logging.Manager. Remove only
        # this instance's entry so repeated hunt sessions do not accumulate
        # logger objects, while never disturbing a replacement logger.
        manager = self._behavior.manager
        if manager.loggerDict.get(self._behavior.name) is self._behavior:
            manager.loggerDict.pop(self._behavior.name, None)
        return True

    def behavior(self, message: str) -> None:
        if self._closed:
            return
        line = message if message.startswith("[") else f"[PYBOT] {message}"
        # Non-blocking: QueueHandler just puts the record on an in-memory
        # queue; the listener thread performs the file/stdout writes.
        self._behavior.info(line)
        if self._on_behavior is not None:
            self._on_behavior(line)
