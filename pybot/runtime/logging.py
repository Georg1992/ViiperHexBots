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
import logging.handlers
import queue
import sys
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from pybot.paths import SESSIONS_DIR

LOGS_DIR = SESSIONS_DIR


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

        self._behavior = logging.getLogger(f"pybot.behavior.{self.session_id}")
        if not self._behavior.handlers:
            self._configure_async_logger(self._behavior, self.session_dir / "behavior.log")

    def _configure_async_logger(self, logger: logging.Logger, path: Path) -> None:
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        log_queue: queue.Queue[logging.LogRecord] = queue.Queue()
        logger.addHandler(logging.handlers.QueueHandler(log_queue))

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

    def behavior(self, message: str) -> None:
        line = message if message.startswith("[") else f"[PYBOT] {message}"
        # Non-blocking: QueueHandler just puts the record on an in-memory
        # queue; the listener thread performs the file/stdout writes.
        self._behavior.info(line)
        if self._on_behavior is not None:
            self._on_behavior(line)
