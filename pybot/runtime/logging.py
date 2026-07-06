"""Session logging for the Python hunt runtime."""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from pybot.paths import PROJECT_ROOT

LOGS_DIR = PROJECT_ROOT / "logs" / "sessions"


class HuntLogger:
    def __init__(
        self,
        session_id: str | None = None,
        *,
        on_behavior: Callable[[str], None] | None = None,
    ) -> None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.session_id = session_id or stamp
        self.session_dir = LOGS_DIR / self.session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self._on_behavior = on_behavior

        self._behavior = logging.getLogger(f"pybot.behavior.{self.session_id}")
        self._system = logging.getLogger(f"pybot.system.{self.session_id}")
        if not self._behavior.handlers:
            self._configure_file_logger(self._behavior, self.session_dir / "behavior.log")
        if not self._system.handlers:
            self._configure_file_logger(self._system, self.session_dir / "system.log")

    @staticmethod
    def _configure_file_logger(logger: logging.Logger, path: Path) -> None:
        logger.setLevel(logging.DEBUG)
        handler = logging.FileHandler(path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S"))
        logger.addHandler(handler)
        logger.propagate = False

    def set_behavior_callback(self, callback: Callable[[str], None] | None) -> None:
        """Set the callback for behavior log lines (replaces direct _on_behavior access)."""
        self._on_behavior = callback

    def behavior(self, message: str) -> None:
        line = message if message.startswith("[") else f"[PYBOT] {message}"
        self._behavior.info(line)
        print(line, file=sys.stdout)
        if self._on_behavior is not None:
            self._on_behavior(line)

    def system(self, level: str, category: str, message: str) -> None:
        self._system.info("%s %s %s", level, category, message)
