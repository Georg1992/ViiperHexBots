"""Application session logging — lazy, non-blocking, background writer.

No files are touched until ``open()`` is called (when the bot starts).
All file I/O runs on a daemon queue-drainer thread so the UI thread
never blocks on disk writes.
"""

from __future__ import annotations

import queue
import shutil
import threading
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from pybot.paths import PROJECT_ROOT

LOGS_DIR = PROJECT_ROOT / "logs" / "sessions"


class AppSessionLog:
    """App-level session log with lazy open + background writer.

    Usage:
        session = AppSessionLog()           # no disk I/O
        session.open()                      # creates dir, starts writer
        session.write_block("bot start",
            "hwnd=12345")                   # queued, non-blocking
        session.write_focus_change("paused") # queued, non-blocking
        session.end("user exit")            # queued, signals writer
    """

    def __init__(self, session_id: str | None = None) -> None:
        self._session_id = session_id
        self._session_dir: Path | None = None
        self._system_log: Path | None = None
        self._opened = False
        self._closed = False
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._writer_thread: threading.Thread | None = None

    # ── Public API ──────────────────────────────────────────────────

    @property
    def session_id(self) -> str:
        """Return the session id, opening lazily if needed."""
        if self._session_id is None:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            self._session_id = stamp
        return self._session_id

    def open(self, session_id: str | None = None) -> None:
        """Create the session directory and start the background writer.

        If *session_id* is provided, use it as the directory name.
        Otherwise generate a fresh timestamp-based name.

        Safe to call multiple times — on subsequent calls the old
        writer is stopped and a fresh session directory is created.
        """
        if self._opened:
            self._close_writer()

        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self._session_id = session_id or stamp
        self._session_dir = LOGS_DIR / self._session_id
        self._session_dir.mkdir(parents=True, exist_ok=True)
        self._system_log = self._session_dir / "system.log"
        self._opened = True
        self._closed = False
        _prune_old_sessions()

        self._writer_thread = threading.Thread(
            target=self._writer_loop,
            name="log-writer",
            daemon=True,
        )
        self._writer_thread.start()

        # Write the session start line (queued, non-blocking)
        self._enqueue("INFO", "session", f"session start id={self._session_id}")

    def write_system(self, level: str, category: str, message: str) -> None:
        """Queue a system log line (non-blocking, returns immediately)."""
        if not self._opened:
            return
        self._enqueue(level, category, message)

    def write_block(self, title: str, content: str) -> None:
        """Queue a structured log block (non-blocking)."""
        if not self._opened or self._system_log is None:
            return
        lines = [f"\n--- {title} ---\n"]
        for raw_line in content.splitlines():
            stripped = raw_line.strip()
            if stripped:
                lines.append(f"  {stripped}\n")
        lines.append("---\n")
        self._queue.put_nowait("".join(lines))

    def write_focus_change(self, reason: str, active_hwnd: int = 0) -> None:
        """Queue a focus-change log line."""
        self.write_system("INFO", "focus", f"{reason} activeHwnd={active_hwnd}")

    def end(self, reason: str) -> None:
        """Queue the session end line and signal the writer to flush."""
        if not self._opened or self._closed:
            return
        self._closed = True
        self._enqueue("INFO", "session", f"session end reason={reason}")
        self._queue.put_nowait(None)  # sentinel: flush + stop
        self._close_writer()

    def _close_writer(self) -> None:
        """Flush pending writes and stop the background writer thread."""
        if self._writer_thread is not None and self._writer_thread.is_alive():
            # Sentinal already queued by end(); wait for it to drain
            self._writer_thread.join(timeout=2.0)
            self._writer_thread = None
        # Drain any leftover items so next open() starts clean
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    # ── Internal ────────────────────────────────────────────────────

    def _enqueue(self, level: str, category: str, message: str) -> None:
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        line = f"[{stamp}] [{level}] [{category}] {message}\n"
        self._queue.put_nowait(line)

    def _writer_loop(self) -> None:
        """Daemon thread: drain the queue and write to system.log."""
        system_log = self._system_log
        if system_log is None:
            return
        while True:
            item = self._queue.get()
            if item is None:  # sentinel — flush and exit
                break
            try:
                with system_log.open("a", encoding="utf-8") as handle:
                    handle.write(item)
            except OSError:
                pass  # best-effort; don't crash the writer


def _prune_old_sessions(keep_count: int = 3) -> None:
    """Remove all but the *keep_count* most recent session directories."""
    if not LOGS_DIR.is_dir():
        return
    sessions = sorted(
        (path for path in LOGS_DIR.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for stale in sessions[keep_count:]:
        shutil.rmtree(stale, ignore_errors=True)
