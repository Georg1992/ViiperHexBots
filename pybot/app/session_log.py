"""Application session logging."""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path

from pybot.paths import PROJECT_ROOT

LOGS_DIR = PROJECT_ROOT / "logs" / "sessions"


class AppSessionLog:
    def __init__(self, session_id: str | None = None) -> None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.session_id = session_id or stamp
        self.session_dir = LOGS_DIR / self.session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.system_log = self.session_dir / "system.log"
        self.behavior_log = self.session_dir / "behavior.log"
        self._prune_old_sessions()
        self.write_system("INFO", "session", f"session start id={self.session_id}")

    def write_system(self, level: str, category: str, message: str) -> None:
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        line = f"[{stamp}] [{level}] [{category}] {message}\n"
        with self.system_log.open("a", encoding="utf-8") as handle:
            handle.write(line)

    def write_block(self, title: str, content: str) -> None:
        with self.system_log.open("a", encoding="utf-8") as handle:
            handle.write(f"\n--- {title} ---\n")
            for raw_line in content.splitlines():
                stripped = raw_line.strip()
                if stripped:
                    handle.write(f"  {stripped}\n")
            handle.write("---\n")

    def write_focus_change(self, reason: str, active_hwnd: int = 0) -> None:
        self.write_system("INFO", "focus", f"{reason} activeHwnd={active_hwnd}")

    def end(self, reason: str) -> None:
        self.write_system("INFO", "session", f"session end reason={reason}")

    @staticmethod
    def _prune_old_sessions(keep_count: int = 3) -> None:
        if not LOGS_DIR.is_dir():
            return
        sessions = sorted(
            (path for path in LOGS_DIR.iterdir() if path.is_dir()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for stale in sessions[keep_count:]:
            shutil.rmtree(stale, ignore_errors=True)
