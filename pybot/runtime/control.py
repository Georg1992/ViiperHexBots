"""Runtime control file for pause/resume/stop from external processes or CLI."""

from __future__ import annotations

import json
import threading
from pathlib import Path


class RuntimeControl:
    def __init__(self, path: Path | None) -> None:
        self._path = path
        self._lock = threading.Lock()

    @property
    def path(self) -> Path | None:
        return self._path

    def write_command(self, command: str) -> None:
        if self._path is None:
            return
        payload = {"command": command}
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(self._path)

    def poll(self) -> str | None:
        if self._path is None or not self._path.is_file():
            return None
        with self._lock:
            try:
                payload = json.loads(self._path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return None
            finally:
                try:
                    self._path.unlink(missing_ok=True)
                except OSError:
                    pass
        command = str(payload.get("command", "")).lower()
        return command or None
