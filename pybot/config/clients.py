"""Client profile helpers."""

from __future__ import annotations

import json
from pathlib import Path

from pybot.paths import PROJECT_ROOT


def list_client_profiles(project_root: Path | None = None) -> list[str]:
    clients_dir = (project_root or PROJECT_ROOT) / "clients"
    if not clients_dir.is_dir():
        return ["Generic"]
    names = sorted(path.stem for path in clients_dir.glob("*.json"))
    return names or ["Generic"]


def client_supports_memory(profile_name: str, project_root: Path | None = None) -> bool:
    root = project_root or PROJECT_ROOT
    profile_path = root / "clients" / f"{profile_name}.json"
    if not profile_path.is_file():
        return False
    data = json.loads(profile_path.read_text(encoding="utf-8"))
    memory = data.get("memory")
    if not isinstance(memory, dict):
        return False
    return bool(memory.get("currentLocationAddress"))
