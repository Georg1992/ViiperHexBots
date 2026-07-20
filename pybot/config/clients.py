"""Client profile helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from pybot.paths import PROJECT_ROOT


@dataclass(frozen=True)
class MemoryAddresses:
    """Module-relative offsets from a client profile (0 = missing)."""

    char_name: int = 0
    current_sp: int = 0
    max_sp: int = 0
    current_weight: int = 0
    max_weight: int = 0

    @property
    def has_any(self) -> bool:
        return any(
            (
                self.char_name,
                self.current_sp,
                self.max_sp,
                self.current_weight,
                self.max_weight,
            )
        )


@dataclass(frozen=True)
class ClientProfile:
    name: str
    process_name: str
    window_title_hint: str
    memory: MemoryAddresses
    raw: dict


def list_client_profiles(project_root: Path | None = None) -> list[str]:
    clients_dir = (project_root or PROJECT_ROOT) / "clients"
    if not clients_dir.is_dir():
        return ["Generic"]
    names = sorted(path.stem for path in clients_dir.glob("*.json"))
    return names or ["Generic"]


def memory_reading_enabled(profile_name: str) -> bool:
    """Memory reading is on for server profiles, off for Generic."""
    return profile_name.strip().lower() != "generic"


def _parse_hex_address(value: object) -> int:
    if value is None:
        return 0
    text = str(value).strip()
    if not text:
        return 0
    return int(text, 0)


def memory_addresses_from_dict(memory: dict | None) -> MemoryAddresses:
    if not isinstance(memory, dict):
        return MemoryAddresses()
    return MemoryAddresses(
        char_name=_parse_hex_address(memory.get("characterNameAddress")),
        current_sp=_parse_hex_address(memory.get("currentSpAddress")),
        max_sp=_parse_hex_address(memory.get("maxSpAddress")),
        current_weight=_parse_hex_address(memory.get("currentWeightAddress")),
        max_weight=_parse_hex_address(memory.get("totalWeightAddress")),
    )


def load_client_profile(
    profile_name: str,
    project_root: Path | None = None,
) -> ClientProfile | None:
    root = project_root or PROJECT_ROOT
    path = root / "clients" / f"{profile_name}.json"
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    memory = data.get("memory")
    return ClientProfile(
        name=str(data.get("name") or profile_name),
        process_name=str(data.get("processName") or ""),
        window_title_hint=str(data.get("windowTitleHint") or ""),
        memory=memory_addresses_from_dict(memory if isinstance(memory, dict) else None),
        raw=data,
    )


def client_supports_memory(profile_name: str, project_root: Path | None = None) -> bool:
    """True when the profile is not Generic and declares memory addresses."""
    if not memory_reading_enabled(profile_name):
        return False
    profile = load_client_profile(profile_name, project_root)
    if profile is None:
        return False
    memory = profile.raw.get("memory")
    if not isinstance(memory, dict):
        return False
    return bool(memory.get("currentLocationAddress")) or profile.memory.has_any
