"""Runtime configuration loaded from config.ini."""

from __future__ import annotations

import configparser
from dataclasses import dataclass
from pathlib import Path

from pybot.paths import PROJECT_ROOT
from pybot.app.mob_catalog import load_mob_catalog, mob_folder_by_index
from pybot.runtime.constants import (
    CELL_SIZE_PX,
    DEFAULT_SEARCH_RANGE_CELLS,
    HUNT_DISCOVERY_INTERVAL_MS,
    HUNT_TELEPORT_DURATION_MS,
)
from pybot.runtime.input.scan_codes import key_name_to_scan_code

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.ini"


def resolve_mob_name(ini: configparser.ConfigParser, mob_name: str | None) -> str:
    if mob_name:
        return mob_name
    selected_monster = ini.getint("MonsterSettings", "SelectedMonster", fallback=1)
    catalog = load_mob_catalog()
    if not catalog:
        raise RuntimeError("No mob catalog found. Run build-mob-descriptor.ps1 first.")
    return mob_folder_by_index(catalog, selected_monster)


@dataclass(frozen=True)
class HuntRuntimeConfig:
    config_path: Path
    hwnd: int
    mob_name: str
    hunt_mode: str
    skill_delay_ms: int
    skill_button: str
    skill_scan_code: int
    teleport_button: str
    teleport_scan_code: int
    search_range_cells: int
    cell_size_px: int
    discovery_interval_ms: int
    teleport_duration_ms: int
    validation_enabled: bool
    control_file: Path | None
    skill_timer_button: str = ""
    skill_timer_scan_code: int = 0
    skill_timer_interval_ms: int = 0
    save_point_button: str = ""
    save_point_scan_code: int = 0
    sp_button: str = ""
    sp_scan_code: int = 0
    open_storage_button: str = ""
    open_storage_scan_code: int = 0
    use_sprite_grf: bool = False


def _read_ini(path: Path) -> configparser.ConfigParser:
    parser = configparser.ConfigParser()
    if path.is_file():
        parser.read(path, encoding="utf-8")
    return parser


def load_runtime_config(
    *,
    config_path: Path | None = None,
    hwnd: int = 0,
    mob_name: str | None = None,
    hunt_mode: str | None = None,
    validation_enabled: bool | None = None,
    control_file: Path | None = None,
    session_id: str | None = None,
) -> HuntRuntimeConfig:
    path = config_path or DEFAULT_CONFIG_PATH
    ini = _read_ini(path)

    skill_delay = ini.getint("Keybindings", "SkillDelay", fallback=500)
    skill_button = ini.get("Keybindings", "SkillButton", fallback="e")
    teleport_button = ini.get("Keybindings", "TeleportButton", fallback="q")
    save_point_button = ini.get("Keybindings", "SavePointButton", fallback="")
    sp_button = ini.get("Keybindings", "SPButton", fallback="")
    open_storage_button = ini.get("Keybindings", "OpenStorageButton", fallback="")
    skill_timer_button = ini.get("Keybindings", "SkillTimerButton", fallback="")
    skill_timer_interval = ini.getint("Keybindings", "SkillTimerInterval", fallback=0)

    search_range = ini.getint("Settings", "SearchRange", fallback=DEFAULT_SEARCH_RANGE_CELLS)
    val_enabled = ini.getint("Settings", "HuntValidationLog", fallback=1) == 1
    if validation_enabled is not None:
        val_enabled = validation_enabled

    resolved_control = control_file
    if resolved_control is None and session_id:
        resolved_control = PROJECT_ROOT / "logs" / "sessions" / session_id / "control.json"

    return HuntRuntimeConfig(
        config_path=path,
        hwnd=hwnd,
        mob_name=resolve_mob_name(ini, mob_name),
        hunt_mode=hunt_mode or ini.get("Settings", "HuntMode", fallback="teleport"),
        skill_delay_ms=skill_delay,
        skill_button=skill_button,
        skill_scan_code=key_name_to_scan_code(skill_button),
        teleport_button=teleport_button,
        teleport_scan_code=key_name_to_scan_code(teleport_button),
        search_range_cells=search_range,
        cell_size_px=CELL_SIZE_PX,
        discovery_interval_ms=HUNT_DISCOVERY_INTERVAL_MS,
        teleport_duration_ms=HUNT_TELEPORT_DURATION_MS,
        validation_enabled=val_enabled,
        control_file=resolved_control,
        skill_timer_button=skill_timer_button,
        skill_timer_scan_code=key_name_to_scan_code(skill_timer_button),
        skill_timer_interval_ms=skill_timer_interval * 1000,
        save_point_button=save_point_button,
        save_point_scan_code=key_name_to_scan_code(save_point_button),
        sp_button=sp_button,
        sp_scan_code=key_name_to_scan_code(sp_button),
        open_storage_button=open_storage_button,
        open_storage_scan_code=key_name_to_scan_code(open_storage_button),
        use_sprite_grf=ini.getint("Settings", "UseSpriteGrf", fallback=0) == 1,
    )
