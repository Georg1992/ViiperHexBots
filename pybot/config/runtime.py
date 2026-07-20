"""Hunt runtime configuration built from application settings."""

from __future__ import annotations

import configparser
from dataclasses import dataclass, replace
from pathlib import Path

from pybot.config.ini_store import load_settings
from pybot.config.schema import AppSettings
from pybot.mobs.catalog import resolve_mob_descriptor_name
from pybot.paths import CONFIG_PATH, SESSIONS_DIR
from pybot.runtime.constants import (
    CELL_SIZE_PX,
    DEFAULT_SEARCH_RANGE_CELLS,
    HUNT_DISCOVERY_INTERVAL_MS,
)
from pybot.runtime.input.scan_codes import key_name_to_scan_code


@dataclass(frozen=True)
class SkillTimerRuntime:
    button: str
    scan_code: int
    interval_ms: int


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
    skill_timers: tuple[SkillTimerRuntime, ...] = ()
    save_point_button: str = ""
    save_point_scan_code: int = 0
    sp_button: str = ""
    sp_scan_code: int = 0
    open_storage_button: str = ""
    open_storage_scan_code: int = 0
    sit_on_low_sp: bool = False
    sit_on_low_sp_button: str = "insert"
    sit_on_low_sp_scan_code: int = 0
    client_profile: str = "Generic"


def resolve_mob_name(
    source: configparser.ConfigParser | AppSettings,
    mob_name: str | None = None,
) -> str:
    if isinstance(source, AppSettings):
        return resolve_mob_descriptor_name(
            selected_monster=source.selected_monster,
            mob_name=mob_name,
        )
    selected_monster = source.getint("MonsterSettings", "SelectedMonster", fallback=1)
    return resolve_mob_descriptor_name(
        selected_monster=selected_monster,
        mob_name=mob_name,
    )


def hunt_runtime_config_from_settings(
    settings: AppSettings,
    *,
    hwnd: int = 0,
    mob_name: str | None = None,
    hunt_mode: str | None = None,
    validation_enabled: bool | None = None,
    control_file: Path | None = None,
    session_id: str | None = None,
) -> HuntRuntimeConfig:
    val_enabled = settings.hunt_validation_log
    if validation_enabled is not None:
        val_enabled = validation_enabled

    resolved_control = control_file
    if resolved_control is None and session_id:
        resolved_control = SESSIONS_DIR / session_id / "control.json"

    skill_timers: list[SkillTimerRuntime] = []
    for timer in settings.skill_timers:
        button = timer.button.strip()
        scan = key_name_to_scan_code(button)
        interval_ms = max(1, int(timer.interval_s)) * 1000
        if button and scan:
            skill_timers.append(
                SkillTimerRuntime(
                    button=button,
                    scan_code=scan,
                    interval_ms=interval_ms,
                )
            )

    return HuntRuntimeConfig(
        config_path=settings.config_path,
        hwnd=hwnd,
        mob_name=resolve_mob_descriptor_name(
            selected_monster=settings.selected_monster,
            mob_name=mob_name,
        ),
        hunt_mode=hunt_mode or settings.hunt_mode,
        skill_delay_ms=settings.skill_delay,
        skill_button=settings.skill_button,
        skill_scan_code=key_name_to_scan_code(settings.skill_button),
        teleport_button=settings.teleport_button,
        teleport_scan_code=key_name_to_scan_code(settings.teleport_button),
        search_range_cells=settings.search_range or DEFAULT_SEARCH_RANGE_CELLS,
        cell_size_px=CELL_SIZE_PX,
        discovery_interval_ms=HUNT_DISCOVERY_INTERVAL_MS,
        teleport_duration_ms=settings.teleport_delay,
        validation_enabled=val_enabled,
        control_file=resolved_control,
        skill_timers=tuple(skill_timers),
        save_point_button=settings.save_point_button,
        save_point_scan_code=key_name_to_scan_code(settings.save_point_button),
        sp_button=settings.sp_button,
        sp_scan_code=key_name_to_scan_code(settings.sp_button),
        open_storage_button=settings.open_storage_button,
        open_storage_scan_code=key_name_to_scan_code(settings.open_storage_button),
        sit_on_low_sp=settings.sit_on_low_sp,
        sit_on_low_sp_button=settings.sit_on_low_sp_button,
        sit_on_low_sp_scan_code=key_name_to_scan_code(settings.sit_on_low_sp_button),
        client_profile=settings.client_profile,
    )


def load_runtime_config(
    *,
    config_path: Path | None = None,
    settings: AppSettings | None = None,
    hwnd: int = 0,
    mob_name: str | None = None,
    hunt_mode: str | None = None,
    validation_enabled: bool | None = None,
    control_file: Path | None = None,
    session_id: str | None = None,
) -> HuntRuntimeConfig:
    resolved_settings = settings or load_settings(config_path or CONFIG_PATH)
    if config_path is not None:
        resolved_settings = replace(resolved_settings, config_path=config_path)
    return hunt_runtime_config_from_settings(
        resolved_settings,
        hwnd=hwnd,
        mob_name=mob_name,
        hunt_mode=hunt_mode,
        validation_enabled=validation_enabled,
        control_file=control_file,
        session_id=session_id,
    )
