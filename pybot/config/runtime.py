"""Hunt runtime configuration built from application settings."""

from __future__ import annotations

import configparser
from dataclasses import dataclass, replace
from pathlib import Path

from pybot.config.ini_store import load_settings
from pybot.config.schema import AppSettings, MobCustomSettings
from pybot.mobs.catalog import resolve_mob_descriptor_name
from pybot.paths import CONFIG_PATH, SESSIONS_DIR
from pybot.runtime.constants import (
    CELL_SIZE_PX,
    HUNT_DISCOVERY_INTERVAL_MS,
)
from pybot.runtime.input.scan_codes import key_name_to_scan_code


@dataclass(frozen=True)
class SkillTimerRuntime:
    button: str
    scan_code: int
    interval_ms: int


@dataclass(frozen=True)
class SelfBuffRuntime:
    button: str
    scan_code: int
    delay_ms: int


@dataclass(frozen=True)
class CustomBehaviorRuntime:
    configured: bool = False
    kiting_tick_ms: int = 0
    debuff_button: str = ""
    debuff_scan_code: int = 0
    heal_button: str = ""
    heal_scan_code: int = 0
    buffs: tuple[SelfBuffRuntime, ...] = ()


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
    creamy_tp_button: str = ""
    creamy_tp_scan_code: int = 0
    skill_timers: tuple[SkillTimerRuntime, ...] = ()
    custom_behavior: CustomBehaviorRuntime = CustomBehaviorRuntime()
    save_point_button: str = ""
    save_point_scan_code: int = 0
    hp_button: str = ""
    hp_scan_code: int = 0
    sp_button: str = ""
    sp_scan_code: int = 0
    # (button, scan_code, delay_ms) for each assigned Open Storage chain step.
    open_storage_steps: tuple[tuple[str, int, int], ...] = ()
    weight_modifier: int = 80
    take_fly_wings: bool = False
    fly_wings_amount: int = 100
    sit_on_low_sp: bool = False
    sit_on_low_sp_button: str = "insert"
    sit_on_low_sp_scan_code: int = 0
    use_sprite_grf: bool = False
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


def _open_storage_steps_from_settings(
    settings: AppSettings,
) -> tuple[tuple[str, int, int], ...]:
    steps: list[tuple[str, int, int]] = []
    for step in settings.open_storage_chain:
        button = step.button.strip()
        if not button:
            continue
        scan = key_name_to_scan_code(button)
        if scan <= 0:
            continue
        steps.append((button, scan, max(0, int(step.delay_ms))))
    return tuple(steps)


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

    resolved_mob_name = resolve_mob_descriptor_name(
        selected_monster=settings.selected_monster,
        mob_name=mob_name,
    )
    custom_settings = settings.mob_custom_settings.get(
        resolved_mob_name.strip().lower(), MobCustomSettings()
    )
    custom_buffs: list[SelfBuffRuntime] = []
    for button, delay_s in (
        (custom_settings.buff1_button, custom_settings.buff1_delay_s),
        (custom_settings.buff2_button, custom_settings.buff2_delay_s),
        (custom_settings.buff3_button, custom_settings.buff3_delay_s),
    ):
        button = button.strip()
        scan = key_name_to_scan_code(button)
        if button and scan > 0 and int(delay_s) > 0:
            custom_buffs.append(
                SelfBuffRuntime(
                    button=button,
                    scan_code=scan,
                    delay_ms=max(1, int(delay_s)) * 1000,
                )
            )
    search_range_cells = int(settings.search_range)
    if not 9 <= search_range_cells <= 16:
        raise ValueError(
            "Search range must be between 9 and 16 cells "
            f"(got {search_range_cells})."
        )

    custom_behavior = CustomBehaviorRuntime(
        configured=resolved_mob_name.strip().lower()
        in settings.mob_custom_settings,
        kiting_tick_ms=max(0, int(round(custom_settings.kiting_tick_s * 1000))),
        debuff_button=custom_settings.debuff_button.strip(),
        debuff_scan_code=key_name_to_scan_code(custom_settings.debuff_button),
        heal_button=custom_settings.heal_button.strip(),
        heal_scan_code=key_name_to_scan_code(custom_settings.heal_button),
        buffs=tuple(custom_buffs),
    )

    return HuntRuntimeConfig(
        config_path=settings.config_path,
        hwnd=hwnd,
        mob_name=resolved_mob_name,
        hunt_mode=settings.hunt_mode if hunt_mode is None else hunt_mode,
        skill_delay_ms=max(200, settings.skill_delay),
        skill_button=settings.skill_button,
        skill_scan_code=key_name_to_scan_code(settings.skill_button),
        teleport_button=settings.teleport_button,
        teleport_scan_code=key_name_to_scan_code(settings.teleport_button),
        creamy_tp_button=settings.creamy_tp_button,
        creamy_tp_scan_code=key_name_to_scan_code(settings.creamy_tp_button),
        search_range_cells=search_range_cells,
        cell_size_px=CELL_SIZE_PX,
        discovery_interval_ms=HUNT_DISCOVERY_INTERVAL_MS,
        teleport_duration_ms=settings.teleport_delay,
        validation_enabled=val_enabled,
        control_file=resolved_control,
        skill_timers=tuple(skill_timers),
        custom_behavior=custom_behavior,
        save_point_button=settings.save_point_button,
        save_point_scan_code=key_name_to_scan_code(settings.save_point_button),
        hp_button=settings.hp_button,
        hp_scan_code=key_name_to_scan_code(settings.hp_button),
        sp_button=settings.sp_button,
        sp_scan_code=key_name_to_scan_code(settings.sp_button),
        open_storage_steps=_open_storage_steps_from_settings(settings),
        weight_modifier=settings.weight_modifier,
        take_fly_wings=settings.take_fly_wings,
        fly_wings_amount=settings.fly_wings_amount,
        sit_on_low_sp=settings.sit_on_low_sp,
        sit_on_low_sp_button=settings.sit_on_low_sp_button,
        sit_on_low_sp_scan_code=key_name_to_scan_code(settings.sit_on_low_sp_button),
        use_sprite_grf=settings.use_sprite_grf,
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
