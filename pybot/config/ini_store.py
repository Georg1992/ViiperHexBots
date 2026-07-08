"""Load and save config.ini with merge-on-write."""

from __future__ import annotations

import configparser
from pathlib import Path

from pybot.config.schema import AppSettings
from pybot.paths import CONFIG_PATH


def _read_ini(path: Path) -> configparser.ConfigParser:
    parser = configparser.ConfigParser()
    if path.is_file():
        parser.read(path, encoding="utf-8")
    return parser


def _ensure_section(parser: configparser.ConfigParser, name: str) -> None:
    if name not in parser:
        parser[name] = {}


def load_settings(path: Path | None = None) -> AppSettings:
    config_path = path or CONFIG_PATH
    parser = _read_ini(config_path)

    warper_x = parser.get("Warper", "X", fallback="")
    if warper_x == "ERROR":
        warper_x = ""
    warper_y = parser.get("Warper", "Y", fallback="")
    if warper_y == "ERROR":
        warper_y = ""

    return AppSettings(
        config_path=config_path,
        last_session_title=parser.get("LastSession", "GameTitle", fallback=""),
        last_session_process=parser.get("LastSession", "GameProcess", fallback=""),
        window_id=parser.getint("Window", "ID", fallback=0),
        window_title=parser.get("Window", "Title", fallback=""),
        window_process=parser.get("Window", "Process", fallback=""),
        client_profile=parser.get("Client", "Profile", fallback="Generic"),
        use_memory_reading=parser.getint("Client", "UseMemoryReading", fallback=0) == 1,
        selected_monster=parser.getint("MonsterSettings", "SelectedMonster", fallback=1),
        mob_recognition_debug=parser.getint("MobRecognition", "Debug", fallback=0),
        search_range=parser.getint("Settings", "SearchRange", fallback=16),
        hunt_mode=parser.get("Settings", "HuntMode", fallback="teleport"),
        time_on_location=parser.getint("Settings", "TimeOnLocation", fallback=20),
        weight_modifier=parser.getint("Settings", "WeightModifier", fallback=49),
        take_fly_wings=parser.getint("Settings", "TakeFlyWings", fallback=0) == 1,
        fly_wings_amount=parser.getint("Settings", "FlyWingsAmount", fallback=100),
        detect_captcha=parser.getint("Settings", "DetectCaptcha", fallback=0) == 1,
        hunt_log_overlay=parser.getint("Settings", "HuntLogOverlay", fallback=1) == 1,
        hunt_validation_log=parser.getint("Settings", "HuntValidationLog", fallback=1) == 1,
        use_sprite_grf=parser.getint("Settings", "UseSpriteGrf", fallback=0) == 1,
        warper_x=warper_x,
        warper_y=warper_y,
        warper_location=parser.getint("Warper", "warperLocation", fallback=0),
        skill_button=parser.get("Keybindings", "SkillButton", fallback="e"),
        skill_delay=parser.getint("Keybindings", "SkillDelay", fallback=500),
        teleport_button=parser.get("Keybindings", "TeleportButton", fallback="q"),
        save_point_button=parser.get("Keybindings", "SavePointButton", fallback=""),
        sp_button=parser.get("Keybindings", "SPButton", fallback=""),
        open_storage_button=parser.get("Keybindings", "OpenStorageButton", fallback=""),
        skill_timer_button=parser.get("Keybindings", "SkillTimerButton", fallback=""),
        skill_timer_interval=parser.getint("Keybindings", "SkillTimerInterval", fallback=20),
    )


def save_settings(settings: AppSettings) -> None:
    path = settings.config_path
    parser = _read_ini(path)

    _ensure_section(parser, "LastSession")
    parser["LastSession"]["GameProcess"] = settings.last_session_process
    parser["LastSession"]["GameTitle"] = settings.last_session_title

    _ensure_section(parser, "Window")
    parser["Window"]["ID"] = str(settings.window_id)
    parser["Window"]["Title"] = settings.window_title
    parser["Window"]["Process"] = settings.window_process

    _ensure_section(parser, "Client")
    parser["Client"]["Profile"] = settings.client_profile
    parser["Client"]["UseMemoryReading"] = "1" if settings.use_memory_reading else "0"

    _ensure_section(parser, "MonsterSettings")
    parser["MonsterSettings"]["SelectedMonster"] = str(settings.selected_monster)

    _ensure_section(parser, "MobRecognition")
    parser["MobRecognition"]["Debug"] = str(settings.mob_recognition_debug)

    _ensure_section(parser, "Settings")
    parser["Settings"]["SearchRange"] = str(settings.search_range)
    parser["Settings"]["HuntMode"] = settings.hunt_mode
    parser["Settings"]["TimeOnLocation"] = str(settings.time_on_location)
    parser["Settings"]["WeightModifier"] = str(settings.weight_modifier)
    parser["Settings"]["TakeFlyWings"] = "1" if settings.take_fly_wings else "0"
    parser["Settings"]["FlyWingsAmount"] = str(settings.fly_wings_amount)
    parser["Settings"]["DetectCaptcha"] = "1" if settings.detect_captcha else "0"
    parser["Settings"]["HuntLogOverlay"] = "1" if settings.hunt_log_overlay else "0"
    parser["Settings"]["HuntValidationLog"] = "1" if settings.hunt_validation_log else "0"
    parser["Settings"]["UseSpriteGrf"] = "1" if settings.use_sprite_grf else "0"

    _ensure_section(parser, "Warper")
    if settings.warper_coords_set:
        parser["Warper"]["X"] = settings.warper_x
        parser["Warper"]["Y"] = settings.warper_y
        parser["Warper"]["warperLocation"] = str(settings.warper_location)
    else:
        parser["Warper"].pop("X", None)
        parser["Warper"].pop("Y", None)
        parser["Warper"].pop("warperLocation", None)

    _ensure_section(parser, "Keybindings")
    parser["Keybindings"]["SkillButton"] = settings.skill_button
    parser["Keybindings"]["SkillDelay"] = str(settings.skill_delay)
    parser["Keybindings"]["TeleportButton"] = settings.teleport_button
    parser["Keybindings"]["SavePointButton"] = settings.save_point_button
    parser["Keybindings"]["SPButton"] = settings.sp_button
    parser["Keybindings"]["OpenStorageButton"] = settings.open_storage_button
    parser["Keybindings"]["SkillTimerButton"] = settings.skill_timer_button
    parser["Keybindings"]["SkillTimerInterval"] = str(settings.skill_timer_interval)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        parser.write(handle)
