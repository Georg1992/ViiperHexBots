"""Load and save config.ini with merge-on-write."""

from __future__ import annotations

import configparser
import json
from pathlib import Path

from pybot.config.clients import memory_reading_enabled
from pybot.config.schema import (
    MAX_OPEN_STORAGE_STEPS,
    MAX_SKILL_TIMERS,
    AppSettings,
    KeyChainStep,
    SkillTimerSetting,
)
from pybot.paths import CONFIG_PATH


def _load_open_storage_chain(parser: configparser.ConfigParser) -> list[KeyChainStep]:
    raw = parser.get("Keybindings", "OpenStorageChain", fallback="").strip()
    if raw:
        try:
            items = json.loads(raw)
        except json.JSONDecodeError:
            items = []
        steps: list[KeyChainStep] = []
        if isinstance(items, list):
            for item in items[:MAX_OPEN_STORAGE_STEPS]:
                if not isinstance(item, dict):
                    continue
                button = str(item.get("key") or item.get("button") or "").strip()
                delay = int(item.get("delay") or item.get("delay_ms") or 0)
                steps.append(KeyChainStep(button=button, delay_ms=max(0, delay)))
        return steps

    # Migrate legacy single OpenStorageButton.
    legacy = parser.get("Keybindings", "OpenStorageButton", fallback="").strip()
    if legacy:
        return [KeyChainStep(button=legacy, delay_ms=0)]
    return []


def _save_open_storage_chain(steps: list[KeyChainStep]) -> str:
    payload = [
        {"key": s.button, "delay": int(s.delay_ms)}
        for s in steps[:MAX_OPEN_STORAGE_STEPS]
        if s.button.strip()
    ]
    return json.dumps(payload, separators=(",", ":"))


def _load_skill_timers(parser: configparser.ConfigParser) -> list[SkillTimerSetting]:
    raw = parser.get("Keybindings", "SkillTimers", fallback="").strip()
    if raw:
        try:
            items = json.loads(raw)
        except json.JSONDecodeError:
            items = []
        timers: list[SkillTimerSetting] = []
        if isinstance(items, list):
            for item in items[:MAX_SKILL_TIMERS]:
                if not isinstance(item, dict):
                    continue
                button = str(item.get("key") or item.get("button") or "").strip()
                interval = int(item.get("delay") or item.get("interval_s") or 20)
                timers.append(SkillTimerSetting(button=button, interval_s=max(1, interval)))
        return timers

    # Migrate legacy single-timer keys.
    button = parser.get("Keybindings", "SkillTimerButton", fallback="").strip()
    interval = parser.getint("Keybindings", "SkillTimerInterval", fallback=20)
    if button:
        return [SkillTimerSetting(button=button, interval_s=max(1, interval))]
    return []


def _save_skill_timers(timers: list[SkillTimerSetting]) -> str:
    payload = [
        {"key": t.button, "delay": int(t.interval_s)}
        for t in timers[:MAX_SKILL_TIMERS]
        if t.button.strip()
    ]
    return json.dumps(payload, separators=(",", ":"))


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
        use_memory_reading=memory_reading_enabled(
            parser.get("Client", "Profile", fallback="Generic")
        ),
        visual_status_reading=parser.getint(
            "Client", "VisualStatusReading", fallback=1
        )
        == 1,
        selected_monster=parser.getint("MonsterSettings", "SelectedMonster", fallback=1),
        search_range=parser.getint("Settings", "SearchRange", fallback=16),
        hunt_mode=parser.get("Settings", "HuntMode", fallback="teleport"),
        time_on_location=parser.getint("Settings", "TimeOnLocation", fallback=20),
        weight_modifier=parser.getint("Settings", "WeightModifier", fallback=80),
        take_fly_wings=parser.getint("Settings", "TakeFlyWings", fallback=0) == 1,
        fly_wings_amount=parser.getint("Settings", "FlyWingsAmount", fallback=100),
        detect_captcha=parser.getint("Settings", "DetectCaptcha", fallback=0) == 1,
        hunt_log_overlay=parser.getint("Settings", "HuntLogOverlay", fallback=1) == 1,
        hunt_validation_log=parser.getint("Settings", "HuntValidationLog", fallback=1) == 1,
        warper_x=warper_x,
        warper_y=warper_y,
        warper_location=parser.getint("Warper", "warperLocation", fallback=0),
        skill_button=parser.get("Keybindings", "SkillButton", fallback="e"),
        skill_delay=parser.getint("Keybindings", "SkillDelay", fallback=500),
        teleport_button=parser.get("Keybindings", "TeleportButton", fallback="q"),
        teleport_delay=parser.getint("Keybindings", "TeleportDelay", fallback=800),
        save_point_button=parser.get("Keybindings", "SavePointButton", fallback=""),
        sp_button=parser.get("Keybindings", "SPButton", fallback=""),
        open_storage_chain=_load_open_storage_chain(parser),
        skill_timers=_load_skill_timers(parser),
        sit_on_low_sp=parser.getint("Keybindings", "SitOnLowSp", fallback=0) == 1,
        sit_on_low_sp_button=(
            parser.get("Keybindings", "SitOnLowSpButton", fallback="insert").strip()
            or "insert"
        ),
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
    # Derived from profile (Generic off, server profiles on); keep in sync on save.
    parser["Client"]["UseMemoryReading"] = (
        "1" if memory_reading_enabled(settings.client_profile) else "0"
    )
    parser["Client"]["VisualStatusReading"] = (
        "1" if settings.visual_status_reading else "0"
    )

    _ensure_section(parser, "MonsterSettings")
    parser["MonsterSettings"]["SelectedMonster"] = str(settings.selected_monster)

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
    parser["Settings"].pop("DeathDetectionEnabled", None)
    parser["Settings"].pop("UseSpriteGrf", None)

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
    parser["Keybindings"]["TeleportDelay"] = str(settings.teleport_delay)
    parser["Keybindings"]["SavePointButton"] = settings.save_point_button
    parser["Keybindings"]["SPButton"] = settings.sp_button
    parser["Keybindings"]["OpenStorageChain"] = _save_open_storage_chain(
        settings.open_storage_chain
    )
    parser["Keybindings"].pop("OpenStorageButton", None)
    parser["Keybindings"]["SkillTimers"] = _save_skill_timers(settings.skill_timers)
    parser["Keybindings"].pop("SkillTimerButton", None)
    parser["Keybindings"].pop("SkillTimerInterval", None)
    parser["Keybindings"]["SitOnLowSpButton"] = settings.sit_on_low_sp_button
    parser["Keybindings"]["SitOnLowSp"] = "1" if settings.sit_on_low_sp else "0"

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        parser.write(handle)
