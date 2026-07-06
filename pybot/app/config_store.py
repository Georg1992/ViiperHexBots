"""Read/write config.ini for the Python application."""

from __future__ import annotations

import configparser
from dataclasses import dataclass, field
from pathlib import Path

from pybot.paths import PROJECT_ROOT

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.ini"


@dataclass
class AppConfig:
    config_path: Path = field(default_factory=lambda: DEFAULT_CONFIG_PATH)

    last_session_title: str = ""
    last_session_process: str = ""

    window_id: int = 0
    window_title: str = ""
    window_process: str = ""

    client_profile: str = "Generic"
    use_memory_reading: bool = False

    selected_monster: int = 1
    mob_recognition_debug: int = 0

    search_range: int = 16
    time_on_location: int = 20
    weight_modifier: int = 49
    take_fly_wings: bool = False
    fly_wings_amount: int = 100
    detect_captcha: bool = False
    hunt_log_overlay: bool = True
    hunt_validation_log: bool = True
    use_sprite_grf: bool = False

    warper_x: str = ""
    warper_y: str = ""
    warper_location: int = 0

    skill_button: str = "e"
    skill_delay: int = 500
    teleport_button: str = "q"
    save_point_button: str = ""
    sp_button: str = ""
    open_storage_button: str = ""
    skill_timer_button: str = ""
    skill_timer_interval: int = 20

    @property
    def warper_coords_set(self) -> bool:
        return bool(self.warper_x and self.warper_y)

    def load(self) -> AppConfig:
        parser = configparser.ConfigParser()
        if self.config_path.is_file():
            parser.read(self.config_path, encoding="utf-8")

        self.last_session_title = parser.get("LastSession", "GameTitle", fallback="")
        self.last_session_process = parser.get("LastSession", "GameProcess", fallback="")

        self.window_id = parser.getint("Window", "ID", fallback=0)
        self.window_title = parser.get("Window", "Title", fallback="")
        self.window_process = parser.get("Window", "Process", fallback="")

        self.client_profile = parser.get("Client", "Profile", fallback="Generic")
        self.use_memory_reading = parser.getint("Client", "UseMemoryReading", fallback=0) == 1

        self.selected_monster = parser.getint("MonsterSettings", "SelectedMonster", fallback=1)
        self.mob_recognition_debug = parser.getint("MobRecognition", "Debug", fallback=0)

        self.search_range = parser.getint("Settings", "SearchRange", fallback=16)
        self.time_on_location = parser.getint("Settings", "TimeOnLocation", fallback=20)
        self.weight_modifier = parser.getint("Settings", "WeightModifier", fallback=49)
        self.take_fly_wings = parser.getint("Settings", "TakeFlyWings", fallback=0) == 1
        self.fly_wings_amount = parser.getint("Settings", "FlyWingsAmount", fallback=100)
        self.detect_captcha = parser.getint("Settings", "DetectCaptcha", fallback=0) == 1
        self.hunt_log_overlay = parser.getint("Settings", "HuntLogOverlay", fallback=1) == 1
        self.hunt_validation_log = parser.getint("Settings", "HuntValidationLog", fallback=1) == 1
        self.use_sprite_grf = parser.getint("Settings", "UseSpriteGrf", fallback=0) == 1

        self.warper_x = parser.get("Warper", "X", fallback="")
        if self.warper_x == "ERROR":
            self.warper_x = ""
        self.warper_y = parser.get("Warper", "Y", fallback="")
        if self.warper_y == "ERROR":
            self.warper_y = ""
        self.warper_location = parser.getint("Warper", "warperLocation", fallback=0)

        self.skill_button = parser.get("Keybindings", "SkillButton", fallback="e")
        self.skill_delay = parser.getint("Keybindings", "SkillDelay", fallback=500)
        self.teleport_button = parser.get("Keybindings", "TeleportButton", fallback="q")
        self.save_point_button = parser.get("Keybindings", "SavePointButton", fallback="")
        self.sp_button = parser.get("Keybindings", "SPButton", fallback="")
        self.open_storage_button = parser.get("Keybindings", "OpenStorageButton", fallback="")
        self.skill_timer_button = parser.get("Keybindings", "SkillTimerButton", fallback="")
        self.skill_timer_interval = parser.getint("Keybindings", "SkillTimerInterval", fallback=20)
        return self

    def save(self) -> None:
        path = self.config_path
        if path.is_file():
            path.unlink()

        parser = configparser.ConfigParser()
        parser["LastSession"] = {
            "GameProcess": self.last_session_process,
            "GameTitle": self.last_session_title,
        }
        parser["Window"] = {
            "ID": str(self.window_id),
            "Title": self.window_title,
            "Process": self.window_process,
        }
        parser["Client"] = {
            "Profile": self.client_profile,
            "UseMemoryReading": "1" if self.use_memory_reading else "0",
        }
        parser["MonsterSettings"] = {"SelectedMonster": str(self.selected_monster)}
        parser["MobRecognition"] = {"Debug": str(self.mob_recognition_debug)}
        parser["Settings"] = {
            "SearchRange": str(self.search_range),
            "TimeOnLocation": str(self.time_on_location),
            "WeightModifier": str(self.weight_modifier),
            "TakeFlyWings": "1" if self.take_fly_wings else "0",
            "FlyWingsAmount": str(self.fly_wings_amount),
            "DetectCaptcha": "1" if self.detect_captcha else "0",
            "HuntLogOverlay": "1" if self.hunt_log_overlay else "0",
            "HuntValidationLog": "1" if self.hunt_validation_log else "0",
            "UseSpriteGrf": "1" if self.use_sprite_grf else "0",
        }
        parser["Warper"] = {}
        if self.warper_coords_set:
            parser["Warper"]["X"] = self.warper_x
            parser["Warper"]["Y"] = self.warper_y
            parser["Warper"]["warperLocation"] = str(self.warper_location)
        parser["Keybindings"] = {
            "SkillButton": self.skill_button,
            "SkillDelay": str(self.skill_delay),
            "TeleportButton": self.teleport_button,
            "SavePointButton": self.save_point_button,
            "SPButton": self.sp_button,
            "OpenStorageButton": self.open_storage_button,
            "SkillTimerButton": self.skill_timer_button,
            "SkillTimerInterval": str(self.skill_timer_interval),
        }

        with path.open("w", encoding="utf-8") as handle:
            parser.write(handle)


def list_client_profiles(project_root: Path | None = None) -> list[str]:
    root = project_root or PROJECT_ROOT
    clients_dir = root / "clients"
    if not clients_dir.is_dir():
        return ["Generic"]
    names = sorted(path.stem for path in clients_dir.glob("*.json"))
    return names or ["Generic"]


def client_supports_memory(profile_name: str, project_root: Path | None = None) -> bool:
    import json

    root = project_root or PROJECT_ROOT
    profile_path = root / "clients" / f"{profile_name}.json"
    if not profile_path.is_file():
        return False
    data = json.loads(profile_path.read_text(encoding="utf-8"))
    memory = data.get("memory")
    if not isinstance(memory, dict):
        return False
    return bool(memory.get("currentLocationAddress"))
