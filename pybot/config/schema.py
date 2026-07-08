"""Application and hunt settings schema."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from pybot.paths import CONFIG_PATH


@dataclass
class AppSettings:
    config_path: Path = field(default_factory=lambda: CONFIG_PATH)

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
    hunt_mode: str = "teleport"
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
