"""Application and hunt settings schema."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from pybot.paths import CONFIG_PATH

MAX_SKILL_TIMERS = 6
MAX_OPEN_STORAGE_STEPS = 7


@dataclass
class SkillTimerSetting:
    """One periodic skill-timer key press."""

    button: str = ""
    interval_s: int = 20


@dataclass
class KeyChainStep:
    """One key + post-key delay in an Open Storage chain."""

    button: str = ""
    delay_ms: int = 0


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
    # Generic only: OCR Basic Info for SP/Weight (ignored when memory reading is on).
    visual_status_reading: bool = True

    selected_monster: int = 1

    search_range: int = 16
    hunt_mode: str = "teleport"
    time_on_location: int = 20
    weight_modifier: int = 80
    take_fly_wings: bool = False
    fly_wings_amount: int = 100
    detect_captcha: bool = False
    hunt_log_overlay: bool = True
    hunt_validation_log: bool = True

    warper_x: str = ""
    warper_y: str = ""
    warper_location: int = 0

    skill_button: str = "e"
    skill_delay: int = 500
    teleport_button: str = "q"
    creamy_tp_button: str = ""
    teleport_delay: int = 800
    save_point_button: str = ""
    sp_button: str = ""
    open_storage_chain: list[KeyChainStep] = field(default_factory=list)
    skill_timers: list[SkillTimerSetting] = field(default_factory=list)
    sit_on_low_sp: bool = False
    sit_on_low_sp_button: str = "insert"

    @property
    def warper_coords_set(self) -> bool:
        return bool(self.warper_x and self.warper_y)
