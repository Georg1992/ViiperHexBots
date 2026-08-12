"""Application and hunt settings schema."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from pybot.settings_defaults import (
    DEFAULT_FLY_WINGS_AMOUNT,
    DEFAULT_SEARCH_RANGE_CELLS,
    DEFAULT_SIT_ON_LOW_SP_BUTTON,
    DEFAULT_SKILL_DELAY_MS,
    DEFAULT_SKILL_TIMER_INTERVAL_S,
    DEFAULT_TELEPORT_DELAY_MS,
    DEFAULT_WEIGHT_MODIFIER,
)
from pybot.paths import CONFIG_PATH

MAX_SKILL_TIMERS = 6
MAX_OPEN_STORAGE_STEPS = 7


@dataclass
class MobCustomSettings:
    """Per-mob kiting and self-cast skill settings."""

    kiting_tick_s: float = 0.0
    # Optional by design: kiting stays disabled until a distance is configured.
    kite_distance_cells: int | None = None
    debuff_button: str = ""
    heal_button: str = ""
    buff1_button: str = ""
    buff1_delay_s: int = 0
    buff2_button: str = ""
    buff2_delay_s: int = 0
    buff3_button: str = ""
    buff3_delay_s: int = 0


@dataclass
class SkillTimerSetting:
    """One periodic skill-timer key press."""

    button: str = ""
    interval_s: int = DEFAULT_SKILL_TIMER_INTERVAL_S


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

    selected_monster: int = 1

    search_range: int = DEFAULT_SEARCH_RANGE_CELLS
    hunt_mode: str = "teleport"
    weight_modifier: int = DEFAULT_WEIGHT_MODIFIER
    take_fly_wings: bool = False
    fly_wings_amount: int = DEFAULT_FLY_WINGS_AMOUNT
    detect_captcha: bool = False
    hunt_log_overlay: bool = True
    hunt_validation_log: bool = True

    warper_x: str = ""
    warper_y: str = ""
    warper_location: int = 0

    skill_button: str = "e"
    skill_delay: int = DEFAULT_SKILL_DELAY_MS
    teleport_button: str = ""
    creamy_tp_button: str = ""
    teleport_delay: int = DEFAULT_TELEPORT_DELAY_MS
    save_point_button: str = ""
    hp_button: str = ""
    sp_button: str = ""
    open_storage_chain: list[KeyChainStep] = field(default_factory=list)
    skill_timers: list[SkillTimerSetting] = field(default_factory=list)
    mob_custom_settings: dict[str, MobCustomSettings] = field(default_factory=dict)
    sit_on_low_sp: bool = False
    sit_on_low_sp_button: str = DEFAULT_SIT_ON_LOW_SP_BUTTON

    use_sprite_grf: bool = False

    @property
    def warper_coords_set(self) -> bool:
        return bool(self.warper_x and self.warper_y)
