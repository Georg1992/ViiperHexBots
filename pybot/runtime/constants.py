"""Hunt timing constants"""

from __future__ import annotations

from pybot.recognition.rules import HUNT_OBJECT_RADIUS, HUNT_TRACK_LOST_LIMIT

HUNT_DISCOVERY_INTERVAL_MS = 1000
HUNT_TELEPORT_DURATION_MS = 800
WORKER_SHUTDOWN_TIMEOUT_S = 2.0
CELL_SIZE_PX = 64
DEFAULT_SEARCH_RANGE_CELLS = 16
WORKER_POLL_INTERVAL_S = 0.05
LOG_REPEAT_INTERVAL_MS = 5000
SIT_LOW_SP_RATIO = 0.05
SIT_RESUME_SP_RATIO = 0.98
SIT_SP_POLL_INTERVAL_S = 0.25
SIT_IDLE_BEFORE_SIT_S = 1.0
# After stand keypress, delay before hunt/timers resume.
SIT_STAND_RESUME_DELAY_S = 0.5
# Minimum gap between distinct skill-timer key presses when several are due.
SKILL_TIMER_STAGGER_MS = 500
# Storage / fly-wings worker (AHK WeightModifier gate is active at >= 50).
STORAGE_WEIGHT_POLL_INTERVAL_S = 0.25
STORAGE_WEIGHT_MODIFIER_MIN = 50
# AHK ItemsToStorage OK-dialog Enter (extended scan code).
STORAGE_ENTER_SCAN_CODE = 284
# Always wait this long after Alt+mouse click (deposit).
ALT_MOUSE_CLICK_DELAY_S = 0.1
# Offset from cell1 template top-left into the first inventory cell center.
STORAGE_CELL1_OFFSET_X = 20
STORAGE_CELL1_OFFSET_Y = 20
# Visible Use-tab item grid (measured on client inventory: 6×6 before window chrome).
STORAGE_INV_COLS = 6
STORAGE_INV_ROWS = 6

__all__ = [
    "HUNT_DISCOVERY_INTERVAL_MS",
    "HUNT_OBJECT_RADIUS",
    "HUNT_TELEPORT_DURATION_MS",
    "WORKER_SHUTDOWN_TIMEOUT_S",
    "HUNT_TRACK_LOST_LIMIT",
    "CELL_SIZE_PX",
    "DEFAULT_SEARCH_RANGE_CELLS",
    "WORKER_POLL_INTERVAL_S",
    "LOG_REPEAT_INTERVAL_MS",
    "SIT_LOW_SP_RATIO",
    "SIT_RESUME_SP_RATIO",
    "SIT_SP_POLL_INTERVAL_S",
    "SIT_IDLE_BEFORE_SIT_S",
    "SIT_STAND_RESUME_DELAY_S",
    "SKILL_TIMER_STAGGER_MS",
    "STORAGE_WEIGHT_POLL_INTERVAL_S",
    "STORAGE_WEIGHT_MODIFIER_MIN",
    "STORAGE_ENTER_SCAN_CODE",
    "ALT_MOUSE_CLICK_DELAY_S",
    "STORAGE_CELL1_OFFSET_X",
    "STORAGE_CELL1_OFFSET_Y",
    "STORAGE_INV_COLS",
    "STORAGE_INV_ROWS",
]
