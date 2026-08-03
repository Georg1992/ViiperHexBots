"""Hunt timing constants"""

from __future__ import annotations

from pybot.recognition.rules import HUNT_OBJECT_RADIUS

HUNT_DISCOVERY_INTERVAL_MS = 250
WORKER_SHUTDOWN_TIMEOUT_S = 2.0
CELL_SIZE_PX = 64
DEFAULT_SEARCH_RANGE_CELLS = 16
WORKER_POLL_INTERVAL_S = 0.05
# Tracking needs to follow moving mobs, but must yield the shared capture
# session so discovery and character-state sampling are not starved.
TRACKING_LOOP_INTERVAL_S = 0.02
TRACKING_OVERLAY_INTERVAL_S = 0.10
LOG_REPEAT_INTERVAL_MS = 5000
# Attack loop spin when no target or after one attack (half worker poll).
ATTACK_IDLE_SPIN_S = WORKER_POLL_INTERVAL_S / 2.0
# Idle SP confirmation: post observation must be this fresh at sample time,
# otherwise an early mid-wait republish of pre-cost SP can look like idle.
SP_IDLE_MAX_OBSERVATION_AGE_MS = 1000
# Discovery scans without a match before track removal.
DISCOVERY_MISS_REMOVE_COUNT = 3
# Idle-attack death / unreachable policy.
IDLE_DEAD_ATTACK_COUNT = 2
IDLE_UNREACHABLE_ATTACK_COUNT = 5
MELEE_IDLE_GUARD_RADIUS_PX = 150
SIT_LOW_SP_RATIO = 0.05
SIT_RESUME_SP_RATIO = 0.98
SIT_SP_POLL_INTERVAL_S = 0.25
SIT_IDLE_BEFORE_SIT_S = 1.0
# After stand keypress, delay before hunt/timers resume.
SIT_STAND_RESUME_DELAY_S = 0.5
# Wait after sit/stand key tap for the toggle animation to finish.
SIT_KEY_SETTLE_S = 0.35
# Extra margin after a sit-placement / danger-escape teleport before the sit
# toggle. The client's landing transition can eat or invert a key sent too
# early, leaving the character standing while the bot believes it is seated.
SIT_POST_TELEPORT_SETTLE_S = 0.8
# Press HP Item Key when vision HP/max is below this.
HP_RESTORE_RATIO = 0.5
# Vision HP poll / min gap between HP Item Key presses.
HP_RESTORE_POLL_S = 1.0
HP_RESTORE_COOLDOWN_S = 1.0
# No HP drop for this long before custom self-heal may run.
HP_HEAL_DAMAGE_QUIET_S = 1.0
# After teleport settle, custom self-heal may cast during this grace window.
HP_POST_TELEPORT_HEAL_S = 2.0
# After a custom heal cast, wait this long and require a fresh HP reading
# before another heal may be sent (vitals refresh lags the game state).
HEAL_VERIFY_DELAY_MS = 300

# Minimum gap between distinct skill-timer key presses when several are due.
SKILL_TIMER_STAGGER_MS = 500
# New-hunt character buffs are deliberately spaced by one second.
STARTUP_BUFF_GAP_S = 1.0
# Let the cursor settle on the character before a startup self-buff cast.
STARTUP_BUFF_CURSOR_DELAY_S = 0.2
# Storage / fly-wings worker (AHK WeightModifier gate is active at >= 50).
STORAGE_WEIGHT_POLL_INTERVAL_S = 0.25
STORAGE_WEIGHT_MODIFIER_MIN = 50
# RO fly wing unit weight — used to decide ItemsToStorage before GetFlyWings.
FLY_WING_WEIGHT = 5
# AHK ItemsToStorage OK-dialog Enter (extended scan code).
STORAGE_ENTER_SCAN_CODE = 284
# Always wait this long after Alt+mouse click (deposit).
ALT_MOUSE_CLICK_DELAY_S = 0.1
# Gap between the two clicks of a kiting double-click so the client
# registers them as a deliberate double-click walk command.
KITE_DOUBLE_CLICK_GAP_S = 0.08
# Settle after moving onto a Use-tab fly wing before Alt+RMB deposit.
STORAGE_WING_AIM_SETTLE_S = 0.25
# Use-tab grid from assets/UI/InventoryPanel.png (8×6, 32px pitch).
STORAGE_INV_COLS = 8
STORAGE_INV_ROWS = 6
# Wait for inventory panel after Alt+E before clicking tabs/slots.
STORAGE_INV_OPEN_TIMEOUT_S = 2.0
STORAGE_INV_OPEN_POLL_S = 0.1
# Shared open/closed menu validation timeout (inventory + storage).
STORAGE_MENU_TIMEOUT_S = 2.0
STORAGE_MENU_POLL_S = 0.1
# After inventory/storage open, close, or tab switch — UI needs time to draw.
STORAGE_UI_SETTLE_S = 0.1
# After moving the cursor off UI before a template capture.
STORAGE_CURSOR_CLEAR_S = 0.05
__all__ = [
    "HUNT_DISCOVERY_INTERVAL_MS",
    "HUNT_OBJECT_RADIUS",
    "WORKER_SHUTDOWN_TIMEOUT_S",
    "CELL_SIZE_PX",
    "DEFAULT_SEARCH_RANGE_CELLS",
    "WORKER_POLL_INTERVAL_S",
    "TRACKING_LOOP_INTERVAL_S",
    "TRACKING_OVERLAY_INTERVAL_S",
    "LOG_REPEAT_INTERVAL_MS",
    "ATTACK_IDLE_SPIN_S",
    "SP_IDLE_MAX_OBSERVATION_AGE_MS",
    "DISCOVERY_MISS_REMOVE_COUNT",
    "IDLE_DEAD_ATTACK_COUNT",
    "IDLE_UNREACHABLE_ATTACK_COUNT",
    "MELEE_IDLE_GUARD_RADIUS_PX",
    "SIT_LOW_SP_RATIO",
    "SIT_RESUME_SP_RATIO",
    "SIT_SP_POLL_INTERVAL_S",
    "SIT_IDLE_BEFORE_SIT_S",
    "SIT_STAND_RESUME_DELAY_S",
    "SIT_KEY_SETTLE_S",
    "SIT_POST_TELEPORT_SETTLE_S",
    "HP_RESTORE_RATIO",
    "HP_RESTORE_POLL_S",
    "HP_RESTORE_COOLDOWN_S",
    "HP_HEAL_DAMAGE_QUIET_S",
    "HP_POST_TELEPORT_HEAL_S",
    "HEAL_VERIFY_DELAY_MS",
    "SKILL_TIMER_STAGGER_MS",
    "STARTUP_BUFF_GAP_S",
    "STARTUP_BUFF_CURSOR_DELAY_S",
    "STORAGE_WEIGHT_POLL_INTERVAL_S",
    "STORAGE_WEIGHT_MODIFIER_MIN",
    "FLY_WING_WEIGHT",
    "STORAGE_ENTER_SCAN_CODE",
    "ALT_MOUSE_CLICK_DELAY_S",
    "KITE_DOUBLE_CLICK_GAP_S",
    "STORAGE_WING_AIM_SETTLE_S",
    "STORAGE_INV_COLS",
    "STORAGE_INV_ROWS",
    "STORAGE_INV_OPEN_TIMEOUT_S",
    "STORAGE_INV_OPEN_POLL_S",
    "STORAGE_MENU_TIMEOUT_S",
    "STORAGE_MENU_POLL_S",
    "STORAGE_UI_SETTLE_S",
    "STORAGE_CURSOR_CLEAR_S",
]
