"""Hunt timing constants"""

from __future__ import annotations

from pybot.settings_defaults import (
    DEFAULT_SEARCH_RANGE_CELLS,
    STORAGE_WEIGHT_MODIFIER_MAX,
    STORAGE_WEIGHT_MODIFIER_MIN,
)
from pybot.recognition.rules import HUNT_OBJECT_RADIUS

HUNT_DISCOVERY_INTERVAL_MS = 250
WORKER_SHUTDOWN_TIMEOUT_S = 2.0
CELL_SIZE_PX = 64
WORKER_POLL_INTERVAL_S = 0.05
# Tracking needs to follow moving mobs, but must yield the shared capture
# session so discovery and character-state sampling are not starved.
TRACKING_LOOP_INTERVAL_S = 0.02
# Attack must not click a coordinate that tracking has only held through misses.
# Normal hits are ~20-50ms old; this allows a short capture hiccup without
# turning a prolonged tracking failure into stale combat input.
MAX_ATTACK_COORD_AGE_MS = 250
LOG_REPEAT_INTERVAL_MS = 5000
# A discovery or tracking pass taking longer than this gets a stage-timing
# warning (capture / lock-wait / compute split) so stalls are diagnosable.
SLOW_SCAN_WARN_MS = 2000
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
# While seated, if the SP feed stays unreadable (OCR layout lost / panel gone /
# stalled native read) this long, recovery relocates: a character parked on an
# unreachable feed can neither finish regen nor react to damage. The same
# window doubles as the regen watchdog: a *readable* SP value that never
# changes for this long while seated means regen is blocked (re-sit toggle
# eaten during a landing, weight penalty, ...), so recovery relocates and
# re-asserts the seated pose in a fresh area.
SIT_SP_FEED_BLIND_RELOCATE_S = 15.0
# While seated, log regen progress at this cadence so a long recovery is
# visibly regenerating instead of looking frozen.
SIT_SP_PROGRESS_LOG_S = 5.0
# Spot failures (frozen SP or blind feed) per recovery session before the
# session ends and the runtime loop takes over again. Each failure already
# teleports to a fresh area, so this only bounds pathological repeat teleports.
SIT_MAX_SPOT_RELOCATIONS = 3
SIT_IDLE_BEFORE_SIT_S = 1.0
# After stand keypress, let the client settle before startup actions,
# discovery, and tracking resume. This also keeps a post-stand buff from
# sharing the first fresh frame with detector work.
SIT_STAND_RESUME_DELAY_S = 0.6
# Wait after sit/stand key tap for the toggle animation to finish.
SIT_KEY_SETTLE_S = 0.35
# Extra margin after a sit-placement / danger-escape teleport before the sit
# toggle. The client's landing transition can eat or invert a key sent too
# early, leaving the character standing while the bot believes it is seated.
# 1.2s: the recorded danger-escape re-sit fired ~0.8s after landing and was
# still inside the client's landing window on the private-server clients.
SIT_POST_TELEPORT_SETTLE_S = 1.2
# Press HP Item Key when vision HP/max is below this.
HP_RESTORE_RATIO = 0.5
# Vision HP poll / min gap between HP Item Key presses.
HP_RESTORE_POLL_S = 1.0
# Minimum gap between successful custom skill-heal casts.
HP_RESTORE_COOLDOWN_S = 1.8
# No HP drop for this long before custom self-heal may run.
HP_HEAL_DAMAGE_QUIET_S = 1.0
# Critical danger must escape at the detector's cadence, not the sit poll.
CRITICAL_DANGER_POLL_INTERVAL_S = WORKER_POLL_INTERVAL_S
# Bounded wait for a preempted session (e.g. storage closing its UI panels)
# before the critical escape presses the teleport key.
CRITICAL_PREEMPT_RELEASE_TIMEOUT_S = 3.0
# After teleport settle, custom self-heal may cast during this grace window.
HP_POST_TELEPORT_HEAL_S = 2.0
# Earliest point at which a custom heal cast may be checked for a fresh HP
# observation. The stricter HP_RESTORE_COOLDOWN_S still controls when a stale
# cast may be classified as blocked and trigger the retry teleport.
HEAL_VERIFY_DELAY_MS = 1000

# Minimum gap between distinct skill-timer key presses when several are due.
SKILL_TIMER_STAGGER_MS = 500
# New-hunt character buffs are deliberately spaced by one second.
STARTUP_BUFF_GAP_S = 1.0
# Let the cursor settle on the character before a startup self-buff cast.
STARTUP_BUFF_CURSOR_DELAY_S = 0.2
# Storage / fly-wings deferred action. Storage is enabled at 50%+, with
# 85% as the maximum supported threshold.
STORAGE_WEIGHT_POLL_INTERVAL_S = 0.25
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
# After moving the cursor off UI before a screen capture.
STORAGE_CURSOR_CLEAR_S = 0.05
__all__ = [
    "HUNT_DISCOVERY_INTERVAL_MS",
    "HUNT_OBJECT_RADIUS",
    "WORKER_SHUTDOWN_TIMEOUT_S",
    "CELL_SIZE_PX",
    "DEFAULT_SEARCH_RANGE_CELLS",
    "WORKER_POLL_INTERVAL_S",
    "TRACKING_LOOP_INTERVAL_S",
    "MAX_ATTACK_COORD_AGE_MS",
    "LOG_REPEAT_INTERVAL_MS",
    "SLOW_SCAN_WARN_MS",
    "ATTACK_IDLE_SPIN_S",
    "SP_IDLE_MAX_OBSERVATION_AGE_MS",
    "DISCOVERY_MISS_REMOVE_COUNT",
    "IDLE_DEAD_ATTACK_COUNT",
    "IDLE_UNREACHABLE_ATTACK_COUNT",
    "MELEE_IDLE_GUARD_RADIUS_PX",
    "SIT_LOW_SP_RATIO",
    "SIT_RESUME_SP_RATIO",
    "SIT_SP_POLL_INTERVAL_S",
    "SIT_SP_FEED_BLIND_RELOCATE_S",
    "SIT_SP_PROGRESS_LOG_S",
    "SIT_MAX_SPOT_RELOCATIONS",
    "SIT_IDLE_BEFORE_SIT_S",
    "SIT_STAND_RESUME_DELAY_S",
    "SIT_KEY_SETTLE_S",
    "SIT_POST_TELEPORT_SETTLE_S",
    "CRITICAL_DANGER_POLL_INTERVAL_S",
    "CRITICAL_PREEMPT_RELEASE_TIMEOUT_S",
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
    "STORAGE_WEIGHT_MODIFIER_MAX",
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
