"""Smoke test: exercise every import and init path touched by start_bot().

Catches ctypes/wintypes/import errors BEFORE the user sees them in the GUI.
"""
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

errors = []
def check(label, fn):
    try:
        fn()
        print(f"  OK  {label}")
    except Exception as e:
        errors.append(label)
        print(f"  FAIL {label}: {type(e).__name__}: {e}")

# ── Phase 1: All module imports ────────────────────────────────
print("=== Phase 1: Module imports ===")

check("win32_util (with WINDOWPLACEMENT)",
    lambda: __import__("pybot.app.win32_util", fromlist=["WINDOWPLACEMENT"]))

check("overlay (with WNDCLASSW, PAINTSTRUCT, lpfnWndProc cast)",
    lambda: __import__("pybot.runtime.overlay", fromlist=["create"]))

check("memory_reader (with SIZE_T shim)",
    lambda: __import__("pybot.runtime.memory_reader", fromlist=["MemoryReader"]))

check("config",
    lambda: __import__("pybot.runtime.config", fromlist=["load_runtime_config"]))

check("hunt_runtime",
    lambda: __import__("pybot.runtime.hunt_runtime", fromlist=["create_runtime_deps", "HuntRuntime"]))

check("bot_controller",
    lambda: __import__("pybot.app.bot_controller", fromlist=["BotController"]))

check("bot_lifecycle",
    lambda: __import__("pybot.app.bot_lifecycle", fromlist=["BotLifecycleManager"]))

check("config_store",
    lambda: __import__("pybot.app.config_store", fromlist=["AppConfig"]))

check("mob_catalog",
    lambda: __import__("pybot.app.mob_catalog", fromlist=["load_mob_catalog"]))

check("main_window",
    lambda: __import__("pybot.app.main_window", fromlist=["MainWindow"]))

# Phase 1b: All worker modules
print("  --- worker modules ---")
check("discovery_worker",
    lambda: __import__("pybot.runtime.workers.discovery_worker", fromlist=["DiscoveryWorker"]))
check("tracking_worker",
    lambda: __import__("pybot.runtime.workers.tracking_worker", fromlist=["TrackingWorker"]))
check("confirm_state_worker",
    lambda: __import__("pybot.runtime.workers.confirm_state_worker", fromlist=["ConfirmStateWorker"]))
check("attack_loop",
    lambda: __import__("pybot.runtime.workers.attack_loop", fromlist=["AttackLoop"]))
check("skill_timer_worker",
    lambda: __import__("pybot.runtime.workers.skill_timer_worker", fromlist=["SkillTimerWorker"]))
check("worker_contexts",
    lambda: __import__("pybot.runtime.workers.worker_contexts", fromlist=["DiscoveryWorkerContext"]))

# Phase 1c: Detection modules
print("  --- detection modules ---")
check("detector_session",
    lambda: __import__("pybot.runtime.detection.detector_session", fromlist=["DetectorSession"]))
check("discovery_filter",
    lambda: __import__("pybot.runtime.detection.discovery_filter", fromlist=["filter_scan_candidates"]))

# Phase 1d: Input modules
print("  --- input modules ---")
check("input_backend",
    lambda: __import__("pybot.runtime.input.input_backend", fromlist=["InputBackend", "ShadowInputBackend"]))
check("viiper_backend",
    lambda: __import__("pybot.runtime.input.viiper_backend", fromlist=["ViiperBackend"]))

# Phase 1e: VIIPER client
print("  --- viiper client ---")
check("viiper_client",
    lambda: __import__("pybot.viiper.client", fromlist=["ViiperClient"]))
check("viiper_keyboard",
    lambda: __import__("pybot.viiper.keyboard", fromlist=["KeyboardState"]))
check("viiper_mouse",
    lambda: __import__("pybot.viiper.mouse", fromlist=["MouseState"]))
check("viiper_stream",
    lambda: __import__("pybot.viiper.stream", fromlist=["DeviceStream"]))

# ── Phase 2: Struct/type instantiation ──────────────────────────
print("\n=== Phase 2: Struct instantiation ===")
import ctypes
from ctypes import wintypes

from pybot.runtime.overlay import WNDCLASSW, PAINTSTRUCT
from pybot.app.win32_util import WINDOWPLACEMENT

check("WINDOWPLACEMENT()",
    lambda: WINDOWPLACEMENT())

check("PAINTSTRUCT()",
    lambda: PAINTSTRUCT())

check("WNDCLASSW() + lpfnWndProc assignment",
    lambda: setattr(WNDCLASSW(), "lpfnWndProc", ctypes.cast(0, ctypes.c_void_p)))

def _test_wndclass_callback():
    from pybot.runtime.overlay import _WND_PROC_CALLBACK
    cls = WNDCLASSW()
    cls.lpfnWndProc = ctypes.cast(_WND_PROC_CALLBACK, ctypes.c_void_p)

check("WNDCLASSW() with real callback",
    lambda: _test_wndclass_callback())

# ── Phase 3: Runtime deps creation ─────────────────────────────
print("\n=== Phase 3: create_runtime_deps ===")
from pybot.runtime.config import load_runtime_config
from pybot.runtime.hunt_runtime import create_runtime_deps

check("load_runtime_config(empty)",
    lambda: load_runtime_config(hwnd=0, mob_name="horn", session_id="smoke"))

config = load_runtime_config(hwnd=0, mob_name="horn", session_id="smoke")
print(f"  INFO: config mob={config.mob_name} hwnd={config.hwnd}")

check("create_runtime_deps",
    lambda: create_runtime_deps(config, session_id="smoke"))

deps = create_runtime_deps(config, session_id="smoke")
print(f"  INFO: deps has {len(deps.workers)} workers: {[name for name, _ in deps.workers]}")

check("HuntRuntime(deps)",
    lambda: __import__("pybot.runtime.hunt_runtime", fromlist=["HuntRuntime"]).HuntRuntime(deps))

# ── Phase 4: enum_game_windows ──────────────────────────────────
print("\n=== Phase 4: enum_game_windows ===")
from pybot.app.win32_util import enum_game_windows
check("enum_game_windows()",
    lambda: enum_game_windows())

windows = enum_game_windows()
print(f"  INFO: found {len(windows)} windows")

# ── Phase 5: Mob catalog ────────────────────────────────────────
print("\n=== Phase 5: Mob catalog ===")
from pybot.app.mob_catalog import load_mob_catalog, mob_folder_by_index
check("load_mob_catalog",
    lambda: load_mob_catalog())

catalog = load_mob_catalog()
print(f"  INFO: catalog has {len(catalog)} entries: {[e.folder_name for e in catalog]}")

if catalog:
    check("mob_folder_by_index(1)",
        lambda: mob_folder_by_index(catalog, 1))

# ── Report ──────────────────────────────────────────────────────
print(f"\n{'='*60}")
if errors:
    print(f"FAILED: {len(errors)} error(s):")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print("ALL CHECKS PASSED — ready for GUI launch")
    sys.exit(0)
