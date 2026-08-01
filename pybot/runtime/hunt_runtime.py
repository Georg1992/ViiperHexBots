"""Python hunt runtime entry point."""

from __future__ import annotations

import argparse
import signal
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from pybot.runtime.capture.hunt_capture import HuntWindowCapture
from pybot.runtime.config import load_runtime_config
from pybot.runtime.control import RuntimeControl
from pybot.runtime.hunt_mode import HuntModeController, create_hunt_mode
from pybot.runtime.hunt_policy import HuntPolicy
from pybot.runtime.hunt_tracks import HuntTracks
from pybot.runtime.input.input_backend import InputBackend
from pybot.runtime.input.viiper_backend import ViiperBackend
from pybot.runtime.logging import HuntLogger
from pybot.runtime.overlay_ports import HuntOverlay, NullOverlay
from pybot.runtime.runtime_context import HuntRuntimeContext
from pybot.runtime.validation_log import HuntValidationLogger
from pybot.recognition.capture import reset_capture_session
from pybot.recognition.detector.detector import load_detector_config
from pybot.runtime.detection.detector_session import DetectorSession
from pybot.runtime.workers.attack_loop import AttackLoop
from pybot.runtime.workers.coord_tracking_worker import CoordTrackingWorker
from pybot.runtime.mob_behaviors import (
    get_configured_mob_behavior,
    get_mob_behavior,
)
from pybot.runtime.danger_detector import DangerDetector
from pybot.runtime.teleport import TeleportController
from pybot.runtime.character_state import CharacterState
from pybot.runtime.workers.character_state_worker import CharacterStateMonitor
from pybot.game_state import PlayerVitals

from pybot.runtime.workers.discovery_worker import DiscoveryWorker
from pybot.runtime.workers.skill_timer_worker import SkillTimerWorker
from pybot.runtime.workers.self_buff_worker import SelfBuffWorker
from pybot.config.clients import (
    MemoryAddresses,
    load_client_profile,
    memory_reading_enabled,
)
from pybot.runtime.constants import (
    STORAGE_WEIGHT_MODIFIER_MIN,
    WORKER_SHUTDOWN_TIMEOUT_S,
)
from pybot.runtime.workers.items_to_storage_worker import ItemsToStorageWorker
from pybot.runtime.workers.sit_on_low_sp_worker import SitOnLowSpWorker
from pybot.runtime.workers.hp_restore_worker import HpRestoreWorker
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ViiperHexBots Python hunt runtime")
    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("run", help="Start hunt runtime (default)")
    run.add_argument("--hwnd", type=int, default=0, help="Game window handle")
    run.add_argument("--mob", type=str, default="")
    run.add_argument("--config", type=str, default="")
    run.add_argument("--run-seconds", type=float, default=0.0)
    run.add_argument("--start-paused", action="store_true")
    run.add_argument("--control-file", type=str, default="")
    run.add_argument("--session-id", type=str, default="")

    for name, help_text in (
        ("stop", "Write stop command to control file"),
        ("pause", "Write pause command to control file"),
        ("resume", "Write resume command to control file"),
    ):
        cmd = sub.add_parser(name, help=help_text)
        cmd.add_argument("--control-file", type=str, required=True)

    parser.set_defaults(command="run")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    if argv is None:
        return parser.parse_args()
    if argv and argv[0] not in {"run", "stop", "pause", "resume", "-h", "--help"}:
        return parser.parse_args(["run", *argv])
    return parser.parse_args(argv)


def write_control_command(command: str, control_file: str) -> int:
    RuntimeControl(Path(control_file)).write_command(command)
    print(f"[PYBOT] control {command} -> {control_file}")
    return 0


@dataclass
class RuntimeDependencies:
    """Pre-built dependencies ready to inject into HuntRuntime.
    Use create_runtime_deps() to build this container.
    """
    ctx: HuntRuntimeContext
    input_backend: InputBackend
    hunt_mode: HuntModeController
    logger: HuntLogger
    workers: list[tuple[str, Callable[[], None]]]
    # Optional for lightweight test/runtime fixtures; production always wires it.
    teleport_controller: TeleportController | None = None


def _build_logger(
    config,
    session_id: str | None,
    behavior_callback: Callable[[str], None] | None,
) -> HuntLogger:
    """Create the hunt logger with optional behavior callback."""
    sid = session_id or time.strftime("%Y%m%d_%H%M%S")
    logger = HuntLogger(
        session_id=sid,
        echo_stdout=behavior_callback is None,
    )
    if behavior_callback:
        logger.set_behavior_callback(behavior_callback)
    return logger


def _build_detectors(config) -> tuple[DetectorSession, DetectorSession]:
    """Create two independent detector sessions (discovery + tracking)."""
    detector_config = load_detector_config()
    detector = DetectorSession(
        config.mob_name,
        detector_config=detector_config,
        use_sprite_grf=config.use_sprite_grf,
    )
    tracker = DetectorSession(
        config.mob_name,
        detector_config=detector_config,
        use_sprite_grf=config.use_sprite_grf,
    )
    return detector, tracker


def _build_context(
    config,
    logger: HuntLogger,
    detector: DetectorSession,
    tracker: DetectorSession,
    overlay: HuntOverlay | None,
) -> HuntRuntimeContext:
    """Build the shared runtime context with all core services."""
    detector_config = load_detector_config()
    tracks = HuntTracks(detector_config)
    policy = HuntPolicy()
    capture = HuntWindowCapture(config)
    validation = HuntValidationLogger(
        logger, tracks,
        enabled=config.validation_enabled,
    )
    control = RuntimeControl(config.control_file)
    return HuntRuntimeContext(
        config=config,
        logger=logger,
        tracks=tracks,
        policy=policy,
        capture=capture,
        detector=detector,
        tracker=tracker,
        validation=validation,
        control=control,
        overlay=overlay or NullOverlay(),
    )


def _validate_teleport_mode(config, tport: TeleportController) -> None:
    """Fail early if teleport mode has no configured key."""
    if config.hunt_mode == "teleport" and tport.active_scan_code() <= 0:
        tp_button = config.teleport_button or "(unset)"
        creamy = config.creamy_tp_button or "(unset)"
        raise ValueError(
            f"Teleport hunt mode requires a teleport key. "
            f"Set at least one of Teleport Key={tp_button!r} "
            f"or Creamy TP Key={creamy!r} in the Keybindings tab."
        )


def _validate_sp_memory(config) -> None:
    """Fail early if sit-on-low-sp is on but SP memory addresses are missing."""
    if not config.sit_on_low_sp:
        return
    if config.sit_on_low_sp_scan_code <= 0:
        raise ValueError(
            "Sit On Low Sp is On but the sit key is invalid "
            f"(button={config.sit_on_low_sp_button!r})."
        )
    profile = load_client_profile(config.client_profile)
    memory = MemoryAddresses() if profile is None else profile.memory
    has_sp_memory = memory.current_sp > 0 and memory.max_sp > 0
    if not has_sp_memory and memory_reading_enabled(config.client_profile):
        raise ValueError(
            "Sit On Low Sp requires a client profile with currentSpAddress "
            f"and maxSpAddress (profile={config.client_profile!r})."
        )


def _validate_weight_memory(config) -> None:
    """Fail early if storage is on but weight memory addresses are missing."""
    if not config.open_storage_steps:
        return
    if config.weight_modifier < STORAGE_WEIGHT_MODIFIER_MIN:
        return
    profile = load_client_profile(config.client_profile)
    memory = MemoryAddresses() if profile is None else profile.memory
    has_weight_memory = memory.current_weight > 0 and memory.max_weight > 0
    if not has_weight_memory and memory_reading_enabled(config.client_profile):
        raise ValueError(
            "Open Storage requires a client profile with currentWeightAddress "
            f"and totalWeightAddress (profile={config.client_profile!r})."
        )


def _build_core_workers(
    ctx: HuntRuntimeContext,
    hunt_mode: HuntModeController,
    input_backend: InputBackend,
    tport: TeleportController,
    player_vitals: PlayerVitals,
    mob_behavior,
    character_state: CharacterState,
    danger: DangerDetector | None = None,
) -> tuple[list[tuple[str, Callable[[], None]]], DangerDetector]:
    """Build always-running workers: danger, coord, discovery, attack.

    Returns ``(workers, danger_detector)`` so callers can reuse the
    DangerDetector instance for sit-worker injection.
    """
    tracking = CoordTrackingWorker(ctx)
    discovery = DiscoveryWorker(ctx, hunt_mode)
    roi = ctx.capture.get_hunt_roi()
    char_x = roi.x + roi.w // 2 if roi else 0
    char_y = roi.y + roi.h // 2 if roi else 0
    if danger is None:
        danger = DangerDetector(
            ctx, tport, character_state, vitals=player_vitals,
        )
    attack = AttackLoop(
        ctx, hunt_mode, input_backend,
        mob_behavior=mob_behavior,
        vitals=player_vitals,
        char_x=char_x, char_y=char_y,
    )
    workers = [
        ("danger", danger.run),
        ("coord", tracking.run),
        ("discovery", discovery.run),
        ("attack", attack.run),
    ]
    return workers, danger


def _build_conditional_workers(
    ctx: HuntRuntimeContext,
    input_backend: InputBackend,
    tport: TeleportController,
    player_vitals: PlayerVitals,
    character_state: CharacterState,
    danger: DangerDetector | None = None,
) -> list[tuple[str, Callable[[], None]]]:
    """Build optional workers: skill timer, hp restore, sit, storage."""
    workers: list[tuple[str, Callable[[], None]]] = []

    if any(t.scan_code and t.interval_ms > 0 for t in ctx.config.skill_timers):
        workers.append(("skill_timer", SkillTimerWorker(ctx, input_backend).run))
    has_buffs = any(
        buff.scan_code > 0 and buff.delay_ms > 0
        for buff in ctx.config.custom_behavior.buffs
    )
    has_timers = any(
        t.scan_code and t.interval_ms > 0
        for t in ctx.config.skill_timers
    )
    if has_buffs:
        workers.append(("custom_buffs", SelfBuffWorker(ctx, input_backend).run))
    else:
        # No character buffs means normal timers may start immediately.
        ctx.mark_startup_buffs_done()
    if not has_timers:
        # No normal timers means combat may start after the buff sequence.
        # SelfBuffWorker also marks this after its final buff when present.
        if not has_buffs:
            ctx.mark_startup_timers_done()
    if ctx.config.hp_scan_code > 0:
        workers.append(
            (
                "hp_restore",
                HpRestoreWorker(ctx, input_backend, player_vitals).run,
            )
        )

    if ctx.config.sit_on_low_sp:
        if danger is None:
            raise RuntimeError("Sit-on-low-SP requires a shared DangerDetector")
        sit_worker = SitOnLowSpWorker(
            ctx, input_backend, tport,
            danger=danger, vitals=player_vitals,
        )
        workers.append(("sit_sp", sit_worker.run))
    if ctx.config.open_storage_steps:
        storage_worker = ItemsToStorageWorker(
            ctx, input_backend, tport, vitals=player_vitals,
        )
        workers.append(("storage", storage_worker.run))

    return workers


def create_runtime_deps(
    config,
    session_id: str | None = None,
    *,
    behavior_callback: Callable[[str], None] | None = None,
    overlay: HuntOverlay | None = None,
    vitals: PlayerVitals | None = None,
) -> RuntimeDependencies:
    """Construct all hunt runtime dependencies.

    Delegates to focused factory functions so each dependency's
    construction is independently readable and testable.
    """
    logger = _build_logger(config, session_id, behavior_callback)
    detector, tracker = _build_detectors(config)
    ctx = _build_context(config, logger, detector, tracker, overlay)
    # Every runtime start is a fresh hunt cycle. Buffs must complete before
    # normal startup timers, and both must complete before combat begins.
    ctx.begin_hunt_startup()

    input_backend: InputBackend = ViiperBackend()
    player_vitals = vitals or PlayerVitals()

    # Character state is shared by the monitor, danger detector, and every
    # teleport path. Create it before the controller so successful teleports
    # can always clear stale visual threat state in production.
    char_state = CharacterState()
    char_monitor = CharacterStateMonitor(ctx, char_state)

    # Create TeleportController early — every teleport concern lives here.
    tport = TeleportController(
        ctx, input_backend, None, character_state=char_state,
    )
    _validate_teleport_mode(config, tport)

    hunt_mode = create_hunt_mode(ctx, input_backend, teleport_controller=tport)
    _validate_sp_memory(config)
    _validate_weight_memory(config)

    # Build danger before the configurable behavior because safe self-heal
    # decisions share its threat/teleport observations.
    danger = DangerDetector(
        ctx, tport, char_state, vitals=player_vitals,
    )
    ctx.danger_detector = danger
    legacy_behavior = get_mob_behavior(config.mob_name)
    if config.custom_behavior.configured:
        mob_behavior = get_configured_mob_behavior(
            config.custom_behavior,
            player_vitals,
            danger,
            legacy_behavior=legacy_behavior,
        )
    else:
        mob_behavior = legacy_behavior

    core_workers, danger = _build_core_workers(
        ctx, hunt_mode, input_backend, tport, player_vitals, mob_behavior,
        character_state=char_state, danger=danger,
    )
    core_workers.append(("charstate", char_monitor.run))

    conditional_workers = _build_conditional_workers(
        ctx, input_backend, tport, player_vitals,
        character_state=char_state, danger=danger,
    )

    # The controller was constructed with the shared CharacterState; only its
    # hunt-mode callback is resolved after create_hunt_mode().
    tport._hunt_mode = hunt_mode

    return RuntimeDependencies(
        ctx=ctx,
        input_backend=input_backend,
        hunt_mode=hunt_mode,
        logger=logger,
        teleport_controller=tport,
        workers=core_workers + conditional_workers,
    )


class HuntRuntime:
    """Hunt runtime - owns the worker threads and control loop.
    All dependencies (context, backends, workers) are injected
    via RuntimeDependencies, not constructed inline.
    Use create_runtime_deps() to build them.
    """
    def __init__(self, deps: RuntimeDependencies) -> None:
        self._ctx = deps.ctx
        self._workers = deps.workers
        self._input_backend = deps.input_backend
        self._teleport = deps.teleport_controller
        self._worker_threads: list[threading.Thread] = []


    def stop(self) -> None:
        # Wake workers blocked on pause/sit gates so they observe stop_event.
        self._ctx.stop_event.set()
        self._ctx.discovery_wake.set()
        self._ctx.resume_gate.set()

    def pause(self) -> None:
        self._ctx.mark_paused()
        self._ctx.logger.behavior("[PYBOT] paused")

    def resume(self) -> None:
        self._ctx.mark_running()
        self._ctx.discovery_wake.set()
        self._ctx.logger.behavior("[PYBOT] resumed")

    def set_search_range_cells(self, cells: int) -> None:
        self._ctx.capture.set_search_range_cells(cells)

    def _shutdown_workers(self) -> None:
        deadline = time.monotonic() + WORKER_SHUTDOWN_TIMEOUT_S
        pending = [thread for thread in self._worker_threads if thread.is_alive()]
        while pending and time.monotonic() < deadline:
            for thread in pending:
                thread.join(timeout=0.05)
            pending = [thread for thread in pending if thread.is_alive()]
        if pending:
            names = ", ".join(thread.name for thread in pending)
            self._ctx.logger.behavior(
                f"[PYBOT] shutdown timeout; workers still alive: {names}"
            )
        self._worker_threads.clear()
        self._input_backend.shutdown()

    def run(self, *, run_seconds: float = 0.0, start_paused: bool = False) -> int:
        ctx = self._ctx
        alive = [t.name for t in self._worker_threads if t.is_alive()]
        if alive:
            raise RuntimeError(
                "HuntRuntime.run refused: workers still alive "
                f"({', '.join(alive)}) — only one worker set may run"
            )
        if start_paused:
            ctx.mark_paused()
        else:
            ctx.mark_running()

        def _handle_stop(signum: int, _frame: object) -> None:
            ctx.logger.behavior(f"[PYBOT] stop signal={signum}")
            ctx.stop_event.set()
            ctx.discovery_wake.set()
            ctx.resume_gate.set()

        # Signal handlers only work in the main thread; when running inside
        # BotController's daemon thread they raise ValueError on Windows.
        # Wrap gracefully so the hunt runtime still works either way.
        try:
            signal.signal(signal.SIGINT, _handle_stop)
        except (ValueError, OSError):
            pass
        if hasattr(signal, "SIGTERM"):
            try:
                signal.signal(signal.SIGTERM, _handle_stop)
            except (ValueError, OSError):
                pass

        roi = ctx.capture.get_hunt_roi()
        roi_text = f"{roi.x},{roi.y} {roi.w}x{roi.h}" if roi else "unavailable"
        reset_capture_session()
        ctx.logger.behavior(
            f"[PYBOT] hunt runtime start mob={ctx.config.mob_name} hwnd={ctx.config.hwnd} "
            f"mode={ctx.config.hunt_mode} roi={roi_text}"
        )
        teleport_button = (
            self._teleport.active_button() if self._teleport is not None else ""
        )
        ctx.logger.behavior(
            f"[MODE] active={ctx.config.hunt_mode} "
            f"skill={ctx.config.skill_button} "
            f"teleport={teleport_button!r}"
        )

        threads = [
            threading.Thread(target=fn, name=name, daemon=True)
            for name, fn in self._workers
        ]
        self._worker_threads = threads

        for thread in threads:
            thread.start()

        ctx.discovery_wake.set()

        deadline = time.monotonic() + run_seconds if run_seconds > 0 else 0.0
        try:
            while not ctx.is_stopped():
                self._poll_control()
                if deadline and time.monotonic() >= deadline:
                    ctx.stop_event.set()
                    break
                ctx.stop_event.wait(0.25)
        finally:
            ctx.logger.behavior("[PYBOT] hunt runtime stopped")
            self._shutdown_workers()
            reset_capture_session()

        return 0

    def _poll_control(self) -> None:
        command = self._ctx.control.poll()
        if command == "stop":
            self._ctx.stop_event.set()
        elif command == "pause":
            self._ctx.mark_paused()
            self._ctx.logger.behavior("[PYBOT] paused")
        elif command == "resume":
            self._ctx.mark_running()
            self._ctx.discovery_wake.set()
            self._ctx.logger.behavior("[PYBOT] resumed")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.command in {"stop", "pause", "resume"}:
        return write_control_command(args.command, args.control_file)

    config = load_runtime_config(
        hwnd=args.hwnd,
        mob_name=args.mob or None,
        config_path=Path(args.config) if args.config else None,
        control_file=Path(args.control_file) if args.control_file else None,
        session_id=args.session_id or time.strftime("%Y%m%d_%H%M%S"),
    )
    deps = create_runtime_deps(config, session_id=args.session_id)
    runtime = HuntRuntime(deps)
    return runtime.run(
        run_seconds=args.run_seconds,
        start_paused=args.start_paused,
    )


if __name__ == "__main__":
    sys.exit(main())
