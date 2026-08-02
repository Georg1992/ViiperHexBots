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
        danger = DangerDetector(ctx, vitals=player_vitals)
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
    try:
        detector, tracker = _build_detectors(config)
        ctx = _build_context(config, logger, detector, tracker, overlay)
        has_buffs = any(
            buff.scan_code > 0 and buff.delay_ms > 0
            for buff in config.custom_behavior.buffs
        )
        has_timers = any(
            timer.scan_code and timer.interval_ms > 0
            for timer in config.skill_timers
        )
        # Every runtime start is a fresh hunt cycle. Only configured workers need
        # their startup milestone replayed after a sit/stand generation reset.
        ctx.begin_hunt_startup(
            require_buffs=has_buffs,
            require_timers=has_timers,
        )

        input_backend: InputBackend = ViiperBackend()
        player_vitals = vitals or PlayerVitals()

        # Create TeleportController early — every teleport concern lives here.
        tport = TeleportController(ctx, input_backend, None)
        _validate_teleport_mode(config, tport)

        hunt_mode = create_hunt_mode(ctx, input_backend, teleport_controller=tport)
        _validate_sp_memory(config)
        _validate_weight_memory(config)

        # Build danger before the configurable behavior because safe self-heal
        # decisions share its threat/teleport observations.
        danger = DangerDetector(ctx, vitals=player_vitals)
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
            danger=danger,
        )
        conditional_workers = _build_conditional_workers(
            ctx, input_backend, tport, player_vitals,
            danger=danger,
        )

        # The controller's hunt-mode callback is resolved after create_hunt_mode().
        tport._hunt_mode = hunt_mode

        return RuntimeDependencies(
            ctx=ctx,
            input_backend=input_backend,
            hunt_mode=hunt_mode,
            logger=logger,
            teleport_controller=tport,
            workers=core_workers + conditional_workers,
        )
    except BaseException:
        # The logger owns a live QueueListener as soon as it is constructed.
        # If any later dependency/validation step fails, no HuntRuntime exists
        # to close it, so release it at this ownership boundary. Cleanup must
        # not replace the original construction error.
        try:
            logger.close()
        except BaseException:
            pass
        try:
            reset_capture_session()
        except BaseException:
            pass
        raise





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
        self._logger = deps.logger
        self._worker_threads: list[threading.Thread] = []
        self._shutdown_complete = threading.Event()

    def is_shutdown_complete(self) -> bool:
        """True only after every worker and input operation has shut down."""
        return self._shutdown_complete.is_set()

    def retry_shutdown(self) -> bool:
        """Retry bounded cleanup after the runtime thread has returned."""
        if self._shutdown_complete.is_set():
            return True
        return self._finalize_shutdown()

    def _finalize_shutdown(self) -> bool:
        """Join workers and release session resources at one ownership boundary."""
        clean_shutdown = self._shutdown_workers()
        logger_closed = self._close_logger() if clean_shutdown else False
        if clean_shutdown and logger_closed:
            reset_capture_session()
            self._shutdown_complete.set()
            return True
        if clean_shutdown and not logger_closed:
            self._ctx.logger.behavior(
                "[PYBOT] shutdown incomplete; logger writer is still active"
            )
        return False

    def _close_logger(self) -> bool:
        """Release the async logger owned by this runtime session."""
        close = getattr(self._logger, "close", None)
        if not callable(close):
            return True
        result = close()
        return result is not False

    def _cancel_input(self) -> None:
        """Cancel an in-flight input operation when the backend supports it."""
        backend = getattr(self, "_input_backend", None)
        cancel = getattr(backend, "cancel_pending", None)
        if callable(cancel):
            cancel()

    def _begin_input_session(self) -> bool:
        """Re-arm a reusable input backend for a fresh hunt session."""
        backend = getattr(self, "_input_backend", None)
        begin = getattr(backend, "begin_session", None)
        if not callable(begin):
            return True
        result = begin()
        return result is not False

    def _shutdown_input(self) -> bool:
        """Release input state while retaining backend compatibility."""
        shutdown = getattr(self._input_backend, "shutdown", None)
        if not callable(shutdown):
            return True
        result = shutdown()
        return result is not False

    def _cleanup_failed_startup(self) -> bool:
        """Best-effort cleanup for any exception before the run loop starts."""
        ctx = self._ctx
        try:
            ctx.stop_event.set()
        except BaseException:
            pass
        for event_name in ("discovery_wake", "resume_gate"):
            try:
                getattr(ctx, event_name).set()
            except BaseException:
                pass

        try:
            clean_shutdown = self._shutdown_workers()
        except BaseException:
            clean_shutdown = False

        logger_closed = False
        if clean_shutdown:
            try:
                logger_closed = self._close_logger()
            except BaseException:
                logger_closed = False
        if clean_shutdown and logger_closed:
            try:
                reset_capture_session()
            except BaseException:
                return False
            self._shutdown_complete.set()
            return True
        return False

    def stop(self) -> None:
        # Cancel input first so workers blocked inside a composite key/mouse
        # operation can unwind and observe stop_event without waiting for the
        # full storage/skill delay. The backend keeps shared VIIPER streams
        # alive; shutdown later sends neutral reports.
        self._cancel_input()
        # Wake workers blocked on pause/sit gates so they observe stop_event.
        self._ctx.stop_event.set()
        self._ctx.discovery_wake.set()
        self._ctx.resume_gate.set()

    def pause(self) -> None:
        # Focus loss must interrupt an in-flight VIIPER macro just like Stop.
        # Otherwise a storage drag/key-chain can continue sending input after
        # the game is no longer active and keep the shared operation lock busy.
        self._cancel_input()
        self._ctx.mark_paused()
        self._ctx.logger.behavior("[PYBOT] paused")

    def resume(self) -> bool:
        # A pause deliberately leaves the shared cancel event set. Re-arm it
        # only after the canceled operation has released its lock, so no worker
        # can send input during the resume transition. The backend uses a
        # bounded acquisition; the UI thread must never wait indefinitely.
        if not self._begin_input_session():
            self._ctx.logger.behavior(
                "[PYBOT] resume deferred — input operation is still unwinding"
            )
            return False
        self._ctx.mark_running()
        self._ctx.discovery_wake.set()
        self._ctx.logger.behavior("[PYBOT] resumed")
        return True

    def set_search_range_cells(self, cells: int) -> None:
        self._ctx.capture.set_search_range_cells(cells)

    def _shutdown_workers(self) -> bool:
        # Make cancellation idempotent and repeat it here for runtimes stopped
        # by a control file, signal, or run_seconds deadline rather than stop().
        self._cancel_input()
        self._ctx.discovery_wake.set()
        self._ctx.resume_gate.set()

        deadline = time.monotonic() + WORKER_SHUTDOWN_TIMEOUT_S
        pending = [thread for thread in self._worker_threads if thread.is_alive()]
        while pending and time.monotonic() < deadline:
            for thread in pending:
                thread.join(timeout=0.05)
            pending = [thread for thread in pending if thread.is_alive()]
        if pending:
            names = ", ".join(thread.name for thread in pending)
            self._ctx.logger.behavior(
                f"[PYBOT] shutdown incomplete; workers still alive: {names}"
            )
            # Never discard live worker handles or close shared input/capture
            # resources while they remain active. The owning controller keeps
            # this runtime non-restartable until a later stop retry succeeds.
            return False

        # A sit worker may have exited after three failed cleanup attempts.
        # Keep runtime ownership and input resources until its dedicated
        # shutdown toggle succeeds; otherwise a restart could invert the
        # character's seated state.
        unresolved = getattr(self._ctx, "sit_cleanup_unresolved", None)
        if isinstance(unresolved, threading.Event) and unresolved.is_set():
            retry_cleanup = getattr(self._ctx, "retry_sit_cleanup", None)
            if not callable(retry_cleanup) or retry_cleanup() is not True:
                self._ctx.logger.behavior(
                    "[PYBOT] shutdown incomplete; seated state unresolved"
                )
                return False

        self._worker_threads.clear()
        if not self._shutdown_input():
            self._ctx.logger.behavior(
                "[PYBOT] input shutdown did not acquire the shared operation lock"
            )
            return False
        return True

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

        try:
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

            if not self._begin_input_session():
                ctx.logger.behavior(
                    "[PYBOT] input session could not be re-armed; startup aborted"
                )
                self._cleanup_failed_startup()
                return 1

            threads = [
                threading.Thread(target=fn, name=name, daemon=True)
                for name, fn in self._workers
            ]
            self._worker_threads = threads

            for thread in threads:
                thread.start()
        except BaseException:
            # A failure anywhere before the control loop starts—including
            # capture setup, logging, input re-arm, thread construction, or a
            # later Thread.start()—must release anything already acquired.
            # Cleanup is deliberately best-effort so the original exception is
            # preserved for the caller and the runtime remains non-restartable
            # when ownership could not be fully released.
            self._cleanup_failed_startup()
            raise

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
            if not self._finalize_shutdown():
                ctx.logger.behavior(
                    "[PYBOT] runtime remains owned because shutdown was incomplete"
                )

        return 0

    def _poll_control(self) -> None:
        command = self._ctx.control.poll()
        if command == "stop":
            self._ctx.stop_event.set()
        elif command == "pause":
            self.pause()
        elif command == "resume":
            self.resume()


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
