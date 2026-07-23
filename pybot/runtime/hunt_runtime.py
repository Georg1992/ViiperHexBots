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
from pybot.runtime.workers.death_detection_worker import DeathDetectionWorker
from pybot.runtime.workers.discovery_worker import DiscoveryWorker
from pybot.runtime.workers.skill_timer_worker import SkillTimerWorker
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


def create_runtime_deps(
    config,
    session_id: str | None = None,
    *,
    behavior_callback: Callable[[str], None] | None = None,
    overlay: HuntOverlay | None = None,
) -> RuntimeDependencies:
    """Construct all hunt runtime dependencies.
    Builds every component the runtime needs (tracks, capture, detector,
    validation logger, input backend, hunt mode controller, etc.) and
    returns them packaged in a RuntimeDependencies container.
    Args:
        config: A HuntRuntimeConfig instance.
        session_id: Optional session identifier (auto-generated if omitted).
        behavior_callback: Optional callback for behavior log messages.
    """
    sid = session_id or time.strftime("%Y%m%d_%H%M%S")
    logger = HuntLogger(
        session_id=sid,
        echo_stdout=behavior_callback is None,
    )
    if behavior_callback:
        logger.set_behavior_callback(behavior_callback)
    detector_config = load_detector_config()
    tracks = HuntTracks(detector_config, skill_delay_ms=config.skill_delay_ms)
    policy = HuntPolicy()
    capture = HuntWindowCapture(config)
    # Two independent detectors: discovery's full scan and tracking's local
    # follow run on separate threads and must never contend on one detector lock.
    detector = DetectorSession(
        config.mob_name,
        detector_config=detector_config,
    )
    tracker = DetectorSession(
        config.mob_name,
        detector_config=detector_config,
    )
    validation = HuntValidationLogger(
        logger,
        tracks,
        enabled=config.validation_enabled,
    )
    control = RuntimeControl(config.control_file)
    ctx = HuntRuntimeContext(
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
    input_backend: InputBackend = (
        ViiperBackend()
    )
    hunt_mode = create_hunt_mode(ctx, input_backend)
    tracking = CoordTrackingWorker(ctx)
    death = DeathDetectionWorker(ctx)
    discovery = DiscoveryWorker(ctx, hunt_mode)
    attack = AttackLoop(ctx, hunt_mode, input_backend)
    workers: list[tuple[str, Callable[[], None]]] = [
        ("coord", tracking.run),
        ("death", death.run),
        ("discovery", discovery.run),
        ("attack", attack.run),
    ]
    if any(t.scan_code and t.interval_ms > 0 for t in ctx.config.skill_timers):
        skill_timer = SkillTimerWorker(ctx, input_backend)
        workers.append(("skill_timer", skill_timer.run))
    if ctx.config.hp_scan_code > 0:
        workers.append(
            ("hp_restore", HpRestoreWorker(ctx, input_backend).run)
        )
    if (
        ctx.config.take_fly_wings
        and ctx.config.open_storage_steps
        and ctx.config.creamy_tp_scan_code <= 0
    ):
        raise ValueError(
            "Take Fly Wings is On but Creamy TP Key is unset. "
            "Creamy TP is required when storage has no more fly wings."
        )
    if not ctx.config.take_fly_wings and ctx.config.creamy_tp_scan_code <= 0:
        raise ValueError(
            "Take Fly Wings is Off but Creamy TP Key is unset. "
            "Creamy TP is used for teleport when fly-wing restock is disabled."
        )
    if ctx.config.sit_on_low_sp:
        if ctx.config.sit_on_low_sp_scan_code <= 0:
            raise ValueError(
                "Sit On Low Sp is On but the sit key is invalid "
                f"(button={ctx.config.sit_on_low_sp_button!r})."
            )
        if ctx.active_teleport_scan_code() <= 0:
            raise ValueError(
                "Sit On Low Sp is On but the teleport key is invalid "
                f"(button={ctx.active_teleport_button()!r}). "
                "Teleport is required to clear mobs before sitting."
            )
        profile = load_client_profile(ctx.config.client_profile)
        memory = MemoryAddresses() if profile is None else profile.memory
        has_sp_memory = memory.current_sp > 0 and memory.max_sp > 0
        # Server profiles need SP memory addresses. Generic uses empty addresses
        # so GameMemoryPoller fills the same MemorySnapshot fields via vision.
        if not has_sp_memory and memory_reading_enabled(ctx.config.client_profile):
            raise ValueError(
                "Sit On Low Sp requires a client profile with currentSpAddress "
                f"and maxSpAddress (profile={ctx.config.client_profile!r})."
            )
        sit_worker = SitOnLowSpWorker(
            ctx, input_backend, memory, hunt_mode=hunt_mode
        )
        workers.append(("sit_sp", sit_worker.run))
    # Storage deposit + GetFlyWings only when Open Storage keychain is assigned.
    if ctx.config.open_storage_steps:
        profile = load_client_profile(ctx.config.client_profile)
        memory = MemoryAddresses() if profile is None else profile.memory
        if ctx.config.weight_modifier >= STORAGE_WEIGHT_MODIFIER_MIN:
            has_weight_memory = memory.current_weight > 0 and memory.max_weight > 0
            if not has_weight_memory and memory_reading_enabled(
                ctx.config.client_profile
            ):
                raise ValueError(
                    "Open Storage requires a client profile with currentWeightAddress "
                    f"and totalWeightAddress (profile={ctx.config.client_profile!r})."
                )
        storage_worker = ItemsToStorageWorker(
            ctx, input_backend, memory, hunt_mode=hunt_mode
        )
        workers.append(("storage", storage_worker.run))

    return RuntimeDependencies(
        ctx=ctx,
        input_backend=input_backend,
        hunt_mode=hunt_mode,
        logger=logger,
        workers=workers,
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
        ctx.logger.behavior(
            f"[MODE] active={ctx.config.hunt_mode} "
            f"skill={ctx.config.skill_button} "
            f"teleport={ctx.active_teleport_button()!r}"
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
