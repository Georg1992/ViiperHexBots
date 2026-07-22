"""Teleport until discovery sees no living mobs (quiet area).

Used before sit regen and before storage / fly-wing restock so UI work
does not run in combat.
"""

from __future__ import annotations

from typing import Protocol

from pybot.runtime.constants import SIT_IDLE_BEFORE_SIT_S, SIT_SP_POLL_INTERVAL_S
from pybot.runtime.detection.discovery_filter import filter_scan_candidates
from pybot.runtime.input.input_backend import InputBackend


class _ClearAreaContext(Protocol):
    def is_stopped(self) -> bool: ...
    def active_teleport_scan_code(self) -> int: ...
    def active_teleport_button(self) -> str: ...
    def note_teleport_for_wings(self) -> None: ...
    def area_reset(self, reason: str = "area_reset") -> None: ...
    def wait_unless_stopped(self, timeout_s: float) -> bool: ...

    @property
    def stop_event(self): ...

    @property
    def logger(self): ...

    @property
    def capture(self): ...

    @property
    def detector(self): ...

    @property
    def config(self): ...

    @property
    def overlay(self): ...


class HuntModeAreaReset(Protocol):
    def on_area_reset(self) -> None: ...


def scan_living_count(ctx: _ClearAreaContext) -> int | None:
    """Run one discovery scan; return filtered living count or None on failure."""
    if not ctx.capture.is_valid():
        return None
    roi = ctx.capture.get_hunt_roi()
    if roi is None:
        return None
    frame = ctx.capture.capture_roi(roi)
    if frame is None or frame.size == 0:
        return None
    scan = ctx.detector.discover_frame(frame, roi)
    if not scan.ok:
        return None
    filtered = filter_scan_candidates(
        scan.detections, roi, ctx.config.cell_size_px
    )
    return len(filtered)


def reset_tracking_after_teleport(
    ctx: _ClearAreaContext,
    hunt_mode: HuntModeAreaReset,
    reason: str,
    *,
    log_tag: str,
) -> None:
    """Clear tracks/policy/overlay and hunt-mode flags after a clear teleport."""
    ctx.area_reset(reason)
    hunt_mode.on_area_reset()
    ctx.overlay.set_track_stats(track_count=0, alive_count=0)
    ctx.overlay.set_track_positions([])
    ctx.logger.behavior(f"[{log_tag}] tracking reset reason={reason}")


def force_teleport(
    ctx: _ClearAreaContext,
    input_backend: InputBackend,
    hunt_mode: HuntModeAreaReset,
    *,
    log_tag: str,
) -> bool:
    """Always press teleport once and settle.

    Used after sit danger (HP drop / hits): damage can come from mobs that
    discovery does not see, so ``teleport_until_clear`` must not skip the key.
    """
    tp_scan = ctx.active_teleport_scan_code()
    if tp_scan <= 0:
        key_name = ctx.active_teleport_button() or "(unset)"
        ctx.logger.behavior(
            f"[{log_tag}] no teleport key configured ({key_name!r}) — "
            "cannot leave screen"
        )
        return False
    ctx.logger.behavior(
        f"[{log_tag}] force teleport key={ctx.active_teleport_button()!r} "
        "(leave screen after danger)"
    )
    input_backend.teleport_key(tp_scan)
    ctx.note_teleport_for_wings()
    ctx.overlay.increment_teleports()
    delay_s = ctx.config.teleport_duration_ms / 1000.0
    if not ctx.wait_unless_stopped(delay_s):
        return False
    reset_tracking_after_teleport(
        ctx, hunt_mode, f"{log_tag.lower()}_force_teleport", log_tag=log_tag
    )
    return True


def teleport_until_clear(
    ctx: _ClearAreaContext,
    input_backend: InputBackend,
    hunt_mode: HuntModeAreaReset,
    *,
    log_tag: str,
) -> bool:
    """Teleport + settle until a discovery scan finds zero living mobs."""
    tp_scan = ctx.active_teleport_scan_code()
    if tp_scan <= 0:
        key_name = ctx.active_teleport_button() or "(unset)"
        ctx.logger.behavior(
            f"[{log_tag}] no teleport key configured ({key_name!r}) — "
            "cannot clear area"
        )
        return False

    while not ctx.is_stopped():
        living = scan_living_count(ctx)
        if living is None:
            ctx.stop_event.wait(SIT_SP_POLL_INTERVAL_S)
            continue
        if living == 0:
            ctx.logger.behavior(f"[{log_tag}] discovery sees no mobs")
            reset_tracking_after_teleport(
                ctx, hunt_mode, f"{log_tag.lower()}_clear", log_tag=log_tag
            )
            return True

        ctx.logger.behavior(
            f"[{log_tag}] discovery living={living} — teleport before UI"
        )
        input_backend.teleport_key(tp_scan)
        ctx.note_teleport_for_wings()
        ctx.overlay.increment_teleports()
        delay_s = ctx.config.teleport_duration_ms / 1000.0
        if not ctx.wait_unless_stopped(delay_s):
            return False
        reset_tracking_after_teleport(
            ctx, hunt_mode, f"{log_tag.lower()}_teleport", log_tag=log_tag
        )
    return False


def teleport_until_quiet(
    ctx: _ClearAreaContext,
    input_backend: InputBackend,
    hunt_mode: HuntModeAreaReset,
    *,
    log_tag: str,
    idle_s: float = SIT_IDLE_BEFORE_SIT_S,
    force_first: bool = False,
) -> bool:
    """Clear area, idle, then re-scan; repeat if mobs appeared during idle.

    A single clear snapshot is not enough: mobs can walk into ROI (or first
    become detectable) during the post-clear idle before sit/storage UI.

    ``force_first``: always teleport once before the clear loop (sit danger —
    hits from undiscovered mobs).
    """
    if force_first:
        if not force_teleport(
            ctx, input_backend, hunt_mode, log_tag=log_tag
        ):
            return False
    while not ctx.is_stopped():
        if not teleport_until_clear(
            ctx, input_backend, hunt_mode, log_tag=log_tag
        ):
            return False
        ctx.logger.behavior(
            f"[{log_tag}] area clear — idle {idle_s:.0f}s before proceed"
        )
        if not ctx.wait_unless_stopped(idle_s):
            return False
        living = scan_living_count(ctx)
        if living is None:
            ctx.logger.behavior(
                f"[{log_tag}] post-idle scan failed — clear again"
            )
            continue
        if living == 0:
            ctx.logger.behavior(f"[{log_tag}] still clear after idle")
            return True
        ctx.logger.behavior(
            f"[{log_tag}] mobs during idle (living={living}) — clear again"
        )
    return False
