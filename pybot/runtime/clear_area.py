"""Teleport until discovery sees no living mobs (quiet area).

Used before sit regen and before storage / fly-wing restock so UI work
does not run in combat.
"""

from __future__ import annotations

from typing import Protocol

from pybot.runtime.constants import SIT_SP_POLL_INTERVAL_S
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
