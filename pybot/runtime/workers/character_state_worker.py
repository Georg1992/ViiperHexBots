"""CharacterStateMonitor — periodic character visual state detection.

Runs in its own thread, captures the hunt ROI and publishes character
position, surrounded state, and generic nearby-mob count into
CharacterState.
"""

from __future__ import annotations

import traceback

from pybot.recognition.nearby_mobs import detect_nearby_any_mobs
from pybot.runtime.character_state import (
    SURROUND_RADIUS_PX,
    CharacterState,
    is_surrounded_by_tracks,
)
from pybot.runtime.constants import HUNT_DISCOVERY_INTERVAL_MS
from pybot.runtime.hunt_tracks import monotonic_ms
from pybot.runtime.workers.worker_contexts import CharacterStateWorkerContext

# How often the monitor captures and analyzes the character ROI.
# Matches the discovery cadence so both run in sync.
CHARACTER_STATE_POLL_INTERVAL_S = HUNT_DISCOVERY_INTERVAL_MS / 1000.0


class CharacterStateMonitor:
    """Captures character ROI, checks surround/nearby mobs, publishes state."""

    def __init__(
        self,
        ctx: CharacterStateWorkerContext,
        character_state: CharacterState,
    ) -> None:
        self._ctx = ctx
        self._state = character_state

    def run(self) -> None:
        ctx = self._ctx
        ctx.logger.behavior("[CHARSTATE] worker started")
        while not ctx.is_stopped():
            try:
                if not ctx.should_run_workers():
                    ctx.wait_while_stopped_or_paused(
                        CHARACTER_STATE_POLL_INTERVAL_S
                    )
                    continue
                self._tick()
            except Exception:
                ctx.logger.behavior(
                    f"[CHARSTATE] tick error:\n{traceback.format_exc()}"
                )

    def _tick(self) -> None:
        ctx = self._ctx

        # Capture the hunt frame — same ROI as discovery/tracking.
        roi = ctx.capture.get_hunt_roi()
        if roi is None:
            return
        frame = ctx.capture.capture_roi(roi)
        if frame is None or frame.size == 0:
            return

        now_ms = monotonic_ms()

        # ── Character position (center of hunt ROI) ───────────────
        char_x = roi.x + roi.w // 2
        char_y = roi.y + roi.h // 2

        # ── Surround detection (tracked mobs) ─────────────────────
        all_mobs = ctx.tracks.positions_snapshot()
        is_surrounded, surround_reason = is_surrounded_by_tracks(
            char_x, char_y, all_mobs,
        )
        nearby_count = 0
        radius_sq = SURROUND_RADIUS_PX * SURROUND_RADIUS_PX
        for mx, my in all_mobs:
            dx = mx - char_x
            dy = my - char_y
            if (dx * dx + dy * dy) <= radius_sq:
                nearby_count += 1

        # ── Generic nearby-mob detection (250x250 center ROI) ─────
        # Detects ANY sprite blobs near the character regardless of type.
        nearby_any = detect_nearby_any_mobs(frame)

        # ── Publish ───────────────────────────────────────────────
        self._state.publish(
            char_x=char_x,
            char_y=char_y,
            is_surrounded=is_surrounded,
            surrounded_reason=surround_reason,
            nearby_mob_count=nearby_count,
            nearby_any_mobs_count=nearby_any,
            tick_ms=now_ms,
        )
