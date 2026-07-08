"""Tracking loop — own thread, keeps every track's coordinates fresh.

Runs fast and independently of discovery. Each tick it captures a frame and
follows every alive track with the LocalTracker (via ``ctx.tracker`` — a
detector dedicated to tracking so it never blocks on the discovery scan's
lock), writing fresh coordinates into the shared HuntTracks store. That store
is the hand-off point: discovery reads those coordinates when it dedups.

Tracking owns position + liveness. A track missed for too many consecutive
ticks is dropped here (discovery never removes tracks).
"""

from __future__ import annotations

import traceback

from pybot.runtime.constants import WORKER_POLL_INTERVAL_S
from pybot.runtime.hunt_tracks import monotonic_ms
from pybot.runtime.constants import WORKER_POLL_INTERVAL_S
from pybot.runtime.detection.detector_session import StateTrackSnapshot
from pybot.runtime.workers.worker_contexts import TrackingWorkerContext


class TrackingWorker:
    """Single-threaded fast loop that follows known tracks and expires lost ones."""

    def __init__(self, ctx: TrackingWorkerContext) -> None:
        self._ctx = ctx

    def run(self) -> None:
        ctx = self._ctx
        ctx.logger.behavior("[TRACK] worker started")
        try:
            while not ctx.stop_event.is_set():
                if ctx.should_run_workers():
                    self._tick()
                ctx.stop_event.wait(WORKER_POLL_INTERVAL_S)
        except Exception:
            ctx.logger.behavior(f"[TRACK] CRASH:\n{traceback.format_exc()}")
            raise

    def _tick(self) -> None:
        ctx = self._ctx
        if not ctx.capture.is_valid():
            return
        roi = ctx.capture.get_hunt_roi()
        if roi is None:
            return

        frame = ctx.capture.capture_roi(roi)
        if frame is None or frame.size == 0:
            ctx.logger.behavior("[TRACK] capture returned empty frame")
            return

        now_ms = monotonic_ms()
        snapshots = [
            StateTrackSnapshot(
                track_id=snap.id,
                x=snap.x,
                y=snap.y,
                scale=snap.discovery_scale if snap.discovery_scale > 0 else 1.0,
            )
            for snap in ctx.tracks.snapshot_alive(now_ms)
        ]

        if snapshots:
            batch = ctx.tracker.track_locals_frame(frame, roi, snapshots)
            removed = ctx.tracks.apply_tracking(batch.results, now_tick=now_ms)
            if removed:
                ctx.logger.behavior(
                    f"[TRACK] dropped {len(removed)} lost track(s): {removed}"
                )

        self._update_overlay(roi, now_ms)

    def _update_overlay(self, roi, now_ms: int) -> None:
        ctx = self._ctx
        track_count, alive = ctx.tracks.overlay_track_state(now_ms)
        ctx.overlay.set_track_stats(track_count=track_count, alive_count=len(alive))
        ctx.overlay.set_track_positions([(t.x, t.y) for t in alive])
        ctx.overlay.set_search_roi(roi.x, roi.y, roi.w, roi.h)
