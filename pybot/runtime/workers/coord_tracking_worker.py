"""Coordinate tracking loop — own thread, follows positions only.

Runs as fast as capture + local follow allow. Each tick captures a frame and
follows every alive track with the LocalTracker (skip_opacity=True), writing
fresh coordinates into the shared HuntTracks store.

This worker owns position tracking exclusively. It never removes tracks —
death detection, joint-absence cleanup, and unreachable expiry are
handled by the DeathDetectionWorker.
"""

from __future__ import annotations

import traceback

from pybot.runtime.constants import LOG_REPEAT_INTERVAL_MS, WORKER_POLL_INTERVAL_S
from pybot.runtime.hunt_tracks import monotonic_ms
from pybot.runtime.detection.detector_session import StateTrackSnapshot
from pybot.runtime.workers.worker_contexts import CoordTrackingWorkerContext


class CoordTrackingWorker:
    """Single-threaded fast loop that follows known tracks (coords only)."""

    def __init__(self, ctx: CoordTrackingWorkerContext) -> None:
        self._ctx = ctx
        self._last_empty_frame_log_ms = 0

    def run(self) -> None:
        ctx = self._ctx
        ctx.logger.behavior("[COORD] worker started")
        while not ctx.stop_event.is_set():
            try:
                if ctx.should_run_tracking():
                    self._tick()
                    if ctx.stop_event.is_set():
                        break
                elif not ctx.should_run_workers():
                    ctx.wait_while_stopped_or_paused(WORKER_POLL_INTERVAL_S)
                else:
                    ctx.stop_event.wait(WORKER_POLL_INTERVAL_S)
            except Exception:
                ctx.logger.behavior(f"[COORD] tick error:\n{traceback.format_exc()}")

    def _tick(self) -> None:
        ctx = self._ctx
        if not ctx.capture.is_valid():
            return
        roi = ctx.capture.get_hunt_roi()
        if roi is None:
            return

        now_ms = monotonic_ms()
        area_epoch, alive_tracks = ctx.tracks.tracking_frame_snapshot(now_ms)
        if not alive_tracks:
            self._update_overlay(now_ms)
            return

        frame = ctx.capture.capture_roi(roi)
        if frame is None or frame.size == 0:
            if now_ms - self._last_empty_frame_log_ms >= LOG_REPEAT_INTERVAL_MS:
                self._last_empty_frame_log_ms = now_ms
                ctx.logger.behavior("[COORD] capture returned empty frame")
            return

        snapshots = [
            StateTrackSnapshot(
                track_id=track.id,
                x=track.x,
                y=track.y,
                scale=track.discovery_scale if track.discovery_scale > 0 else 1.0,
                opacity_baseline=track.opacity_baseline,
                opacity_baseline_samples=track.opacity_baseline_samples,
                opacity_decay_streak=track.opacity_decay_streak,
                moving=track.moving,
                vel_x=track.vel_x,
                vel_y=track.vel_y,
                lost_count=track.lost_count,
                attack_count=track.attack_count,
                created_tick=track.created_tick,
                now_tick=now_ms,
                discovery_obs_x=track.discovery_obs_x,
                discovery_obs_y=track.discovery_obs_y,
                discovery_obs_tick=track.discovery_obs_tick,
            )
            for track in alive_tracks
        ]

        batch = ctx.tracker.track_locals_frame(frame, roi, snapshots)
        results = batch.results

        missed_ids = ctx.tracks.apply_tracking(
            results,
            now_tick=now_ms,
            area_epoch=area_epoch,
        )

        # Local miss → wake discovery so it can refresh soft priors.
        if missed_ids and not ctx.discovery_suspend.is_set():
            ctx.discovery_wake.set()

        self._update_overlay(now_ms)

    def _update_overlay(self, now_ms: int) -> None:
        ctx = self._ctx
        track_count, alive = ctx.tracks.overlay_track_state(now_ms)
        ctx.overlay.set_track_stats(track_count=track_count, alive_count=len(alive))
        ctx.overlay.set_track_positions([(t.x, t.y) for t in alive])
