"""Tracking loop — own thread, keeps every track's coordinates fresh.

Runs fast and independently of discovery. Each tick it captures a frame and
follows every alive track with the LocalTracker (via ``ctx.tracker`` — a
detector dedicated to tracking so it never blocks on the discovery scan's
lock), writing fresh coordinates into the shared HuntTracks store.

Tracking is the sole writer of authoritative position and in-ROI removal.
Discovery publishes soft ``discovery_obs_*`` priors and ``discovery_absent``,
and removes immediately on static death-silhouette match (ghost site). On a
local miss, tracking searches/snaps from the prior and drops on joint absence.
Opacity confirm also removes and ghosts. Otherwise: lost miss-limit or
unreachable.
"""

from __future__ import annotations

import traceback

from pybot.runtime.constants import LOG_REPEAT_INTERVAL_MS, WORKER_POLL_INTERVAL_S
from pybot.runtime.hunt_tracks import monotonic_ms
from pybot.runtime.detection.detector_session import StateTrackSnapshot
from pybot.runtime.workers.worker_contexts import TrackingWorkerContext


class TrackingWorker:
    """Single-threaded fast loop that follows known tracks and expires lost ones."""

    def __init__(self, ctx: TrackingWorkerContext) -> None:
        self._ctx = ctx
        self._last_empty_frame_log_ms = 0

    def run(self) -> None:
        ctx = self._ctx
        ctx.logger.behavior("[TRACK] worker started")
        while not ctx.stop_event.is_set():
            try:
                if ctx.should_run_tracking():
                    self._tick()
                    ctx.stop_event.wait(WORKER_POLL_INTERVAL_S)
                elif not ctx.should_run_workers():
                    ctx.wait_while_stopped_or_paused(WORKER_POLL_INTERVAL_S)
                else:
                    # Storage UI: idle without scanning.
                    ctx.stop_event.wait(WORKER_POLL_INTERVAL_S)
            except Exception:
                ctx.logger.behavior(f"[TRACK] tick error:\n{traceback.format_exc()}")

    def _tick(self) -> None:
        ctx = self._ctx
        if not ctx.capture.is_valid():
            return
        roi = ctx.capture.get_hunt_roi()
        if roi is None:
            return

        frame = ctx.capture.capture_roi(roi)
        if frame is None or frame.size == 0:
            now_ms = monotonic_ms()
            if now_ms - self._last_empty_frame_log_ms >= LOG_REPEAT_INTERVAL_MS:
                self._last_empty_frame_log_ms = now_ms
                ctx.logger.behavior("[TRACK] capture returned empty frame")
            return

        now_ms = monotonic_ms()
        area_epoch, alive_tracks = ctx.tracks.tracking_frame_snapshot(now_ms)
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

        results = []
        if snapshots:
            batch = ctx.tracker.track_locals_frame(frame, roi, snapshots)
            results = batch.results

        dead_ids, lost_ids, unreachable_ids = ctx.tracks.apply_tracking(
            results,
            now_tick=now_ms,
            area_epoch=area_epoch,
        )
        if dead_ids:
            ctx.logger.behavior(
                f"[TRACK] dropped {len(dead_ids)} dead track(s): {dead_ids}"
            )
        if lost_ids:
            ctx.logger.behavior(
                f"[TRACK] dropped {len(lost_ids)} lost track(s): {lost_ids}"
            )
        if unreachable_ids:
            ctx.logger.behavior(
                f"[TRACK] dropped {len(unreachable_ids)} unreachable track(s): "
                f"{unreachable_ids}"
            )

        # Deaths are communicated to discovery via removed_sites (dedup ghosts)
        # so the next cadence scan will not recreate corpses. Do not wake
        # discovery on death — that would immediately re-scan the body.
        # Lost / unreachable still wake so discovery can confirm clear or
        # recreate a still-living mob; hunt-mode teleport also wakes itself.
        if (lost_ids or unreachable_ids) and not ctx.discovery_suspend.is_set():
            ctx.discovery_wake.set()

        self._update_overlay(now_ms)

    def _update_overlay(self, now_ms: int) -> None:
        ctx = self._ctx
        track_count, alive = ctx.tracks.overlay_track_state(now_ms)
        ctx.overlay.set_track_stats(track_count=track_count, alive_count=len(alive))
        ctx.overlay.set_track_positions([(t.x, t.y) for t in alive])
