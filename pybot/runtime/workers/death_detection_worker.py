"""Death detection loop — own thread, opacity probe only.

Runs sequentially after each coordinate-tracking tick. Waits on
``coord_tick_done``, then captures a frame and probes opacity on every
alive track using the positions freshly updated by the coords worker.
Death is confirmed via opacity fade while stationary; dead tracks are
removed and kill samples are recorded.

This worker is the sole death detector. The coordinate-tracking worker
never flags death.
"""

from __future__ import annotations

import traceback

from pybot.recognition.detector.tracking.opacity_probe import probe_track_death
from pybot.runtime.constants import LOG_REPEAT_INTERVAL_MS, WORKER_POLL_INTERVAL_S
from pybot.runtime.hunt_tracks import monotonic_ms
from pybot.runtime.workers.worker_contexts import DeathDetectionWorkerContext


class DeathDetectionWorker:
    """Opacity-only death detector. Reads positions from coords, probes opacity."""

    def __init__(self, ctx: DeathDetectionWorkerContext) -> None:
        self._ctx = ctx
        self._last_empty_frame_log_ms = 0

    def run(self) -> None:
        ctx = self._ctx
        ctx.logger.behavior("[DEATH] worker started")
        while not ctx.stop_event.is_set():
            try:
                if not ctx.should_run_workers():
                    ctx.wait_while_stopped_or_paused(WORKER_POLL_INTERVAL_S)
                    continue
                # Wait for the coords worker to finish its tick.
                if not ctx.coord_tick_done.wait(timeout=0.5):
                    continue
                ctx.coord_tick_done.clear()
                if ctx.stop_event.is_set():
                    break
                self._tick()
            except Exception:
                ctx.logger.behavior(
                    f"[DEATH] tick error:\n{traceback.format_exc()}"
                )

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
            return

        frame = ctx.capture.capture_roi(roi)
        if frame is None or frame.size == 0:
            if now_ms - self._last_empty_frame_log_ms >= LOG_REPEAT_INTERVAL_MS:
                self._last_empty_frame_log_ms = now_ms
                ctx.logger.behavior("[DEATH] capture returned empty frame")
            return

        descriptor = ctx.tracker.ensure_descriptor()
        config = ctx.tracker.detector_config()
        death_results: list[tuple[int, float, int, int, bool]] = []

        for track in alive_tracks:
            baseline, samples, streak, dead = probe_track_death(
                frame,
                descriptor,
                x=track.x,
                y=track.y,
                scale=track.discovery_scale if track.discovery_scale > 0 else 1.0,
                opacity_baseline=track.opacity_baseline,
                opacity_baseline_samples=track.opacity_baseline_samples,
                opacity_decay_streak=track.opacity_decay_streak,
                config=config,
                moving=track.moving,
                now_tick=now_ms,
            )
            death_results.append((track.id, baseline, samples, streak, dead))

        dead_ids = ctx.tracks.apply_death_results(
            death_results,
            now_tick=now_ms,
            area_epoch=area_epoch,
        )
        if dead_ids:
            ctx.logger.behavior(
                f"[DEATH] confirmed {len(dead_ids)} dead track(s): {dead_ids}"
            )
