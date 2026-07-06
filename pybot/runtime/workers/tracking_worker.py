"""Local coordinate tracking for all active discovered tracks."""

from __future__ import annotations

from pybot.runtime._mob_rec_path import import_hunt_track_rules
_hunt = import_hunt_track_rules()
LocalTrackObservation = _hunt.LocalTrackObservation
is_attackable = _hunt.is_attackable
is_pending = _hunt.is_pending
from pybot.runtime.hunt_tracks import monotonic_ms
from pybot.runtime.workers.worker_contexts import TrackingWorkerContext
from pybot.runtime.detection.detector_session import StateTrackSnapshot
from pybot.runtime import overlay as hunt_overlay


class TrackingWorker:
    def __init__(self, ctx: TrackingWorkerContext) -> None:
        self._ctx = ctx

    def run(self) -> None:
        self._ctx.logger.behavior("[TRACK] worker started")
        while not self._ctx.stop_event.wait(self._ctx.config.state_interval_ms / 1000.0):
            if not self._ctx.should_run_workers():
                continue
            self._run_local_track_batch()

    def _run_local_track_batch(self) -> None:
        ctx = self._ctx

        # Push search region border even when no tracks exist yet
        if ctx.capture.is_valid():
            roi = ctx.capture.get_hunt_roi()
            if roi is not None:
                hunt_overlay.set_search_roi(roi.x, roi.y, roi.w, roi.h)

        if not ctx.tracks.has_known_targets():
            return
        if not ctx.capture.is_valid():
            return

        roi = ctx.capture.get_hunt_roi()
        if roi is None:
            return

        now = monotonic_ms()
        requests = ctx.tracks.collect_local_track_requests(now_tick=now)
        if not requests:
            return

        old_snaps = {
            int(req["id"]): ctx.tracks.snapshot_for_track(int(req["id"]), now) for req in requests
        }
        snapshots = [
            StateTrackSnapshot(
                track_id=int(req["id"]),
                x=int(req["x"]),
                y=int(req["y"]),
                scale=float(req.get("scale", 0.0) or 0.0),
            )
            for req in requests
        ]

        try:
            batch = ctx.detector.track_locals(roi, snapshots)
        except Exception as exc:
            ctx.logger.behavior(f"[TRACK] batch failed reason={exc}")
            return

        observations = [
            LocalTrackObservation(
                id=result.track_id,
                found=result.found,
                x=result.x,
                y=result.y,
                confidence=result.confidence,
                miss_reason=result.miss_reason,
            )
            for result in batch.results
        ]
        needs_confirm = ctx.tracks.apply_local_track_observations(observations, now_tick=now)
        for result in batch.results:
            ctx.validation.log_local_track_observation(
                track_id=result.track_id,
                found=result.found,
                old_snap=old_snaps.get(result.track_id),
                x=result.x,
                y=result.y,
                miss_reason=result.miss_reason,
            )
        ctx.validation.log_track_batch(
            track_ids=[result.track_id for result in batch.results],
            found_count=batch.found_count,
            duration_ms=batch.duration_ms,
            needs_confirm_ids=needs_confirm,
            every_n=ctx.config.validation_state_every_n,
        )
        detail = ",".join(
            f"{result.track_id}:{'ok' if result.found else result.miss_reason}"
            for result in batch.results
        )
        ctx.logger.behavior(
            "[TRACK] batch "
            f"tracks={len(batch.results)} "
            f"found={batch.found_count} "
            f"coordUpdates={batch.coord_updates} "
            f"durationMs={batch.duration_ms} "
            f"confirmQueue={len(needs_confirm)} "
            f"results={detail}"
        )
        # Update overlay with current track stats and positions
        now_tick = monotonic_ms()
        hunt_overlay.set_track_stats(
            track_count=ctx.tracks.get_track_count(),
            alive_count=ctx.tracks.get_alive_or_pending_count(now_tick),
            attackable_count=ctx.tracks.get_attackable_count(now_tick),
        )
        tracks = ctx.tracks.tracks_for_policy(now_tick)
        positions = [
            (
                t.x,
                t.y,
                "attackable" if is_attackable(t, now_tick)
                else "pending" if is_pending(t, now_tick)
                else "alive",
            )
            for t in tracks
        ]
        hunt_overlay.set_track_positions(positions)
