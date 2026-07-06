"""Canonical state confirm — post-attack direct and dead/gone lifecycle."""

from __future__ import annotations

from pybot.runtime._mob_rec_path import import_hunt_track_rules
_hunt = import_hunt_track_rules()
StateObservation = _hunt.StateObservation
from pybot.runtime.hunt_tracks import monotonic_ms
from pybot.runtime.workers.worker_contexts import ConfirmStateWorkerContext
from pybot.runtime.detection.detector_session import StateObservationResult, StateTrackSnapshot
from pybot.runtime import overlay as hunt_overlay


class ConfirmStateWorker:
    def __init__(self, ctx: ConfirmStateWorkerContext) -> None:
        self._ctx = ctx

    def run(self) -> None:
        self._ctx.logger.behavior("[STATE] confirm worker started")
        while not self._ctx.stop_event.wait(self._ctx.config.state_interval_ms / 1000.0):
            if not self._ctx.should_run_workers():
                continue
            if self._run_urgent_direct():
                continue
            self._run_state_confirm()

    def _run_urgent_direct(self) -> bool:
        ctx = self._ctx
        now = monotonic_ms()
        request = ctx.urgent.pop_ready(now)
        if request is None:
            return False
        if not ctx.capture.is_valid():
            ctx.logger.behavior("[STATE] direct skipped reason=invalid_hwnd")
            return True

        roi = ctx.capture.get_hunt_roi()
        if roi is None:
            ctx.logger.behavior("[STATE] direct skipped reason=invalid_roi")
            return True

        old_snap = ctx.tracks.snapshot_for_track(request.track_id, now)
        snapshot = StateTrackSnapshot(
            track_id=request.track_id,
            x=request.x,
            y=request.y,
            scale=self._scale_for_track(request.track_id),
        )
        try:
            batch = ctx.detector.state_direct(roi, snapshot)
        except Exception as exc:
            ctx.logger.behavior(f"[STATE] direct failed id={request.track_id} reason={exc}")
            return True

        self._apply_state_observations(batch.observations, now, old_snaps={request.track_id: old_snap})
        ctx.tracks.clear_local_track_miss(request.track_id)
        ctx.logger.behavior(
            "[STATE] direct "
            f"id={request.track_id} "
            f"durationMs={batch.duration_ms} "
            f"observations={len(batch.observations)} "
            f"coordUpdates={batch.coord_updates}"
        )
        return True

    def _run_state_confirm(self) -> bool:
        ctx = self._ctx
        now = monotonic_ms()
        track_id = ctx.tracks.select_state_confirm_track_id(now)
        if track_id <= 0:
            return False
        if not ctx.capture.is_valid():
            return False

        roi = ctx.capture.get_hunt_roi()
        if roi is None:
            return False

        track = ctx.tracks.get_track_by_id(track_id)
        if track is None:
            return False

        old_snap = ctx.tracks.snapshot_for_track(track_id, now)
        snapshot = StateTrackSnapshot(
            track_id=track_id,
            x=track.x,
            y=track.y,
            scale=self._scale_for_track(track_id),
        )
        try:
            batch = ctx.detector.state_confirm(roi, snapshot)
        except Exception as exc:
            ctx.logger.behavior(f"[STATE] confirm failed id={track_id} reason={exc}")
            ctx.tracks.clear_local_track_miss(track_id)
            return True

        self._apply_state_observations(batch.observations, now, old_snaps={track_id: old_snap})
        ctx.tracks.clear_local_track_miss(track_id)
        obs_detail = ",".join(f"{obs.track_id}:{obs.state}" for obs in batch.observations)
        ctx.logger.behavior(
            "[STATE] confirm "
            f"id={track_id} "
            f"durationMs={batch.duration_ms} "
            f"observations={len(batch.observations)} "
            f"results={obs_detail}"
        )
        return True

    def _scale_for_track(self, track_id: int) -> float:
        track = self._ctx.tracks.get_track_by_id(track_id)
        if track is None:
            return 0.0
        if track.discovery_scale > 0:
            return track.discovery_scale
        if track.candidate_scale > 0:
            return track.candidate_scale
        return 0.0

    def _apply_state_observations(
        self,
        observations: list[StateObservationResult],
        now_tick: int,
        *,
        old_snaps: dict[int, object],
    ) -> None:
        ctx = self._ctx
        mapped = [
            StateObservation(
                id=obs.track_id,
                state=obs.state,  # type: ignore[arg-type]
                x=obs.x,
                y=obs.y,
                confidence=obs.confidence,
            )
            for obs in observations
        ]
        ctx.tracks.apply_state_observations(mapped, now_tick=now_tick)
        for obs in observations:
            ctx.validation.log_state_observation(
                track_id=obs.track_id,
                obs_state=obs.state,
                old_snap=old_snaps.get(obs.track_id),
                obs_x=obs.x,
                obs_y=obs.y,
            )
            if obs.state == "dead":
                hunt_overlay.increment_kills()
