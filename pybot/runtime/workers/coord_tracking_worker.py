"""Centralized coordinate tracking loop.

Each cycle captures one fresh frame, snapshots all active Tracks, updates them
against that immutable frame, and commits the ordered results. Discovery only
supplies new candidates; it is not part of ordinary Track recovery.
"""

from __future__ import annotations

import time
import traceback

from pybot.runtime.constants import (
    LOG_REPEAT_INTERVAL_MS,
    SLOW_SCAN_WARN_MS,
    TRACKING_LOOP_INTERVAL_S,
    WORKER_POLL_INTERVAL_S,
)
from pybot.runtime.hunt_tracks import monotonic_ms
from pybot.runtime.detection.detector_session import StateTrackSnapshot
from pybot.runtime.workers.worker_contexts import CoordTrackingWorkerContext


class CoordTrackingWorker:
    """Single coordinator for all active local Tracks."""

    def __init__(self, ctx: CoordTrackingWorkerContext) -> None:
        self._ctx = ctx
        self._last_empty_frame_log_ms = 0
        self._last_slow_track_log_ms = 0
        self._logged_first_tick: set[tuple[int, int]] = set()
        self._logged_first_tick_epoch: int | None = None

    def run(self) -> None:
        ctx = self._ctx
        ctx.logger.behavior("[COORD] worker started")
        while not ctx.stop_event.is_set():
            try:
                if ctx.should_run_tracking():
                    self._tick()
                    if ctx.stop_event.is_set():
                        break
                    wake = getattr(ctx, "tracking_wake", None)
                    if wake is not None and callable(getattr(wake, "wait", None)):
                        woke = wake.wait(TRACKING_LOOP_INTERVAL_S)
                        if woke or wake.is_set():
                            wake.clear()
                            continue
                    ctx.stop_event.wait(TRACKING_LOOP_INTERVAL_S)
                elif not ctx.should_run_workers():
                    ctx.wait_while_stopped_or_paused(WORKER_POLL_INTERVAL_S)
                else:
                    ctx.stop_event.wait(WORKER_POLL_INTERVAL_S)
            except Exception:
                ctx.logger.behavior(f"[COORD] tick error:\n{traceback.format_exc()}")
                if ctx.stop_event.wait(0.25):
                    break

    def _tick(self) -> None:
        ctx = self._ctx
        if not ctx.should_run_tracking() or not ctx.capture.is_valid():
            return
        roi = ctx.capture.get_hunt_roi()
        if roi is None:
            return

        now_ms = monotonic_ms()
        area_epoch, alive_tracks = ctx.tracks.tracking_frame_snapshot(now_ms)
        if self._logged_first_tick_epoch != area_epoch:
            self._logged_first_tick.clear()
            self._logged_first_tick_epoch = area_epoch
        candidates = ctx.tracks.get_and_clear_new_candidates()
        if not alive_tracks and not candidates:
            prune = getattr(ctx.tracker, "prune_track_states", None)
            if callable(prune):
                prune({track.id for track in ctx.tracks.snapshot_alive()})
            self._update_overlay(now_ms)
            return

        frame = ctx.capture.capture_roi(roi, observer="tracking")
        if frame is None or frame.size == 0:
            if candidates:
                ctx.tracks.requeue_discovery_candidates(candidates, expected_epoch=area_epoch)
            if now_ms - self._last_empty_frame_log_ms >= LOG_REPEAT_INTERVAL_MS:
                self._last_empty_frame_log_ms = now_ms
                ctx.logger.behavior("[COORD] capture returned empty frame")
            return

        if candidates:
            self._process_discovery_candidates(candidates, frame, roi, now_ms, area_epoch)
            if ctx.tracks.area_epoch != area_epoch:
                self._update_overlay(now_ms)
                return

        _current_epoch, alive_tracks = ctx.tracks.tracking_frame_snapshot(monotonic_ms())
        snapshots = [
            StateTrackSnapshot(
                track_id=track.id,
                x=track.x,
                y=track.y,
                scale=track.discovery_scale,
                vel_x=track.vel_x,
                vel_y=track.vel_y,
                prediction_valid=track.lost_count == 0,
                anchor_required=True,
            )
            for track in alive_tracks
            if track.discovery_scale > 0
        ]
        if snapshots:
            batch = ctx.tracker.track_locals_frame(frame, roi, snapshots)
            if ctx.should_run_tracking() and ctx.tracks.area_epoch == area_epoch:
                completed_ms = monotonic_ms()
                missed, deaths = ctx.tracks.apply_tracking(
                    batch.results,
                    now_tick=completed_ms,
                    area_epoch=area_epoch,
                )
                for event in deaths:
                    ctx.tracker.discard_track_state(event.track_id)
                self._log_opacity_deaths(deaths)
                self._log_first_ticks(alive_tracks, batch.results, area_epoch, now_ms)
            else:
                missed, deaths = [], []
            self._warn_if_slow_tracking(batch, snapshots)
            # A miss is retained locally and will enter the Track's internal
            # recovery ladder on the next fresh frame. Discovery is independent
            # validation, not the recovery mechanism. It may still be notified
            # for independent absence accounting, but never supplies recovery.
            if missed and not ctx.discovery_suspend.is_set():
                ctx.discovery_wake.set()
            del missed
        prune = getattr(ctx.tracker, "prune_track_states", None)
        if callable(prune):
            prune({track.id for track in ctx.tracks.snapshot_alive()})
        self._update_overlay(now_ms)

    def _process_discovery_candidates(
        self,
        candidates,
        frame,
        roi,
        now_ms: int,
        area_epoch: int,
    ) -> int:
        """Acquire candidates on the current frame and commit live Tracks."""
        ctx = self._ctx
        existing = ctx.tracks.positions_snapshot(now_ms)
        config = ctx.tracker.detector_config()
        dedup_radius = int(config["trackDedupRadiusPx"])
        cluster_radius = int(config["discoveryClusterRadiusPx"])
        dedup_sq = dedup_radius * dedup_radius
        cluster_sq = cluster_radius * cluster_radius
        pending = []
        snapshots = []
        for index, candidate in enumerate(candidates):
            if candidate.candidate_scale <= 0:
                continue
            if any((candidate.x - x) ** 2 + (candidate.y - y) ** 2 <= dedup_sq for x, y in existing):
                continue
            provisional_id = -(index + 1)
            pending.append((provisional_id, candidate))
            snapshots.append(StateTrackSnapshot(
                track_id=provisional_id,
                x=candidate.x,
                y=candidate.y,
                scale=candidate.candidate_scale,
                prediction_valid=False,
                anchor_required=False,
            ))
        if not snapshots:
            return 0

        results = ctx.tracker.track_locals_frame(frame, roi, snapshots).results
        committed_positions: list[tuple[int, int]] = []
        committed = 0
        candidate_by_id = dict(pending)
        for result in results:
            candidate = candidate_by_id.get(result.track_id)
            if candidate is None:
                continue
            if not result.found:
                ctx.tracker.discard_track_state(result.track_id)
                continue
            x, y = result.x, result.y
            if any((x - px) ** 2 + (y - py) ** 2 <= dedup_sq for px, py in existing):
                ctx.tracker.discard_track_state(result.track_id)
                continue
            if any((x - px) ** 2 + (y - py) ** 2 <= cluster_sq for px, py in committed_positions):
                ctx.tracker.discard_track_state(result.track_id)
                continue
            if not ctx.should_run_tracking() or ctx.tracks.area_epoch != area_epoch:
                ctx.tracker.discard_track_state(result.track_id)
                continue
            # The coordinate came from the fresh acquisition frame, so stamp
            # the Track at commit time rather than with the pre-capture tick.
            created_ms = monotonic_ms()
            track = ctx.tracks.create_track(
                ctx.config.mob_name,
                x,
                y,
                candidate.confidence,
                candidate.candidate_scale,
                now_tick=created_ms,
                area_epoch=area_epoch,
                discovery_bbox=candidate.bbox,
            )
            if track is None or not ctx.tracker.transfer_track_state(result.track_id, track.id):
                if track is not None:
                    ctx.tracks.remove_track(track.id)
                ctx.tracker.discard_track_state(result.track_id)
                continue
            committed_positions.append((x, y))
            committed += 1

        if committed:
            self._wake_attack_if_created(committed)
            ctx.logger.behavior(f"[COORD] created {committed} track(s) from discovery candidates")
        # Failed acquisition remains eligible for a later Discovery candidate,
        # but a failed local frame itself is not promoted to a Track.
        return committed

    def _log_first_ticks(self, tracks, results, area_epoch: int, now_ms: int) -> None:
        for track in tracks:
            key = (area_epoch, track.id)
            if key in self._logged_first_tick or now_ms <= track.created_tick:
                continue
            self._logged_first_tick.add(key)
            result = next((item for item in results if item.track_id == track.id), None)
            if result is None:
                continue
            distance = int(((result.x - track.x) ** 2 + (result.y - track.y) ** 2) ** 0.5)
            self._ctx.logger.behavior(
                f"[TRACK] first_tick track={track.id} age={(now_ms - track.created_tick) / 1000.0:.2f}s "
                f"found={result.found} at=({result.x},{result.y}) shift={distance}px "
                f"miss_reason={getattr(result, 'miss_reason', '')}"
            )

    def _log_opacity_deaths(self, events) -> None:
        if self._ctx.config.use_sprite_grf:
            return
        for event in events:
            ratio = event.opacity_score / event.baseline if event.baseline > 0 else 0.0
            self._ctx.logger.behavior(
                f"[DEATH] path=opacity id={event.track_id} @{event.x},{event.y} "
                f"score={event.opacity_score:.3f} baseline={event.baseline:.3f} "
                f"ratio={ratio:.2f} streak={event.streak} — track removed, death-site recorded"
            )

    def _wake_attack_if_created(self, created: int) -> None:
        if created > 0:
            wake = getattr(self._ctx, "attack_wake", None)
            if wake is not None:
                wake.set()

    def _warn_if_slow_tracking(self, batch, snapshots) -> None:
        duration_ms = getattr(batch, "duration_ms", 0)
        if duration_ms < SLOW_SCAN_WARN_MS:
            return
        now_ms = monotonic_ms()
        if now_ms - self._last_slow_track_log_ms < LOG_REPEAT_INTERVAL_MS:
            return
        self._last_slow_track_log_ms = now_ms
        self._ctx.logger.behavior(
            f"[COORD] SLOW tracking total={duration_ms}ms tracks={len(snapshots)} "
            f"found={getattr(batch, 'found_count', 0)} coord_updates={getattr(batch, 'coord_updates', 0)}"
        )

    def _update_overlay(self, now_ms: int) -> None:
        del now_ms
        track_count, alive = self._ctx.tracks.overlay_track_state()
        self._ctx.overlay.set_track_stats(track_count=track_count, alive_count=len(alive))
        self._ctx.overlay.set_track_positions([(track.x, track.y) for track in alive])
