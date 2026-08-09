"""Coordinate tracking loop — own thread, follows positions and creates tracks.

Runs as fast as capture + local follow allow. Each tick:
1. Ingests discovery candidates, runs local-follow on the *current fresh frame*
   to get exact coordinates, and creates tracks at those coordinates.
2. Follows every alive track with the LocalTracker, writing fresh coordinates
   into the shared HuntTracks store.

Tracking owns track creation and all position writes. Discovery only
publishes candidate positions; tracking resolves them on a current frame
so tracks are created at the EXACT mob position, not 0.5s ago.

On local miss, wakes discovery so it can confirm the mob via its full
detection pipeline. Local follow still uses ``score_at`` (silhouette gate)
to accept a hit; heatmap peaks only propose candidates.
"""

from __future__ import annotations

import threading
import traceback
from contextlib import nullcontext

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
    """Single-threaded fast loop that follows known tracks and creates new ones."""

    def __init__(self, ctx: CoordTrackingWorkerContext) -> None:
        self._ctx = ctx
        self._last_empty_frame_log_ms = 0
        self._last_slow_track_log_ms = 0
        # Track IDs whose first-tick data has been logged (one shot per track).
        self._logged_first_tick: set[tuple[int, int]] = set()
        self._logged_first_tick_epoch: int | None = None
        # Coordinator-owned lifecycle state: Track objects remain passive, and
        # at most one local-follow job is active for any Track at a time.
        self._inflight_track_ids: set[int] = set()
        self._tracking_round_robin = 0
        self._inflight_lock = threading.Lock()

    def run(self) -> None:
        ctx = self._ctx
        ctx.logger.behavior("[COORD] worker started")
        while not ctx.stop_event.is_set():
            try:
                if ctx.should_run_tracking():
                    self._tick()
                    # Yield the capture session so discovery and
                    # character-state sampling get predictable turns.
                    ctx.stop_event.wait(TRACKING_LOOP_INTERVAL_S)
                    if ctx.stop_event.is_set():
                        break
                elif not ctx.should_run_workers():
                    ctx.wait_while_stopped_or_paused(WORKER_POLL_INTERVAL_S)
                else:
                    # Session gates suspend observation. Retry after a short
                    # stop-aware yield; do not wait on resume_gate, which
                    # belongs only to action workers.
                    ctx.stop_event.wait(WORKER_POLL_INTERVAL_S)
            except Exception:
                ctx.logger.behavior(f"[COORD] tick error:\n{traceback.format_exc()}")
                if ctx.stop_event.wait(0.25):
                    break

    def _tick(self) -> None:
        ctx = self._ctx
        # ``run()`` checks this lifecycle gate. Sit/storage/heal transitions
        # stop observation; epoch and publication checks below reject stale
        # state before publication.
        if not ctx.should_run_tracking():
            return
        if not ctx.capture.is_valid():
            return
        roi = ctx.capture.get_hunt_roi()
        if roi is None:
            return

        now_ms = monotonic_ms()
        area_epoch, alive_tracks = ctx.tracks.tracking_frame_snapshot(now_ms)
        if self._logged_first_tick_epoch != area_epoch:
            with self._inflight_lock:
                self._inflight_track_ids.clear()
        # Track IDs are intentionally monotonic across areas. Keep the
        # one-shot diagnostic set scoped to the current area so long teleport
        # sessions do not retain one tuple per historical track forever.
        if self._logged_first_tick_epoch != area_epoch:
            self._logged_first_tick.clear()
            self._logged_first_tick_epoch = area_epoch

        # Check for new discovery candidates — tracking creates these tracks
        # on the current fresh frame at exact coordinates.
        candidates = ctx.tracks.get_and_clear_new_candidates()

        if not ctx.should_run_tracking():
            ctx.tracks.requeue_discovery_candidates(
                candidates, expected_epoch=area_epoch,
            )
            return

        # Skip if nothing to do
        if not alive_tracks and not candidates:
            self._update_overlay(now_ms)
            return

        frame = ctx.capture.capture_roi(roi, observer="tracking")
        if frame is None or frame.size == 0:
            if candidates:
                ctx.tracks.requeue_discovery_candidates(
                    candidates, expected_epoch=area_epoch,
                )
            if now_ms - self._last_empty_frame_log_ms >= LOG_REPEAT_INTERVAL_MS:
                self._last_empty_frame_log_ms = now_ms
                ctx.logger.behavior("[COORD] capture returned empty frame")
            return

        publication_lock = getattr(ctx, "observation_publication_lock", None)

        # ── Step 1: Create tracks from discovery candidates ──────────────
        if candidates and not ctx.should_run_tracking():
            ctx.tracks.requeue_discovery_candidates(
                candidates, expected_epoch=area_epoch,
            )
            return
        if candidates:
            new_count = self._process_discovery_candidates(
                candidates, frame, roi, now_ms, area_epoch,
            )
            # Candidate resolution is complete; new tracks enter the normal
            # warm-template path on the next frame.

        # ── Step 2: Follow existing tracks ───────────────────────────────
        if not alive_tracks:
            self._update_overlay(now_ms)
            return

        # Only tracks with a valid scale enter local tracking. Consume a
        # reanchor only for those submitted tracks; otherwise a malformed
        # track could lose its hint without ever giving Tracking a chance to
        # use it.
        trackable_tracks = [
            track for track in alive_tracks if track.discovery_scale > 0
        ]
        for track in trackable_tracks:
            anchor = ctx.tracks.consume_reanchor(
                track.id,
                expected_epoch=area_epoch,
            )
            if anchor is not None:
                track.pending_reanchor = anchor

        snapshots = [
            StateTrackSnapshot(
                track_id=track.id,
                x=track.x,
                y=track.y,
                scale=track.discovery_scale,
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
                # A track that has not been observed recently cannot safely
                # extrapolate its last frame displacement across the whole
                # stalled interval. Let local tracking reacquire around the
                # last confirmed position instead of biasing the search with
                # stale momentum.
                prediction_valid=(
                    now_ms - track.updated_tick
                    <= max(250, 3 * int(TRACKING_LOOP_INTERVAL_S * 1000))
                ),
                reanchor_x=(
                    track.pending_reanchor[0]
                    if track.pending_reanchor is not None
                    else None
                ),
                reanchor_y=(
                    track.pending_reanchor[1]
                    if track.pending_reanchor is not None
                    else None
                ),
            )
            for track in trackable_tracks
        ]
        if not snapshots:
            self._update_overlay(now_ms)
            return

        # Publish each completed track immediately. The executor may finish a
        # fast mob while another local search is still running; waiting for the
        # whole batch before applying results would preserve the old head-of-
        # line blocking even though computation is parallel.
        published_missed_ids: list[int] = []
        published_opacity_deaths = []
        callback_seen_ids: set[int] = set()
        publish_lock = threading.Lock()

        def publish_result(result) -> None:
            # Fake/legacy tracker sessions used by lightweight callers may not
            # support the completion callback. Mark callback delivery before
            # lifecycle checks so a stale completion is never re-applied by the
            # compatibility fallback below.
            with publish_lock:
                callback_seen_ids.add(result.track_id)
            if not ctx.should_run_tracking() or ctx.tracks.area_epoch != area_epoch:
                return
            # Completion can happen several scheduler ticks after the frame
            # was captured. Store the publication time, not the old tick-start
            # time, so Gameplay sees the real age of each coordinate.
            completed_ms = monotonic_ms()
            commit_guard = (
                nullcontext()
                if publication_lock is None
                else publication_lock
            )
            with commit_guard:
                if (
                    not ctx.should_run_tracking()
                    or ctx.tracks.area_epoch != area_epoch
                ):
                    return
                missed, deaths = ctx.tracks.apply_tracking(
                    [result],
                    now_tick=completed_ms,
                    area_epoch=area_epoch,
                )
            with publish_lock:
                published_missed_ids.extend(missed)
                published_opacity_deaths.extend(deaths)
            self._log_opacity_deaths(deaths)

        submit_async = getattr(ctx.tracker, "submit_track_locals_frame", None)
        if callable(submit_async) and getattr(
            ctx.tracker, "supports_async_tracking", False
        ) is True:
            # Rotate the submission order so a dense scene does not permanently
            # privilege the first four track IDs when the executor is saturated.
            offset = self._tracking_round_robin % len(snapshots)
            ordered = snapshots[offset:] + snapshots[:offset]
            self._tracking_round_robin += 1
            with self._inflight_lock:
                selected = [
                    snapshot for snapshot in ordered
                    if snapshot.track_id not in self._inflight_track_ids
                ]
                selected_ids = {snapshot.track_id for snapshot in selected}
                self._inflight_track_ids.update(selected_ids)

            def async_publish(result) -> None:
                try:
                    publish_result(result)
                    if (
                        not result.found
                        and not ctx.discovery_suspend.is_set()
                        and ctx.tracks.area_epoch == area_epoch
                    ):
                        ctx.discovery_wake.set()
                finally:
                    with self._inflight_lock:
                        self._inflight_track_ids.discard(result.track_id)

            try:
                accepted = set(
                    submit_async(
                        frame,
                        roi,
                        selected,
                        on_result=async_publish,
                    )
                )
            except BaseException:
                with self._inflight_lock:
                    self._inflight_track_ids.difference_update(selected_ids)
                raise
            with self._inflight_lock:
                self._inflight_track_ids.difference_update(selected_ids - accepted)
            self._update_overlay(now_ms)
            return

        batch = ctx.tracker.track_locals_frame(
            frame,
            roi,
            snapshots,
            on_result=publish_result,
        )
        self._warn_if_slow_tracking(batch, snapshots)
        results = batch.results

        # Data collection: log first tracking tick for newly created tracks
        # to measure actual delay and movement before tracking gets its first
        # chance to follow. Each track is logged at most once.
        for track in alive_tracks:
            if (area_epoch, track.id) in self._logged_first_tick:
                continue
            age_ms = now_ms - track.created_tick
            if age_ms <= 0:
                continue
            self._logged_first_tick.add((area_epoch, track.id))
            age_sec = age_ms / 1000.0
            result = next((r for r in results if r.track_id == track.id), None)
            if result is not None:
                dx = result.x - track.x
                dy = result.y - track.y
                dist = int((dx * dx + dy * dy) ** 0.5)
                miss_reason = getattr(result, "miss_reason", "")
                ctx.logger.behavior(
                    f"[TRACK] first_tick track={track.id} "
                    f"age={age_sec:.2f}s snap=({track.x},{track.y}) "
                    f"found={result.found} at=({result.x},{result.y}) "
                    f"shift={dist}px "
                    f"miss_reason={miss_reason}"
                )

        # A teleport/area reset may win while local matching is computing.
        # Never publish a frame from the old screen into the new area, and do
        # not let an in-flight transition turn a stale hit into liveness.
        # Results were committed individually as each executor job completed.
        # A transition may have invalidated some or all callbacks; the store's
        # epoch check already discarded those stale publications.
        #
        # Compatibility path: simple test/fake sessions return a batch but do
        # not implement ``on_result`` delivery. Apply that complete batch once
        # here; real DetectorSession callbacks have already marked every result
        # and therefore cannot be double-published.
        if not callback_seen_ids:
            commit_guard = (
                nullcontext()
                if publication_lock is None
                else publication_lock
            )
            with commit_guard:
                if (
                    ctx.should_run_tracking()
                    and ctx.tracks.area_epoch == area_epoch
                ):
                    missed, deaths = ctx.tracks.apply_tracking(
                        results,
                        now_tick=now_ms,
                        area_epoch=area_epoch,
                    )
                    published_missed_ids.extend(missed)
                    published_opacity_deaths.extend(deaths)
        missed_ids = list(published_missed_ids)
        opacity_deaths = list(published_opacity_deaths)

        self._log_opacity_deaths(opacity_deaths)

        # Local miss → wake discovery so it can confirm removal.
        if missed_ids and not ctx.discovery_suspend.is_set():
            ctx.discovery_wake.set()

        self._update_overlay(now_ms)

    def _process_discovery_candidates(
        self,
        candidates,
        frame,
        roi,
        now_ms: int,
        area_epoch: int,
    ) -> int:
        """Run local-follow on fresh frame for each candidate; create tracks.

        Returns the number of tracks created.
        """
        ctx = self._ctx
        mob_name = ctx.config.mob_name
        publication_lock = getattr(ctx, "observation_publication_lock", None)

        # Get existing track positions for dedup — don't create a track
        # for a mob that already has one.
        existing_positions = ctx.tracks.positions_snapshot(now_ms)
        config = ctx.tracker.detector_config()
        dedup_radius = int(config["trackDedupRadiusPx"])
        dedup_sq = dedup_radius * dedup_radius

        pending: list = []
        snaps: list[StateTrackSnapshot] = []
        for candidate in candidates:
            cx, cy = candidate.x, candidate.y

            # Dedup: skip if this candidate matches an existing track
            duplicate = False
            for px, py in existing_positions:
                if (cx - px) ** 2 + (cy - py) ** 2 <= dedup_sq:
                    duplicate = True
                    break
            if duplicate:
                continue

            # Fail closed: discovery must supply a positive scale.
            if candidate.candidate_scale <= 0:
                continue

            # Negative IDs are provisional and never collide with real track
            # IDs. Their one-shot template is transferred after create_track.
            provisional_id = -(len(pending) + 1)
            snaps.append(
                StateTrackSnapshot(
                    track_id=provisional_id,
                    x=cx,
                    y=cy,
                    scale=candidate.candidate_scale,
                    now_tick=now_ms,
                )
            )
            pending.append(candidate)

        if not snaps:
            return 0

        batch = ctx.tracker.track_locals_frame(frame, roi, snaps)
        self._warn_if_slow_tracking(batch, snaps)

        if not getattr(batch, "ok", True) or len(batch.results) != len(pending):
            self._discard_provisional_templates(batch.results)
            ctx.tracks.requeue_discovery_candidates(
                pending, expected_epoch=area_epoch,
            )
            return 0

        created = 0
        commit_guard = (
            nullcontext()
            if publication_lock is None
            else publication_lock
        )
        with commit_guard:
            if (
                not ctx.should_run_tracking()
                or ctx.tracks.area_epoch != area_epoch
            ):
                self._discard_provisional_templates(batch.results)
                ctx.tracks.requeue_discovery_candidates(
                    pending, expected_epoch=area_epoch,
                )
                return 0
            for index, (candidate, result) in enumerate(
                zip(pending, batch.results, strict=True)
            ):
                if (
                    not ctx.should_run_tracking()
                    or ctx.tracks.area_epoch != area_epoch
                ):
                    # A transition won while local-follow was running. Do not
                    # create old-area tracks; the next discovery frame owns the
                    # fresh area.
                    self._discard_provisional_templates(batch.results[index:])
                    ctx.tracks.requeue_discovery_candidates(
                        pending[index:], expected_epoch=area_epoch,
                    )
                    self._wake_attack_if_created(created)
                    return created
                if not result.found:
                    ctx.tracker.discard_track_template(result.track_id)
                    continue
                cx, cy = result.x, result.y
                create_x, create_y = cx, cy

                # Re-check dedup after earlier creates in this batch.
                duplicate = False
                for px, py in existing_positions:
                    if (create_x - px) ** 2 + (create_y - py) ** 2 <= dedup_sq:
                        duplicate = True
                        break
                if duplicate:
                    ctx.tracker.discard_track_template(result.track_id)
                    continue

                track = ctx.tracks.create_track(
                    mob_name,
                    create_x,
                    create_y,
                    candidate.confidence,
                    candidate.candidate_scale,
                    now_tick=now_ms,
                    area_epoch=area_epoch,
                )
                if track is None:
                    # Area epoch advanced — do not requeue into the new screen.
                    self._discard_provisional_templates(batch.results[index:])
                    self._wake_attack_if_created(created)
                    return created

                if not ctx.tracker.transfer_track_template(result.track_id, track.id):
                    # Never publish a real track without its warm template.
                    ctx.tracks.remove_track(track.id)
                    self._wake_attack_if_created(created)
                    return created
                existing_positions.append((create_x, create_y))
                created += 1

        if created > 0:
            ctx.logger.behavior(
                f"[COORD] created {created} track(s) from discovery candidates"
            )
        self._wake_attack_if_created(created)
        return created

    def _log_opacity_deaths(self, events) -> None:
        """Log opacity removals consistently for sync and async completions."""
        ctx = self._ctx
        if ctx.config.use_sprite_grf:
            return
        for event in events:
            ratio = (
                event.opacity_score / event.baseline
                if event.baseline > 0
                else 0.0
            )
            ctx.logger.behavior(
                f"[DEATH] path=opacity id={event.track_id} "
                f"@{event.x},{event.y} "
                f"score={event.opacity_score:.3f} baseline={event.baseline:.3f} "
                f"ratio={ratio:.2f} streak={event.streak} "
                f"— track removed, death-site recorded"
            )

    def _wake_attack_if_created(self, created: int) -> None:
        """Publish the attack wake after any partial candidate commit."""
        if created <= 0:
            return
        attack_wake = getattr(self._ctx, "attack_wake", None)
        if attack_wake is not None:
            attack_wake.set()

    def _discard_provisional_templates(self, results) -> None:
        """Drop one-shot templates that will not become live track IDs."""
        for result in results:
            self._ctx.tracker.discard_track_template(result.track_id)

    def _warn_if_slow_tracking(self, batch, snapshots) -> None:
        # getattr defaults keep fake batches (SimpleNamespace in tests) fast.
        duration_ms = getattr(batch, "duration_ms", 0)
        if duration_ms < SLOW_SCAN_WARN_MS:
            return
        now_ms = monotonic_ms()
        if now_ms - self._last_slow_track_log_ms < LOG_REPEAT_INTERVAL_MS:
            return
        self._last_slow_track_log_ms = now_ms
        ctx = self._ctx
        ctx.logger.behavior(
            f"[COORD] SLOW tracking total={duration_ms}ms "
            f"lock_wait={getattr(batch, 'lock_wait_ms', 0)}ms "
            f"compute={getattr(batch, 'compute_ms', 0)}ms "
            f"tracks={len(snapshots)} found={getattr(batch, 'found_count', 0)} "
            f"coord_updates={getattr(batch, 'coord_updates', 0)}"
        )

    def _update_overlay(self, now_ms: int) -> None:
        """Push the freshest stored coords to the overlay every tick.

        Runs right after each tracking pass, so the positions published are
        the latest written under the store lock. The overlay setters dedupe
        identical values (no repaint when nothing changed) and painting is
        coalesced on the UI thread, so publishing at the full tracking
        cadence only costs a lock + list compare per tick — never a stale
        dot up to 100 ms behind the mob.
        """
        ctx = self._ctx
        track_count, alive = ctx.tracks.overlay_track_state(now_ms)
        current_epoch = ctx.tracks.area_epoch
        active_keys = {(current_epoch, track.id) for track in alive}
        self._logged_first_tick.intersection_update(active_keys)
        ctx.overlay.set_track_stats(track_count=track_count, alive_count=len(alive))
        ctx.overlay.set_track_positions([(t.x, t.y) for t in alive])
