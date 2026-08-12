"""Discovery loop — own thread, detects living mobs and publishes candidates.

Schedule: every ``discovery_interval_ms`` (default 250ms), and immediately
when ``discovery_wake`` is set after a teleport settle delay or when
tracking wakes it on a local miss. While ``discovery_suspend`` is set (claim →
teleport key → delay), or while a lifecycle session owns the character
(``should_run_discovery`` is false), this worker does not scan — only waits
for the post-delay or session-end wake.

One discovery pass (same frame):
1. Silhouette scan for living mobs (living refs only).
2. Match detections to existing tracks; publish new-mob candidates for
   tracking (which creates tracks on its next fresh frame at exact coords).

        Removal factors run in ``HuntTracks.process_discovery_scan()``:
- Factor 1: Tracks outside the hunt ROI → removed immediately (bookkeeping).
- Factor 2: Tracks missed for 3+ consecutive discovery scans → removed.
  Misses inside the character melee disk (ROI center) do not count while
  local tracking still has the mob (``lost_count == 0``) — player sprite
  occludes discovery silhouette there. Once tracking has also lost it,
  misses count so corpses under the character are removed. If the track
  was already opacity-fading, a death site is recorded so corpse heat
  cannot be rediscovered; otherwise bookkeeping only.
- Earlier misses: ``discovery_miss_count`` increments; track stays alive
  until the remove threshold.

Discovery never creates tracks directly — tracking owns track creation and
all position writes. Discovery only matches detections (resetting
miss_count) and publishes new-candidate positions for tracking to ingest.

Teleport clear requires zero living scan candidates, not merely zero alive
tracks after ghost matching. Capture-time position snapshots keep dedup and
absence in the same spacetime as detections despite concurrent tracking.
"""

from __future__ import annotations

import time
import traceback
from contextlib import nullcontext

from pybot.recognition.rules import DiscoveryDetection
from pybot.runtime.constants import (
    LOG_REPEAT_INTERVAL_MS,
    SLOW_SCAN_WARN_MS,
    WORKER_POLL_INTERVAL_S,
)
from pybot.runtime.hunt_tracks import monotonic_ms
from pybot.runtime.detection.discovery_filter import filter_scan_candidates
from pybot.runtime.diagnostics import (
    format_thread_cpu_deltas,
    format_thread_dump,
    frame_stats,
    game_process_cpu_snapshot,
    sample_threads_while,
)
from pybot.runtime.workers.worker_contexts import DiscoveryWorkerContext


class DiscoveryWorker:
    """Scans for living mobs, creates/matches tracks, marks absent for tracking."""

    def __init__(self, ctx: DiscoveryWorkerContext, hunt_mode) -> None:
        self._ctx = ctx
        self._hunt_mode = hunt_mode
        self._scan_count = 0
        self._last_empty_frame_log_ms = 0
        self._last_slow_scan_log_ms = 0

    def run(self) -> None:
        ctx = self._ctx
        ctx.logger.behavior("[DISCOVERY] worker started")
        interval_s = ctx.config.discovery_interval_ms / 1000.0
        while not ctx.stop_event.is_set():
            try:
                if not ctx.should_run_discovery():
                    # Session gates suspend mob observation while HP/status and
                    # danger feeds continue independently.
                    ctx.stop_event.wait(interval_s)
                    continue
                # Keep the observer on its cadence. _scan() rejects a stale
                # result at the epoch/session publication boundary.
                woke = self._wait_for_discovery_wake(interval_s)
                # Only consume a wake that this wait actually received. If the
                # cadence timed out, a teleport may set the event at the same
                # boundary; clearing it here would lose the required
                # post-teleport discovery wake.
                if woke:
                    ctx.discovery_wake.clear()
                if not ctx.should_run_discovery():
                    continue
                self._scan()
            except Exception:
                ctx.logger.behavior(f"[DISCOVERY] tick error:\n{traceback.format_exc()}")
                if ctx.stop_event.wait(0.25):
                    break

    def _wait_for_discovery_wake(self, timeout_s: float) -> bool:
        ctx = self._ctx
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline and not ctx.stop_event.is_set():
            if not ctx.should_run_discovery():
                return False
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            if ctx.discovery_wake.wait(min(WORKER_POLL_INTERVAL_S, remaining)):
                return True
        return False

    def _scan(self) -> None:
        ctx = self._ctx
        if ctx.stop_event.is_set():
            return
        if not ctx.capture.is_valid():
            return
        roi = ctx.capture.get_hunt_roi()
        if roi is None:
            return

        # Capture the hunt generation before the frame. A sit/stand transition
        # can happen while detection runs; its result must not unlock the next
        # hunt's startup sequence.
        expected_generation = int(getattr(ctx, "hunt_generation", 0))

        # One atomic sample before capture so detections, dedup, and absence
        # share one time reference while tracking moves live tracks.
        now_ms = monotonic_ms()
        area_epoch, existing_positions, existing_track_positions = (
            ctx.tracks.discovery_frame_snapshot(now_ms)
        )

        capture_started_ms = monotonic_ms()
        frame = ctx.capture.capture_roi(roi, observer="discovery")
        capture_ms = monotonic_ms() - capture_started_ms
        if ctx.stop_event.is_set():
            return
        if frame is None or frame.size == 0:
            now_ms = monotonic_ms()
            if capture_ms >= SLOW_SCAN_WARN_MS and (
                now_ms - self._last_empty_frame_log_ms >= LOG_REPEAT_INTERVAL_MS
            ):
                self._last_empty_frame_log_ms = now_ms
                ctx.logger.behavior(
                    "[DISCOVERY] SLOW capture returned empty frame "
                    f"capture={capture_ms}ms"
                )
            elif now_ms - self._last_empty_frame_log_ms >= LOG_REPEAT_INTERVAL_MS:
                self._last_empty_frame_log_ms = now_ms
                ctx.logger.behavior(
                    f"[DISCOVERY] capture returned empty frame capture={capture_ms}ms"
                )
            return

        heat_track_positions: list[tuple[int, ...]] = []
        # Tracking tuning belongs to the detector session, not the runtime
        # behavior config. Keep the fallback for lightweight test doubles and
        # older detector implementations that do not expose detector_config().
        detector_config_getter = getattr(ctx.detector, "detector_config", None)
        detector_config = (
            detector_config_getter() if callable(detector_config_getter) else {}
        )
        if not isinstance(detector_config, dict):
            detector_config = {}
        max_prediction = max(
            1,
            int(detector_config.get("localTrackMaxSearchRadiusPx", 600)),
        )
        for entry in existing_track_positions:
            track_id = int(entry[0])
            current_x = int(entry[1])
            current_y = int(entry[2])
            velocity_x = float(entry[3]) if len(entry) > 3 else 0.0
            velocity_y = float(entry[4]) if len(entry) > 4 else 0.0
            lost_count = max(0, int(entry[5])) if len(entry) > 5 else 0
            scale = float(entry[7]) if len(entry) > 7 else 1.0
            horizon = min(3, lost_count + 1)
            prediction_x = velocity_x * horizon
            prediction_y = velocity_y * horizon
            prediction_length = (prediction_x * prediction_x + prediction_y * prediction_y) ** 0.5
            if prediction_length > max_prediction and prediction_length > 0.0:
                factor = max_prediction / prediction_length
                prediction_x *= factor
                prediction_y *= factor
            heat_track_positions.append(
                (
                    track_id,
                    current_x,
                    current_y,
                    scale,
                    int(round(current_x + prediction_x)),
                    int(round(current_y + prediction_y)),
                )
            )

        # Keep older detector doubles/mocks usable while the production
        # DetectorSession receives optional heat-presence metadata. A TypeError
        # here means only that the alternate implementation has the old method
        # signature; the normal two-argument call remains the safe fallback.
        # While the scan runs, sample all thread stacks + per-thread CPU so a
        # multi-second detect can be attributed with numbers: one bot thread
        # hogging, or every bot thread starved while the game process burns CPU.
        slow_samples = []
        # The game client's process is identified through the captured window.
        window_id = getattr(ctx.capture, "hwnd", None)
        game_cpu_before = game_process_cpu_snapshot(window_id)
        try:
            scan, slow_samples = sample_threads_while(
                lambda: ctx.detector.discover_frame(
                    frame,
                    roi,
                    heat_track_positions=heat_track_positions,
                )
            )
        except TypeError as exc:
            if "heat_track_positions" not in str(exc):
                raise
            scan, slow_samples = sample_threads_while(
                lambda: ctx.detector.discover_frame(frame, roi)
            )
        game_cpu_after = game_process_cpu_snapshot(window_id)
        if game_cpu_before is not None and game_cpu_after is not None:
            game_cpu_ms = int((game_cpu_after[1] - game_cpu_before[1]) * 1000)
            game_wall_ms = max(1.0, scan.duration_ms)
            game_cpu_diag = (
                f"gameCpu={game_cpu_ms}ms "
                f"gameCpuWall={game_cpu_ms / game_wall_ms:.2f} "
                f"gamePid={game_cpu_after[0]} "
            )
        else:
            game_cpu_diag = ""
        if not scan.ok:
            # A failed observation (capture or detection error) must not
            # poison the hunt clear/startup state. Discovery and tracking use
            # independent detector sessions; there is no cross-pipeline skip.
            self._hunt_mode.note_discovery_scan_failed(scan.fail_reason)
            return

        total_ms = capture_ms + scan.duration_ms
        if total_ms >= SLOW_SCAN_WARN_MS and (
            monotonic_ms() - self._last_slow_scan_log_ms >= LOG_REPEAT_INTERVAL_MS
        ):
            self._last_slow_scan_log_ms = monotonic_ms()
            timing = scan.timing
            ctx.logger.behavior(
                f"[DISCOVERY] SLOW scan total={total_ms}ms "
                f"capture={capture_ms}ms lock_wait={scan.lock_wait_ms}ms "
                f"detect={scan.detect_ms}ms "
                f"cpu={int(timing.get('cpuMs', 0.0))}ms "
                f"cpuWall={timing.get('cpuWallRatio', 1.0):.2f} "
                f"threadCpu={int(timing.get('threadCpuMs', 0.0))}ms "
                f"threadCpuWall={timing.get('threadCpuWall', 1.0):.2f} "
                f"{game_cpu_diag}"
                f"frame={int(timing.get('frameW', 0.0))}x"
                f"{int(timing.get('frameH', 0.0))} "
                f"downscale={int(timing.get('downscale', 0.0))} "
                f"raw={scan.raw_count} "
                f"accepted={scan.accepted_count} "
                f"blobCount={int(timing.get('blobCount', 0.0))} "
                f"checks={int(timing.get('silhouetteCheckCount', 0.0))} "
                f"heatmap={int(timing.get('spriteHeatmap', 0.0) * 1000)}ms "
                f"hmResize={int(timing.get('heatmapWorkResize', 0.0) * 1000)}ms "
                f"hmPalette={int(timing.get('heatmapPalettePass', 0.0) * 1000)}ms "
                f"hmFinish={int(timing.get('heatmapFinish', 0.0) * 1000)}ms "
                f"blobs={int(timing.get('blobCenters', 0.0) * 1000)}ms "
                f"palette={int(timing.get('silhouettePaletteHeatmap', 0.0) * 1000)}ms "
                f"gate={int(timing.get('silhouetteGate', 0.0) * 1000)}ms "
                f"gateMax={int(timing.get('maxGate', 0.0) * 1000)}ms "
                f"gateBBox={int(timing.get('maxGateWidth', 0.0))}x"
                f"{int(timing.get('maxGateHeight', 0.0))}"
            )
            ctx.logger.behavior(f"[DISCOVERY] SLOW frame_stats {frame_stats(frame)}")
            if slow_samples:
                lines = [
                    f"  t={elapsed:.2f}s {name}: {info}"
                    for elapsed, thread_snapshot in slow_samples
                    for name, info, _cpu in thread_snapshot
                ]
                cpu_lines = format_thread_cpu_deltas(slow_samples)
                if cpu_lines:
                    lines.append("[DISCOVERY] SLOW per-thread CPU:")
                    lines.extend(cpu_lines)
                ctx.logger.behavior(
                    "[DISCOVERY] SLOW during-scan samples:\n" + "\n".join(lines)
                )
            ctx.logger.behavior(f"[DISCOVERY] SLOW threads:\n{format_thread_dump()}")

        self._scan_count += 1

        filtered = filter_scan_candidates(scan.detections)

        detections = [
            DiscoveryDetection(
                x=item.x,
                y=item.y,
                confidence=item.confidence,
                candidate_scale=item.candidate_scale,
                living=True,
                bbox=item.bbox,
            )
            for item in filtered
        ]

        # area_epoch gates create/remove under the tracks lock so a teleport
        # between detect and reconcile cannot spawn or clear into the new area.
        # process_discovery_scan matches detections, marks absence, handles
        # removal factors, and publishes new candidates for tracking to create
        # on its next fresh frame at exact coordinates.
        # Commit track mutations and all derived state under the same boundary
        # as area reset. This establishes one lock order: transition boundary
        # first, then HuntTracks' internal lock. It prevents a discovery scan
        # from holding track state while reset waits for the transition lock.
        transition_lock = getattr(ctx, "area_transition_lock", None)
        guard = nullcontext() if transition_lock is None else transition_lock
        with guard:
            publication_lock = getattr(
                ctx, "observation_publication_lock", nullcontext()
            )
            with publication_lock:
                # A frame captured during teleport/loading is observation-only.
                # Do not mutate tracks or clear candidates while the area
                # transition or sit session owns the publication boundary.
                if (
                    not ctx.should_run_discovery()
                    or ctx.tracks.area_epoch != area_epoch
                    or expected_generation != int(getattr(ctx, "hunt_generation", 0))
                ):
                    return
                summary = ctx.tracks.process_discovery_scan(
                    detections,
                    mob_name=ctx.config.mob_name,
                    now_tick=now_ms,
                    existing_positions=existing_positions,
                    existing_track_positions=existing_track_positions,
                    area_epoch=area_epoch,
                    hunt_roi=roi,
                    heat_supported_track_ids=getattr(
                        scan, "heat_supported_track_ids", frozenset()
                    ),
                )
                if summary.added_count > 0:
                    # Discovery has published candidates; wake the coordinator
                    # immediately instead of waiting for its 20 ms cadence.
                    tracking_wake = getattr(ctx, "tracking_wake", None)
                    if tracking_wake is not None:
                        tracking_wake.set()

                # A scan that began on the old screen must fail closed before
                # it can publish any observable state.
                if (
                    not ctx.should_run_discovery()
                    or ctx.tracks.area_epoch != area_epoch
                    or expected_generation != int(getattr(ctx, "hunt_generation", 0))
                ):
                    return

            verbose = (
                summary.added_count > 0
                or summary.removed_count > 0
                or self._scan_count <= 3
                or self._scan_count % 20 == 0
                or summary.death_sites_active > 0
            )
            if verbose:
                ctx.validation.log_discovery_scan(
                    raw_count=scan.raw_count,
                    filtered_count=len(filtered),
                    duration_ms=scan.duration_ms,
                    summary=summary,
                )
                ctx.logger.behavior(
                    f"[DISCOVERY] scan#{self._scan_count} "
                    f"raw={scan.raw_count} filtered={len(filtered)} "
                    f"added={summary.added_count} removed={summary.removed_count} "
                    f"matched={summary.matched_count} "
                    f"death_sites={summary.death_sites_active} "
                    f"tracks={ctx.tracks.get_track_count()}"
                )

            if summary.removed_out_of_range_ids:
                ctx.logger.behavior(
                    f"[DISCOVERY] path=out-of-range "
                    f"ids={summary.removed_out_of_range_ids}"
                )
            if summary.removed_discovery_miss_ids:
                ctx.logger.behavior(
                    f"[DISCOVERY] path=miss-2 "
                    f"ids={summary.removed_discovery_miss_ids}"
                )
            # Detections seen but nothing new created while death sites are active —
            # likely corpse heat matched a death site.
            # sprite.grf removes death animations — no corpse heat to block.
            if not ctx.config.use_sprite_grf and (
                len(filtered) > 0
                and summary.added_count == 0
                and summary.alive_after == 0
                and summary.death_sites_active > 0
            ):
                ctx.logger.behavior(
                    f"[DEATH] path=death-site-block "
                    f"detections={len(filtered)} matched={summary.matched_count} "
                    f"death_sites={summary.death_sites_active} "
                    f"— no new track (corpse heat held by death site)"
                )

            # Teleport clear = nothing attackable (no alive tracks, no new candidates).
            # Corpse heat held only by death sites must not block teleport — sites are
            # screen-local and wiped when we leave the area.
            living_for_clear = (
                0
                if summary.alive_after == 0 and summary.added_count == 0
                else max(len(filtered), summary.alive_after)
            )
            self._hunt_mode.note_discovery_scan_completed(
                living_count=living_for_clear,
                added_count=summary.added_count,
                area_epoch=area_epoch,
            )
            # Startup actions are held until at least one successful discovery
            # scan confirms an empty area. An empty track store alone is not proof
            # that the first frame has been checked.
            mark_startup_clear = getattr(ctx, "mark_startup_area_clear", None)
            if callable(mark_startup_clear):
                mark_startup_clear(
                    living_for_clear == 0,
                    expected_generation=expected_generation,
                )

