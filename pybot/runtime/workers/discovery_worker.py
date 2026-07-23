"""Discovery loop — own thread, finds new mobs and signals track evidence.

Schedule: every ``discovery_interval_ms`` (default 1s), and immediately when
``discovery_wake`` is set after a teleport settle delay. While
``discovery_suspend`` is set (claim → teleport key → delay), or while
storage UI is open (``should_run_discovery`` false), this worker does
not scan — only waits for the post-delay wake / storage end.

One discovery pass (same frame):
1. Living heatmap → silhouette scan for new / matched mobs (living refs only).
2. Known-track peaks: living vs death silhouette; death wins → immediate kill.
3. Reconcile: create / match / mark in-ROI absent / drop outside-ROI only.

Tracking owns authoritative position and opacity / lost / unreachable removal.
Discovery never overwrites authoritative x/y; it does remove when the static
death silhouette beats living on a known track.

Teleport clear requires zero living scan candidates, not merely zero alive
tracks after ghost matching. Capture-time position snapshots keep dedup and
absence in the same spacetime as detections despite concurrent tracking.
"""

from __future__ import annotations

import time
import traceback

from pybot.recognition.rules import DiscoveryDetection
from pybot.runtime.constants import LOG_REPEAT_INTERVAL_MS, WORKER_POLL_INTERVAL_S
from pybot.runtime.hunt_tracks import monotonic_ms
from pybot.runtime.detection.discovery_filter import filter_scan_candidates
from pybot.runtime.workers.worker_contexts import DiscoveryWorkerContext


class DiscoveryWorker:
    """Scans for living mobs, flags discovery deaths, creates/matches tracks."""

    def __init__(self, ctx: DiscoveryWorkerContext, hunt_mode) -> None:
        self._ctx = ctx
        self._hunt_mode = hunt_mode
        self._scan_count = 0
        self._last_empty_frame_log_ms = 0

    def run(self) -> None:
        ctx = self._ctx
        ctx.logger.behavior("[DISCOVERY] worker started")
        interval_s = ctx.config.discovery_interval_ms / 1000.0
        while not ctx.stop_event.is_set():
            try:
                if not ctx.should_run_discovery():
                    # Sit/pause: wait on resume gate. Storage: poll until UI done.
                    if not ctx.should_run_workers():
                        ctx.wait_while_stopped_or_paused(interval_s)
                    else:
                        ctx.stop_event.wait(interval_s)
                    continue
                if ctx.discovery_suspend.is_set():
                    # Teleport in flight: ignore the 1s cadence; wait for wake.
                    if not self._wait_for_discovery_wake(interval_s):
                        continue
                    ctx.discovery_wake.clear()
                    if ctx.discovery_suspend.is_set() or not ctx.should_run_discovery():
                        continue
                    self._scan()
                    continue
                # Normal cadence, or immediate scan when teleport releases wake.
                woke = self._wait_for_discovery_wake(interval_s)
                if woke:
                    ctx.discovery_wake.clear()
                if not ctx.should_run_discovery() or ctx.discovery_suspend.is_set():
                    continue
                self._scan()
            except Exception:
                ctx.logger.behavior(f"[DISCOVERY] tick error:\n{traceback.format_exc()}")

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
        if ctx.stop_event.is_set() or ctx.discovery_suspend.is_set():
            return
        if not ctx.capture.is_valid():
            return
        roi = ctx.capture.get_hunt_roi()
        if roi is None:
            return

        # One atomic sample before capture so detections, dedup, and absence
        # share one time reference while tracking moves live tracks.
        now_ms = monotonic_ms()
        area_epoch, existing_positions, existing_track_positions = (
            ctx.tracks.discovery_frame_snapshot(now_ms)
        )

        frame = ctx.capture.capture_roi(roi)
        if ctx.stop_event.is_set():
            return
        if frame is None or frame.size == 0:
            now_ms = monotonic_ms()
            if now_ms - self._last_empty_frame_log_ms >= LOG_REPEAT_INTERVAL_MS:
                self._last_empty_frame_log_ms = now_ms
                ctx.logger.behavior("[DISCOVERY] capture returned empty frame")
            return

        # Living-only for new peaks; known tracks get alive+dead silhouette checks.
        scan = ctx.detector.discover_frame(
            frame,
            roi,
            known_tracks=existing_track_positions,
        )
        if not scan.ok:
            self._hunt_mode.note_discovery_scan_failed(scan.fail_reason)
            return

        self._scan_count += 1

        death_confirmed = scan.death_confirmed or []
        if death_confirmed:
            flagged = ctx.tracks.note_discovery_deaths(
                death_confirmed,
                area_epoch=area_epoch,
                now_tick=now_ms,
            )
            if flagged:
                ctx.logger.behavior(
                    f"[DISCOVERY] death removed for tracker: {flagged}"
                )

        filtered = filter_scan_candidates(scan.detections, roi, ctx.config.cell_size_px)
        ctx.overlay.set_scan_living(len(filtered))

        detections = [
            DiscoveryDetection(
                x=item.x,
                y=item.y,
                confidence=item.confidence,
                candidate_scale=item.candidate_scale,
                living=True,
            )
            for item in filtered
        ]

        # area_epoch gates create/remove under the tracks lock so a teleport
        # between detect and reconcile cannot spawn or clear into the new area.
        summary = ctx.tracks.reconcile_detections(
            detections,
            mob_name=ctx.config.mob_name,
            now_tick=now_ms,
            existing_positions=existing_positions,
            existing_track_positions=existing_track_positions,
            area_epoch=area_epoch,
            hunt_roi=roi,
        )
        if ctx.tracks.area_epoch != area_epoch or ctx.discovery_suspend.is_set():
            return

        verbose = (
            summary.added_count > 0
            or summary.removed_count > 0
            or bool(death_confirmed)
            or self._scan_count <= 3
            or self._scan_count % 20 == 0
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
                f"deaths={len(death_confirmed)} "
                f"tracks={ctx.tracks.get_track_count()}"
            )

        # Teleport clear requires the scan itself to see no living candidates.
        # Ghost-matched corpse heat (alive_after=0, matched>0) must still block
        # clear — otherwise we teleport, wipe removed_sites, and recreate the
        # corpse as a fresh track.
        self._hunt_mode.note_discovery_scan_completed(
            living_count=len(filtered),
            added_count=summary.added_count,
            area_epoch=area_epoch,
        )
