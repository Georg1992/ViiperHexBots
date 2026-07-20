"""Discovery loop — own thread, finds NEW mobs only.

Runs a full scan every ``discovery_interval_ms`` (or immediately when a teleport
wakes it) and creates tracks for mobs that aren't already tracked. It never
moves or removes existing tracks — tracking (its own thread) owns that.

Correct coord hand-off: the known-object positions used for dedup are sampled
immediately before this worker captures its frame, so detections (from that
frame) and the positions they're compared against share one time reference
even though tracking is concurrently moving the live tracks. Reconcile also
refuses creates when the area epoch advanced since that sample (teleport).
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
    """Single-threaded loop that scans and creates tracks for new mobs only."""

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
                if not ctx.should_run_workers():
                    ctx.wait_while_stopped_or_paused(interval_s)
                    continue
                # Teleport sets discovery_wake for an immediate post-reset scan.
                # Otherwise wait discovery_interval_ms (default 1s) between scans.
                woke = self._wait_for_discovery_wake(interval_s)
                if woke:
                    ctx.discovery_wake.clear()
                if not ctx.should_run_workers():
                    continue
                self._scan()
            except Exception:
                ctx.logger.behavior(f"[DISCOVERY] tick error:\n{traceback.format_exc()}")

    def _wait_for_discovery_wake(self, timeout_s: float) -> bool:
        ctx = self._ctx
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline and not ctx.stop_event.is_set():
            if not ctx.should_run_workers():
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

        # Sample known positions + area epoch immediately before capture so
        # detections and dedup share one time reference while tracking moves
        # live tracks concurrently.
        now_ms = monotonic_ms()
        existing_positions = ctx.tracks.positions_snapshot(now_ms)
        area_epoch = ctx.tracks.area_epoch

        frame = ctx.capture.capture_roi(roi)
        if ctx.stop_event.is_set():
            return
        if frame is None or frame.size == 0:
            now_ms = monotonic_ms()
            if now_ms - self._last_empty_frame_log_ms >= LOG_REPEAT_INTERVAL_MS:
                self._last_empty_frame_log_ms = now_ms
                ctx.logger.behavior("[DISCOVERY] capture returned empty frame")
            return

        scan = ctx.detector.discover_frame(frame, roi)
        if not scan.ok:
            self._hunt_mode.note_discovery_scan_failed(scan.fail_reason)
            return

        self._scan_count += 1
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

        # area_epoch gates create under the tracks lock so a teleport between
        # detect and reconcile cannot spawn pre-reset ghosts into the new area.
        summary = ctx.tracks.reconcile_detections(
            detections,
            mob_name=ctx.config.mob_name,
            now_tick=now_ms,
            existing_positions=existing_positions,
            area_epoch=area_epoch,
        )
        if ctx.tracks.area_epoch != area_epoch:
            return

        verbose = (
            summary.added_count > 0
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
                f"added={summary.added_count} matched={summary.matched_count} "
                f"tracks={ctx.tracks.get_track_count()}"
            )

        self._hunt_mode.note_discovery_scan_completed(
            living_count=len(filtered),
            added_count=summary.added_count,
            area_epoch=area_epoch,
        )
