"""Discovery scan worker."""

from __future__ import annotations

import time

from pybot.runtime._mob_rec_path import import_hunt_track_rules
_hunt = import_hunt_track_rules()
DiscoveryDetection = _hunt.DiscoveryDetection
from pybot.runtime.hunt_tracks import monotonic_ms
from pybot.runtime.workers.worker_contexts import DiscoveryWorkerContext
from pybot.runtime.detection.discovery_filter import filter_scan_candidates
from pybot.runtime import overlay as hunt_overlay


class DiscoveryWorker:
    def __init__(self, ctx: DiscoveryWorkerContext, hunt_mode) -> None:
        self._ctx = ctx
        self._hunt_mode = hunt_mode

    def run(self) -> None:
        self._ctx.logger.behavior("[DISCOVERY] worker started")
        self._run_scan()
        poll_s = 0.2  # 200ms — responsive to wake signals without busy-waiting
        interval_s = self._ctx.config.discovery_interval_ms / 1000.0
        elapsed = 0.0
        while True:
            if self._ctx.is_stopped():
                return
            if self._ctx.discovery_wake.is_set():
                self._ctx.discovery_wake.clear()
                if self._ctx.should_run_workers():
                    self._run_scan()
                elapsed = 0.0  # Reset timer after a wake-triggered scan
            if self._ctx.stop_event.wait(poll_s):
                return
            elapsed += poll_s
            if elapsed >= interval_s:
                if self._ctx.should_run_workers():
                    self._run_scan()
                elapsed = 0.0

    def _run_scan(self) -> None:
        ctx = self._ctx
        if not ctx.capture.is_valid():
            self._hunt_mode.note_discovery_scan_failed("invalid_hwnd")
            ctx.logger.behavior("[DISCOVERY] scan skipped reason=invalid_hwnd")
            return

        roi = ctx.capture.get_hunt_roi()
        if roi is None:
            self._hunt_mode.note_discovery_scan_failed("invalid_roi")
            ctx.logger.behavior("[DISCOVERY] scan skipped reason=invalid_roi")
            return

        ctx.tracks.set_roi_center(roi.center_x, roi.center_y)
        start = time.perf_counter()
        try:
            scan = ctx.detector.discover(roi)
        except Exception as exc:
            self._hunt_mode.note_discovery_scan_failed(str(exc))
            ctx.logger.behavior(f"[DISCOVERY] scan failed reason={exc}")
            return

        if not scan.ok:
            self._hunt_mode.note_discovery_scan_failed(scan.fail_reason)
            ctx.logger.behavior(f"[DISCOVERY] scan failed reason={scan.fail_reason}")
            return

        filtered = filter_scan_candidates(scan.detections, roi, ctx.config.cell_size_px)
        detections = [
            DiscoveryDetection(
                x=item.x,
                y=item.y,
                confidence=item.confidence,
                candidate_scale=item.candidate_scale,
                living=item.living,
                dead=item.dead,
            )
            for item in filtered
        ]

        hunt_overlay.set_scan_living(len(filtered))

        summary = ctx.tracks.reconcile_detections(
            detections,
            mob_name=ctx.config.mob_name,
            now_tick=monotonic_ms(),
        )
        self._hunt_mode.note_discovery_scan_completed(
            living_count=len(filtered),
            added_count=summary.added_count,
        )
        duration_ms = int((time.perf_counter() - start) * 1000)
        ctx.validation.log_discovery_scan(
            raw_count=scan.raw_count,
            filtered_count=len(filtered),
            added_count=summary.added_count,
            duration_ms=duration_ms,
            summary=summary,
        )
        ctx.logger.behavior(
            "[DISCOVERY] scan "
            f"raw={scan.raw_count} "
            f"accepted={len(filtered)} "
            f"added={summary.added_count} "
            f"matched={summary.matched_count} "
            f"removed={summary.removed_count} "
            f"durationMs={duration_ms}"
        )
