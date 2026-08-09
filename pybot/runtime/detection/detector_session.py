"""Detector session — bridges hunt workers to pybot.recognition."""

from __future__ import annotations

import os
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from pybot.paths import PROJECT_ROOT
from pybot.recognition.capture import capture_region
from pybot.recognition.detector.detector import MobDetector, load_detector_config
from pybot.recognition.detector.tracking.local_tracker import (
    LocalTrackResult,
    clear_track_templates,
    discard_track_template,
    transfer_track_template,
)
from pybot.runtime.capture.window_roi import HuntRoi


# Discovery and tracking have separate detector sessions and separate locks.
# They intentionally run in parallel; OpenCV's native worker count is bounded
# once at runtime startup by ``configure_opencv_runtime``.


@dataclass(frozen=True)
class RawDetection:
    x: int
    y: int
    confidence: float
    candidate_scale: float
    living: bool
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)


@dataclass(frozen=True)
class DiscoveryScanResult:
    ok: bool
    fail_reason: str
    raw_count: int
    accepted_count: int
    detections: list[RawDetection]
    duration_ms: int
    elapsed_s: float
    # Stage timing — total duration_ms = lock_wait_ms + detect_ms. Splitting
    # them turns a mystery multi-second scan into lock contention vs compute.
    lock_wait_ms: int = 0
    detect_ms: int = 0
    # Detector-internal stage timings (seconds), retained so a slow live scan
    # can be attributed to heatmap, blobs, or one silhouette candidate.
    timing: dict[str, float] = field(default_factory=dict)



@dataclass(frozen=True)
class StateTrackSnapshot:
    """Track inputs for one local-follow pass (screen coordinates)."""

    track_id: int
    x: int
    y: int
    scale: float = 0.0
    opacity_baseline: float = 0.0
    opacity_baseline_samples: int = 0
    opacity_decay_streak: int = 0
    moving: bool = False
    vel_x: float = 0.0
    vel_y: float = 0.0
    lost_count: int = 0
    attack_count: int = 0
    created_tick: int = 0
    now_tick: int = 0
    prediction_valid: bool = True
    # Discovery's one-shot drift proposal. Tracking consumes it as a search
    # center while authoritative x/y remain the shared track position.
    reanchor_x: int | None = None
    reanchor_y: int | None = None



@dataclass(frozen=True)
class LocalTrackBatchResult:
    ok: bool
    fail_reason: str
    results: list[LocalTrackResult]
    duration_ms: int
    found_count: int
    coord_updates: int
    # Stage timing — total duration_ms = lock_wait_ms + compute_ms.
    lock_wait_ms: int = 0
    compute_ms: int = 0
    # Per-track wall time, keyed by track ID. This is both runtime telemetry
    # and the benchmark boundary for deciding whether one slow mob is starving
    # the rest of the batch.
    track_durations_ms: dict[int, float] = field(default_factory=dict)


class DetectorSession:
    """One MobDetector behind an RLock — no IPC, no scale hard-lock."""

    def __init__(
        self,
        mob_name: str,
        project_root: Path | None = None,
        *,
        detector_config: dict | None = None,
        use_sprite_grf: bool = False,
    ) -> None:
        root = PROJECT_ROOT if project_root is None else project_root
        config = (
            load_detector_config()
            if detector_config is None
            else detector_config
        )
        self._mob_name = mob_name.lower()
        self._detector = MobDetector(root, config, use_sprite_grf=use_sprite_grf)
        # Discovery keeps its short detector-initialization/scan lock. Local
        # tracking deliberately does not use this coarse lock: its independent
        # jobs run through the bounded executor below.
        self._lock = threading.RLock()
        configured_workers = int(config.get("localTrackingWorkerCount", 4))
        self._tracking_worker_count = max(
            1,
            min(configured_workers, max(1, os.cpu_count() or 1)),
        )
        self._tracking_executor = ThreadPoolExecutor(
            max_workers=self._tracking_worker_count,
            thread_name_prefix=f"local-track-{self._mob_name}",
        )
        self._tracking_state_lock = threading.Lock()
        self._tracking_generation = 0
        self._tracking_closed = False
        self._tracking_inflight = 0
        self._tracking_reset_pending = False
        self._tracking_slots = threading.BoundedSemaphore(
            self._tracking_worker_count,
        )
        self._tracking_api_lock = threading.Lock()

    def ensure_descriptor(self):
        """Return the cached MobDescriptor for the session's mob."""
        return self._detector.ensure_descriptor(self._mob_name)

    def detector_config(self) -> dict:
        """Return the detector config dict."""
        return self._detector.config

    @property
    def mob_name(self) -> str:
        return self._mob_name

    def discover(self, roi: HuntRoi) -> DiscoveryScanResult:
        frame = capture_region(roi.x, roi.y, roi.w, roi.h)
        if frame is None:
            return DiscoveryScanResult(
                ok=False,
                fail_reason="capture_failed",
                raw_count=0,
                accepted_count=0,
                detections=[],
                duration_ms=0,
                elapsed_s=0.0,
            )
        return self.discover_frame(frame, roi)

    def discover_frame(
        self,
        frame: np.ndarray | None,
        roi: HuntRoi,
    ) -> DiscoveryScanResult:
        """Discovery scan — silhouette-gate pipeline.

        sprite.grf mode uses a deterministic 4× discovery scale for every
        mob, independent of descriptor dimensions. The pipeline is identical
        otherwise. Dedup against existing tracks is
        handled by TrackReconciler after detection.
        """
        if frame is None or frame.size == 0:
            return DiscoveryScanResult(
                ok=False,
                fail_reason="capture_failed",
                raw_count=0,
                accepted_count=0,
                detections=[],
                duration_ms=0,
                elapsed_s=0.0,
            )
        start = time.perf_counter()
        with self._lock:
            locked_at = time.perf_counter()
            result = self._detector.detect(
                frame,
                self._mob_name,
            )
        elapsed_s = time.perf_counter() - start
        duration_ms = int(elapsed_s * 1000)
        lock_wait_ms = int((locked_at - start) * 1000)
        detect_ms = duration_ms - lock_wait_ms

        accepted = [
            RawDetection(
                x=candidate.center_x + roi.x,
                y=candidate.center_y + roi.y,
                confidence=candidate.final_score,
                candidate_scale=candidate.candidate_scale,
                living=candidate.accepted,
                bbox=(
                    candidate.bbox[0] + roi.x,
                    candidate.bbox[1] + roi.y,
                    candidate.bbox[2],
                    candidate.bbox[3],
                ),
            )
            for candidate in result.accepted
        ]
        return DiscoveryScanResult(
            ok=True,
            fail_reason="",
            raw_count=len(result.candidates),
            accepted_count=len(accepted),
            detections=accepted,
            duration_ms=duration_ms,
            elapsed_s=elapsed_s,
            lock_wait_ms=lock_wait_ms,
            detect_ms=detect_ms,
            timing=dict(result.timing),
        )

    def transfer_track_template(
        self,
        source_track_id: int,
        target_track_id: int,
    ) -> bool:
        """Transfer a provisional local-track template to its real ID."""
        return transfer_track_template(
            self._detector,
            source_track_id,
            target_track_id,
        )

    def discard_track_template(self, track_id: int) -> None:
        """Discard an uncommitted provisional local-track template."""
        discard_track_template(self._detector, track_id)

    def clear_track_templates(self) -> None:
        """Drop temporal patches and invalidate jobs crossing an area boundary."""
        with self._tracking_state_lock:
            self._tracking_generation += 1
            self._tracking_reset_pending = self._tracking_inflight > 0
        clear_track_templates(self._detector)

    def close(self, *, wait: bool = True) -> None:
        """Stop the bounded tracking executor exactly once."""
        with self._tracking_state_lock:
            if self._tracking_closed:
                return
            self._tracking_closed = True
            self._tracking_generation += 1
        self._tracking_executor.shutdown(wait=wait, cancel_futures=True)

    @property
    def tracking_worker_count(self) -> int:
        """Configured bounded worker count for local tracking."""
        return self._tracking_worker_count

    @property
    def supports_async_tracking(self) -> bool:
        """Whether the coordinator may submit non-blocking per-track jobs."""
        return True

    def _prepare_tracking_jobs(
        self,
        roi: HuntRoi,
        track_snapshots: list[StateTrackSnapshot],
    ) -> list[tuple[StateTrackSnapshot, dict, list[tuple[int, int]]]]:
        """Build immutable per-track inputs from one coordinator snapshot."""
        all_roi_positions = [
            (snapshot.x - roi.x, snapshot.y - roi.y)
            for snapshot in track_snapshots
        ]
        jobs = []
        for i, snapshot in enumerate(track_snapshots):
            track = {
                "trackId": snapshot.track_id,
                "x": snapshot.x - roi.x,
                "y": snapshot.y - roi.y,
            }
            if snapshot.scale > 0:
                track["scale"] = snapshot.scale
            track["opacityBaseline"] = snapshot.opacity_baseline
            track["opacityBaselineSamples"] = snapshot.opacity_baseline_samples
            track["opacityDecayStreak"] = snapshot.opacity_decay_streak
            track["moving"] = snapshot.moving
            track["velX"] = snapshot.vel_x
            track["velY"] = snapshot.vel_y
            track["lostCount"] = snapshot.lost_count
            track["attackCount"] = snapshot.attack_count
            track["createdTick"] = snapshot.created_tick
            track["nowTick"] = snapshot.now_tick
            track["prediction_valid"] = snapshot.prediction_valid
            if snapshot.reanchor_x is not None and snapshot.reanchor_y is not None:
                track["reanchor_x"] = snapshot.reanchor_x - roi.x
                track["reanchor_y"] = snapshot.reanchor_y - roi.y
            other_positions = [
                pos for j, pos in enumerate(all_roi_positions) if j != i
            ]
            jobs.append((snapshot, track, other_positions))
        return jobs

    def _run_tracking_job(
        self,
        frame: np.ndarray,
        roi: HuntRoi,
        item: tuple[StateTrackSnapshot, dict, list[tuple[int, int]]],
    ) -> tuple[LocalTrackResult, float]:
        snapshot, track, other_positions = item
        job_start = time.perf_counter()
        try:
            result = self._detector.track_local(
                frame,
                self._mob_name,
                track,
                offset_x=roi.x,
                offset_y=roi.y,
                suppress_positions=other_positions if other_positions else None,
            )
        except BaseException:
            result = LocalTrackResult(
                track_id=snapshot.track_id,
                found=False,
                x=snapshot.x,
                y=snapshot.y,
                confidence=0.0,
                miss_reason="tracking_exception",
            )
        return result, (time.perf_counter() - job_start) * 1000.0

    def _finish_tracking_job(self) -> None:
        clear_after = False
        with self._tracking_state_lock:
            self._tracking_inflight = max(0, self._tracking_inflight - 1)
            if self._tracking_inflight == 0 and self._tracking_reset_pending:
                self._tracking_reset_pending = False
                clear_after = True
        # The semaphore is shared by synchronous and asynchronous callers. A
        # completed job must release the slot only after its cache publication
        # has finished, otherwise a new job could start while the old job is
        # still mutating its temporal template.
        self._tracking_slots.release()
        if clear_after:
            clear_track_templates(self._detector)

    def _acquire_tracking_slot(
        self,
        *,
        blocking: bool,
        expected_generation: int | None = None,
    ) -> bool:
        """Reserve one bounded executor slot for either tracking API."""
        if not self._tracking_slots.acquire(blocking=blocking):
            return False
        with self._tracking_state_lock:
            generation_changed = (
                expected_generation is not None
                and expected_generation != self._tracking_generation
            )
            if (
                self._tracking_closed
                or self._tracking_reset_pending
                or generation_changed
            ):
                self._tracking_slots.release()
                return False
            self._tracking_inflight += 1
        return True

    def submit_track_locals_frame(
        self,
        frame: np.ndarray | None,
        roi: HuntRoi,
        track_snapshots: list[StateTrackSnapshot],
        *,
        on_result: Callable[[LocalTrackResult], None],
    ) -> list[int]:
        """Submit independent local-follow jobs without waiting for the slowest.

        The returned IDs are the jobs accepted for this frame. Callers keep one
        in-flight job per Track; completed results are delivered immediately by
        ``on_result``. The executor queue is bounded by the active snapshot
        count and worker count, never by the coordinator's tick rate.
        """
        if frame is None or frame.size == 0 or not track_snapshots:
            return []
        with self._lock:
            self._detector.ensure_descriptor(self._mob_name)
        with self._tracking_state_lock:
            if self._tracking_closed or self._tracking_reset_pending:
                return []
            generation = self._tracking_generation
        jobs = self._prepare_tracking_jobs(roi, track_snapshots)
        accepted: list[int] = []
        for item in jobs:
            # A coordinator can call this every tracking tick, but only the
            # bounded worker slots may be active/queued. Unaccepted snapshots
            # remain eligible for the next tick instead of entering an
            # unbounded executor queue.
            if not self._acquire_tracking_slot(
                blocking=False,
                expected_generation=generation,
            ):
                break
            try:
                future = self._tracking_executor.submit(
                    self._run_tracking_job, frame, roi, item,
                )
            except BaseException:
                self._finish_tracking_job()
                raise
            accepted.append(item[0].track_id)

            def complete(
                done: Future,
                *,
                track_id: int = item[0].track_id,
                expected_generation: int = generation,
            ) -> None:
                try:
                    try:
                        result, _duration = done.result()
                    except BaseException:
                        return
                    with self._tracking_state_lock:
                        current = self._tracking_generation
                    if current == expected_generation:
                        try:
                            on_result(result)
                        except Exception:
                            # A consumer callback must never escape from the
                            # executor completion thread or skip slot cleanup.
                            # The coordinator independently validates epochs;
                            # a callback failure only loses that publication.
                            pass
                finally:
                    self._finish_tracking_job()

            future.add_done_callback(complete)
        return accepted

    def track_locals_frame(
        self,
        frame: np.ndarray | None,
        roi: HuntRoi,
        track_snapshots: list[StateTrackSnapshot],
        *,
        on_result: Callable[[LocalTrackResult], None] | None = None,
    ) -> LocalTrackBatchResult:
        if frame is None or frame.size == 0:
            return LocalTrackBatchResult(
                ok=False,
                fail_reason="capture_failed",
                results=[],
                duration_ms=0,
                found_count=0,
                coord_updates=0,
            )
        if not track_snapshots:
            return LocalTrackBatchResult(
                ok=True,
                fail_reason="",
                results=[],
                duration_ms=0,
                found_count=0,
                coord_updates=0,
            )
        with self._tracking_api_lock:
            return self._track_locals_frame_serial(
                frame, roi, track_snapshots, on_result=on_result,
            )

    def _track_locals_frame_serial(
        self,
        frame: np.ndarray | None,
        roi: HuntRoi,
        track_snapshots: list[StateTrackSnapshot],
        *,
        on_result: Callable[[LocalTrackResult], None] | None = None,
    ) -> LocalTrackBatchResult:
        start = time.perf_counter()
        # Ensure descriptor/cache initialization happens before worker fan-out;
        # the hot path then reads immutable detector state and only the temporal
        # template cache uses its narrow mutation lock.
        with self._lock:
            self._detector.ensure_descriptor(self._mob_name)
        locked_at = time.perf_counter()

        with self._tracking_state_lock:
            if self._tracking_closed:
                return LocalTrackBatchResult(
                    ok=False,
                    fail_reason="session_closed",
                    results=[],
                    duration_ms=0,
                    found_count=0,
                    coord_updates=0,
                )
            generation = self._tracking_generation

        # Collect all track positions before fan-out so every job uses the same
        # cross-track suppression snapshot. Jobs never mutate HuntTracks or the
        # input frame; publication belongs to the coordinator callback.
        all_roi_positions: list[tuple[int, int]] = [
            (snapshot.x - roi.x, snapshot.y - roi.y)
            for snapshot in track_snapshots
        ]
        jobs = []
        for i, snapshot in enumerate(track_snapshots):
            track = {
                "trackId": snapshot.track_id,
                "x": snapshot.x - roi.x,
                "y": snapshot.y - roi.y,
            }
            if snapshot.scale > 0:
                track["scale"] = snapshot.scale
            track["opacityBaseline"] = snapshot.opacity_baseline
            track["opacityBaselineSamples"] = snapshot.opacity_baseline_samples
            track["opacityDecayStreak"] = snapshot.opacity_decay_streak
            track["moving"] = snapshot.moving
            track["velX"] = snapshot.vel_x
            track["velY"] = snapshot.vel_y
            track["lostCount"] = snapshot.lost_count
            track["attackCount"] = snapshot.attack_count
            track["createdTick"] = snapshot.created_tick
            track["nowTick"] = snapshot.now_tick
            track["prediction_valid"] = snapshot.prediction_valid
            if snapshot.reanchor_x is not None and snapshot.reanchor_y is not None:
                track["reanchor_x"] = snapshot.reanchor_x - roi.x
                track["reanchor_y"] = snapshot.reanchor_y - roi.y
            other_positions = [
                pos for j, pos in enumerate(all_roi_positions) if j != i
            ]
            jobs.append((snapshot, track, other_positions))

        def run_one(item):
            snapshot, track, other_positions = item
            job_start = time.perf_counter()
            try:
                result = self._detector.track_local(
                    frame,
                    self._mob_name,
                    track,
                    offset_x=roi.x,
                    offset_y=roi.y,
                    suppress_positions=other_positions if other_positions else None,
                )
            except BaseException:
                # A single malformed/native tracking job must not kill the
                # coordinator or suppress publication for other mobs.
                result = LocalTrackResult(
                    track_id=snapshot.track_id,
                    found=False,
                    x=snapshot.x,
                    y=snapshot.y,
                    confidence=0.0,
                    miss_reason="tracking_exception",
                )
            return result, (time.perf_counter() - job_start) * 1000.0

        # Keep at most ``worker_count`` jobs submitted at once. This avoids
        # ThreadPoolExecutor's otherwise-unbounded internal work queue while
        # still allowing every active track to be processed in this batch.
        results_by_id: dict[int, LocalTrackResult] = {}
        durations: dict[int, float] = {}
        next_job = 0
        in_flight = {}
        scheduling_aborted = False
        while next_job < len(jobs) or in_flight:
            while next_job < len(jobs) and len(in_flight) < self._tracking_worker_count:
                if not self._acquire_tracking_slot(
                    blocking=True,
                    expected_generation=generation,
                ):
                    # Reset/close won while the batch was being scheduled. Let
                    # already submitted jobs finish, but never enqueue another
                    # old-generation job.
                    scheduling_aborted = True
                    next_job = len(jobs)
                    break
                try:
                    future = self._tracking_executor.submit(run_one, jobs[next_job])
                except BaseException:
                    self._finish_tracking_job()
                    raise
                in_flight[future] = jobs[next_job][0].track_id
                next_job += 1
            if not in_flight:
                break
            done, _pending = wait(
                tuple(in_flight), return_when=FIRST_COMPLETED,
            )
            for future in done:
                track_id = in_flight.pop(future)
                try:
                    result, duration_ms = future.result()
                    results_by_id[track_id] = result
                    durations[track_id] = duration_ms
                    with self._tracking_state_lock:
                        current_generation = self._tracking_generation
                    if on_result is not None and current_generation == generation:
                        on_result(result)
                finally:
                    self._finish_tracking_job()

        # Preserve snapshot order for deterministic application/telemetry even
        # though completion order is intentionally independent. A reset may
        # abort scheduling after a prefix of the batch; return a failed partial
        # result rather than raising KeyError or exposing old-area data.
        if scheduling_aborted or len(results_by_id) != len(jobs):
            partial_results = [
                results_by_id[snapshot.track_id]
                for snapshot, _, _ in jobs
                if snapshot.track_id in results_by_id
            ]
            return LocalTrackBatchResult(
                ok=False,
                fail_reason="tracking_reset",
                results=partial_results,
                duration_ms=int((time.perf_counter() - start) * 1000),
                found_count=sum(1 for result in partial_results if result.found),
                coord_updates=0,
                lock_wait_ms=int((locked_at - start) * 1000),
                compute_ms=max(
                    0,
                    int((time.perf_counter() - start) * 1000)
                    - int((locked_at - start) * 1000),
                ),
                track_durations_ms={
                    track_id: durations[track_id]
                    for track_id in durations
                    if track_id in {result.track_id for result in partial_results}
                },
            )
        results = [results_by_id[snapshot.track_id] for snapshot, _, _ in jobs]
        end = time.perf_counter()
        duration_ms = int((end - start) * 1000)
        lock_wait_ms = int((locked_at - start) * 1000)
        compute_ms = max(0, duration_ms - lock_wait_ms)
        with self._tracking_state_lock:
            generation_changed = generation != self._tracking_generation
        if generation_changed:
            # Area reset cleared the cache while jobs were running. Clear once
            # more after all jobs have joined so no old task leaves a patch in
            # the new area's cache.
            clear_track_templates(self._detector)
        found_count = sum(1 for result in results if result.found)
        coord_updates = sum(
            1
            for result, snapshot in zip(results, track_snapshots, strict=True)
            if result.found and (result.x != snapshot.x or result.y != snapshot.y)
        )
        return LocalTrackBatchResult(
            ok=True,
            fail_reason="",
            results=results,
            duration_ms=duration_ms,
            found_count=found_count,
            coord_updates=coord_updates,
            lock_wait_ms=lock_wait_ms,
            compute_ms=compute_ms,
            track_durations_ms=durations,
        )

