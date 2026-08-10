"""Detector sessions bridging runtime workers to recognition.

Discovery and tracking use separate sessions, but tracking itself has one
execution path: one immutable frame, one immutable Track snapshot, sequential
cheap local updates, and one ordered result batch.  This keeps coordinate age
coherent without per-Track threads or an async completion queue.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from pybot.paths import PROJECT_ROOT
from pybot.recognition.capture import capture_region
from pybot.recognition.detector.detector import MobDetector, load_detector_config
from pybot.recognition.detector.tracking.local_tracker import (
    LocalTrackResult,
    clear_track_states,
    discard_track_state,
    prune_track_states,
    transfer_track_state,
)
from pybot.runtime.capture.window_roi import HuntRoi


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
    lock_wait_ms: int = 0
    detect_ms: int = 0
    timing: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class StateTrackSnapshot:
    """Immutable inputs for one shared-frame local-follow cycle."""

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
    updated_tick: int = 0
    prediction_valid: bool = True


@dataclass(frozen=True)
class LocalTrackBatchResult:
    ok: bool
    fail_reason: str
    results: list[LocalTrackResult]
    duration_ms: int
    found_count: int
    coord_updates: int
    lock_wait_ms: int = 0
    compute_ms: int = 0
    track_durations_ms: dict[int, float] = field(default_factory=dict)


class DetectorSession:
    """One detector session with a single serialized tracking cycle."""

    def __init__(
        self,
        mob_name: str,
        project_root: Path | None = None,
        *,
        detector_config: dict | None = None,
        use_sprite_grf: bool = False,
    ) -> None:
        root = PROJECT_ROOT if project_root is None else project_root
        config = load_detector_config() if detector_config is None else config_or_copy(detector_config)
        self._mob_name = mob_name.lower()
        self._detector = MobDetector(root, config, use_sprite_grf=use_sprite_grf)
        self._lock = threading.RLock()
        self._tracking_lock = threading.Lock()
        self._closed = False

    def ensure_descriptor(self):
        return self._detector.ensure_descriptor(self._mob_name)

    def detector_config(self) -> dict:
        return self._detector.config

    @property
    def mob_name(self) -> str:
        return self._mob_name

    def discover(self, roi: HuntRoi) -> DiscoveryScanResult:
        frame = capture_region(roi.x, roi.y, roi.w, roi.h)
        if frame is None:
            return DiscoveryScanResult(False, "capture_failed", 0, 0, [], 0, 0.0)
        return self.discover_frame(frame, roi)

    def discover_frame(self, frame: np.ndarray | None, roi: HuntRoi) -> DiscoveryScanResult:
        if frame is None or frame.size == 0:
            return DiscoveryScanResult(False, "capture_failed", 0, 0, [], 0, 0.0)
        start = time.perf_counter()
        with self._lock:
            locked_at = time.perf_counter()
            result = self._detector.detect(frame, self._mob_name)
        elapsed = time.perf_counter() - start
        accepted = [
            RawDetection(
                x=c.center_x + roi.x,
                y=c.center_y + roi.y,
                confidence=c.final_score,
                candidate_scale=c.candidate_scale,
                living=c.accepted,
                bbox=(c.bbox[0] + roi.x, c.bbox[1] + roi.y, c.bbox[2], c.bbox[3]),
            )
            for c in result.accepted
        ]
        duration_ms = int(elapsed * 1000)
        lock_wait_ms = int((locked_at - start) * 1000)
        return DiscoveryScanResult(
            ok=True,
            fail_reason="",
            raw_count=len(result.candidates),
            accepted_count=len(accepted),
            detections=accepted,
            duration_ms=duration_ms,
            elapsed_s=elapsed,
            lock_wait_ms=lock_wait_ms,
            detect_ms=max(0, duration_ms - lock_wait_ms),
            timing=dict(result.timing),
        )

    def transfer_track_state(self, source_track_id: int, target_track_id: int) -> bool:
        return transfer_track_state(self._detector, source_track_id, target_track_id)

    def discard_track_state(self, track_id: int) -> None:
        discard_track_state(self._detector, track_id)

    def prune_track_states(self, active_track_ids: set[int]) -> None:
        prune_track_states(self._detector, active_track_ids)

    def clear_track_states(self) -> None:
        with self._tracking_lock:
            clear_track_states(self._detector)

    def close(self, *, wait: bool = True) -> None:
        del wait
        with self._tracking_lock:
            if self._closed:
                return
            self._closed = True
            clear_track_states(self._detector)

    def _track_one(
        self,
        frame: np.ndarray,
        roi: HuntRoi,
        snapshot: StateTrackSnapshot,
        other_positions: list[tuple[int, int]],
    ) -> tuple[LocalTrackResult, float]:
        started = time.perf_counter()
        track = {
            "trackId": snapshot.track_id,
            "x": snapshot.x - roi.x,
            "y": snapshot.y - roi.y,
            "scale": snapshot.scale,
            "moving": snapshot.moving,
            "velX": snapshot.vel_x,
            "velY": snapshot.vel_y,
            "lostCount": snapshot.lost_count,
            "nowTick": snapshot.now_tick,
            "updatedTick": snapshot.updated_tick,
            "prediction_valid": snapshot.prediction_valid,
        }
        try:
            result = self._detector.track_local(
                frame,
                self._mob_name,
                track,
                offset_x=roi.x,
                offset_y=roi.y,
                suppress_positions=other_positions or None,
            )
        except Exception:
            result = LocalTrackResult(
                track_id=snapshot.track_id,
                found=False,
                x=snapshot.x,
                y=snapshot.y,
                confidence=0.0,
                miss_reason="tracking_exception",
            )
        return result, (time.perf_counter() - started) * 1000.0

    def track_locals_frame(
        self,
        frame: np.ndarray | None,
        roi: HuntRoi,
        track_snapshots: list[StateTrackSnapshot],
        *,
        on_result=None,
    ) -> LocalTrackBatchResult:
        """Track every snapshot against the same frame in snapshot order."""
        if frame is None or frame.size == 0:
            return LocalTrackBatchResult(False, "capture_failed", [], 0, 0, 0)
        if not track_snapshots:
            return LocalTrackBatchResult(True, "", [], 0, 0, 0)
        started = time.perf_counter()
        with self._tracking_lock:
            if self._closed:
                return LocalTrackBatchResult(False, "session_closed", [], 0, 0, 0)
            with self._lock:
                self._detector.ensure_descriptor(self._mob_name)
            positions = [(s.x - roi.x, s.y - roi.y) for s in track_snapshots]
            results: list[LocalTrackResult] = []
            durations: dict[int, float] = {}
            for index, snapshot in enumerate(track_snapshots):
                other = [position for i, position in enumerate(positions) if i != index]
                result, duration = self._track_one(frame, roi, snapshot, other)
                results.append(result)
                durations[result.track_id] = duration
                if on_result is not None:
                    on_result(result)
        duration_ms = int((time.perf_counter() - started) * 1000)
        found_count = sum(1 for result in results if result.found)
        coord_updates = sum(
            1 for result, snapshot in zip(results, track_snapshots)
            if result.found and (result.x != snapshot.x or result.y != snapshot.y)
        )
        return LocalTrackBatchResult(
            True, "", results, duration_ms, found_count, coord_updates,
            lock_wait_ms=0, compute_ms=duration_ms, track_durations_ms=durations,
        )


def config_or_copy(config: dict) -> dict:
    """Copy caller configuration so detector/session state stays private."""
    return dict(config)
