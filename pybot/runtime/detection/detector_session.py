"""Direct mob-recognition access for hunt workers."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import pybot.runtime._mob_rec_path as _mob_rec_path  # sets up sys.path for mob-recognition modules
from pybot.paths import PROJECT_ROOT
from pybot.runtime.capture.window_roi import HuntRoi

from capture import capture_region
from detector import (
    STATE_PROFILE_DIRECT,
    STATE_PROFILE_FULL,
    SimpleMobDetector,
    load_simple_config,
)
from tracking.local_tracker import LocalTrackResult
from tracking.state_recognizer import evaluate_track_state


@dataclass(frozen=True)
class RawDetection:
    x: int
    y: int
    confidence: float
    candidate_scale: float
    living: bool
    dead: bool


@dataclass(frozen=True)
class DiscoveryScanResult:
    ok: bool
    fail_reason: str
    raw_count: int
    accepted_count: int
    detections: list[RawDetection]
    duration_ms: int
    elapsed_s: float


@dataclass(frozen=True)
class StateTrackSnapshot:
    track_id: int
    x: int
    y: int
    scale: float = 0.0


@dataclass(frozen=True)
class StateObservationResult:
    track_id: int
    state: str
    confidence: float
    x: int
    y: int
    candidate_scale: float = 0.0
    observed_at_ms: int = 0


@dataclass(frozen=True)
class StateBatchResult:
    ok: bool
    fail_reason: str
    observations: list[StateObservationResult]
    duration_ms: int
    coord_updates: int


@dataclass(frozen=True)
class LocalTrackBatchResult:
    ok: bool
    fail_reason: str
    results: list[LocalTrackResult]
    duration_ms: int
    found_count: int
    coord_updates: int


class DetectorSession:
    """One SimpleMobDetector behind an RLock — no IPC, no scale hard-lock."""

    def __init__(self, mob_name: str, project_root: Path | None = None) -> None:
        root = project_root or PROJECT_ROOT
        config = load_simple_config()
        self._mob_name = mob_name.lower()
        self._detector = SimpleMobDetector(root, config)
        self._detector.apply_runtime_config(config)
        self._lock = threading.RLock()

    def is_busy(self) -> bool:
        acquired = self._lock.acquire(blocking=False)
        if acquired:
            self._lock.release()
            return False
        return True

    @property
    def mob_name(self) -> str:
        return self._mob_name

    def discover(self, roi: HuntRoi) -> DiscoveryScanResult:
        frame = capture_region(roi.x, roi.y, roi.w, roi.h)
        return self.discover_frame(frame, roi)

    def discover_frame(self, frame: np.ndarray, roi: HuntRoi) -> DiscoveryScanResult:
        start = time.perf_counter()
        with self._lock:
            result = self._detector.detect(frame, self._mob_name)
        elapsed_s = time.perf_counter() - start
        duration_ms = int(elapsed_s * 1000)

        raw = [
            RawDetection(
                x=candidate.center_x + roi.x,
                y=candidate.center_y + roi.y,
                confidence=candidate.final_score,
                candidate_scale=candidate.candidate_scale,
                living=candidate.accepted and not candidate.is_dead,
                dead=candidate.is_dead,
            )
            for candidate in result.accepted
        ]
        return DiscoveryScanResult(
            ok=True,
            fail_reason="",
            raw_count=len(raw),
            accepted_count=len(raw),
            detections=raw,
            duration_ms=duration_ms,
            elapsed_s=elapsed_s,
        )

    def track_locals(
        self,
        roi: HuntRoi,
        track_snapshots: list[StateTrackSnapshot],
    ) -> LocalTrackBatchResult:
        frame = capture_region(roi.x, roi.y, roi.w, roi.h)
        return self.track_locals_frame(frame, roi, track_snapshots)

    def track_locals_frame(
        self,
        frame: np.ndarray,
        roi: HuntRoi,
        track_snapshots: list[StateTrackSnapshot],
    ) -> LocalTrackBatchResult:
        if not track_snapshots:
            return LocalTrackBatchResult(
                ok=True,
                fail_reason="",
                results=[],
                duration_ms=0,
                found_count=0,
                coord_updates=0,
            )
        start = time.perf_counter()
        results: list[LocalTrackResult] = []
        with self._lock:
            for snapshot in track_snapshots:
                track = {
                    "trackId": snapshot.track_id,
                    "x": snapshot.x - roi.x,
                    "y": snapshot.y - roi.y,
                }
                if snapshot.scale > 0:
                    track["scale"] = snapshot.scale
                results.append(
                    self._detector.track_local(
                        frame,
                        self._mob_name,
                        track,
                        offset_x=roi.x,
                        offset_y=roi.y,
                    )
                )
        duration_ms = int((time.perf_counter() - start) * 1000)
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
        )

    def state_confirm(
        self,
        roi: HuntRoi,
        track_snapshot: StateTrackSnapshot,
    ) -> StateBatchResult:
        frame = capture_region(roi.x, roi.y, roi.w, roi.h)
        return self.state_confirm_frame(frame, roi, track_snapshot)

    def state_confirm_frame(
        self,
        frame: np.ndarray,
        roi: HuntRoi,
        track_snapshot: StateTrackSnapshot,
    ) -> StateBatchResult:
        return self._state_eval_frame(
            frame,
            roi,
            track_snapshot,
            profile=STATE_PROFILE_FULL,
        )

    def state_direct(
        self,
        roi: HuntRoi,
        track_snapshot: StateTrackSnapshot,
    ) -> StateBatchResult:
        frame = capture_region(roi.x, roi.y, roi.w, roi.h)
        return self.state_direct_frame(frame, roi, track_snapshot)

    def state_direct_frame(
        self,
        frame: np.ndarray,
        roi: HuntRoi,
        track_snapshot: StateTrackSnapshot,
    ) -> StateBatchResult:
        return self._state_eval_frame(
            frame,
            roi,
            track_snapshot,
            profile=STATE_PROFILE_DIRECT,
        )

    def _state_eval_frame(
        self,
        frame: np.ndarray,
        roi: HuntRoi,
        track_snapshot: StateTrackSnapshot,
        *,
        profile,
    ) -> StateBatchResult:
        observed_at_ms = int(time.perf_counter() * 1000)
        start = time.perf_counter()
        with self._lock:
            update = evaluate_track_state(
                self._detector,
                frame,
                self._mob_name,
                track_snapshot.track_id,
                track_snapshot.x - roi.x,
                track_snapshot.y - roi.y,
                offset_x=roi.x,
                offset_y=roi.y,
                scale_hint=track_snapshot.scale if track_snapshot.scale > 0 else None,
                profile=profile,
            )
        duration_ms = int((time.perf_counter() - start) * 1000)
        observation = _update_to_observation(update, observed_at_ms=observed_at_ms)
        coord_updates = 1 if observation.state == "alive" else 0
        return StateBatchResult(
            ok=True,
            fail_reason="",
            observations=[observation],
            duration_ms=duration_ms,
            coord_updates=coord_updates,
        )


def _update_to_observation(update: dict, *, observed_at_ms: int = 0) -> StateObservationResult:
    candidate_scale = float(update.get("candidateScale", 0.0) or 0.0)
    return StateObservationResult(
        track_id=int(update["trackId"]),
        state=str(update["state"]),
        confidence=float(update.get("confidence", 0.0) or 0.0),
        x=int(update.get("x", 0) or 0),
        y=int(update.get("y", 0) or 0),
        candidate_scale=candidate_scale,
        observed_at_ms=observed_at_ms,
    )
