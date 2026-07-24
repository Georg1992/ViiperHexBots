"""Detector session — bridges hunt workers to pybot.recognition."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pybot.paths import PROJECT_ROOT
from pybot.recognition.capture import capture_region
from pybot.recognition.detector.detector import MobDetector, load_detector_config
from pybot.recognition.detector.tracking.local_tracker import LocalTrackResult
from pybot.runtime.capture.window_roi import HuntRoi


@dataclass(frozen=True)
class RawDetection:
    x: int
    y: int
    confidence: float
    candidate_scale: float
    living: bool


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
    discovery_obs_x: int = 0
    discovery_obs_y: int = 0
    discovery_obs_tick: int = 0


@dataclass(frozen=True)
class LocalTrackBatchResult:
    ok: bool
    fail_reason: str
    results: list[LocalTrackResult]
    duration_ms: int
    found_count: int
    coord_updates: int


class DetectorSession:
    """One MobDetector behind an RLock — no IPC, no scale hard-lock."""

    def __init__(
        self,
        mob_name: str,
        project_root: Path | None = None,
        *,
        detector_config: dict | None = None,
    ) -> None:
        root = project_root or PROJECT_ROOT
        config = detector_config if detector_config is not None else load_detector_config()
        self._mob_name = mob_name.lower()
        self._detector = MobDetector(root, config)
        self._lock = threading.RLock()

    def is_busy(self) -> bool:
        acquired = self._lock.acquire(blocking=False)
        if acquired:
            self._lock.release()
            return False
        return True

    def ensure_descriptor(self):
        """Return the cached MobDescriptor for the session's mob."""
        return self._detector.ensure_descriptor(self._mob_name)

    def detector_config(self) -> dict:
        """Return the detector config dict (for opacity probe thresholds)."""
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
        *,
        known_tracks: list[tuple[int, int, int, float]] | None = None,
    ) -> DiscoveryScanResult:
        """Discovery scan: living silhouette gate only. Death is death-worker-owned.

        ``known_tracks`` are ``(track_id, screen_x, screen_y, scale)`` at capture
        time. Empty on the first scan (living-only). Later scans find heatmap
        peaks near known-track coords, skip pre-gates, and score against living
        silhouettes so an existing track is matched rather than created anew.
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
        frame_known: list[tuple[int, int, int, float]] = [
            (int(track_id), int(screen_x) - roi.x, int(screen_y) - roi.y, float(scale))
            for track_id, screen_x, screen_y, scale in (known_tracks or ())
        ]
        start = time.perf_counter()
        with self._lock:
            result = self._detector.detect(
                frame,
                self._mob_name,
                known_tracks=frame_known or None,
            )
        elapsed_s = time.perf_counter() - start
        duration_ms = int(elapsed_s * 1000)

        accepted = [
            RawDetection(
                x=candidate.center_x + roi.x,
                y=candidate.center_y + roi.y,
                confidence=candidate.final_score,
                candidate_scale=candidate.candidate_scale,
                living=candidate.accepted,
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
        )

    def track_locals(
        self,
        roi: HuntRoi,
        track_snapshots: list[StateTrackSnapshot],
    ) -> LocalTrackBatchResult:
        frame = capture_region(roi.x, roi.y, roi.w, roi.h)
        if frame is None:
            return LocalTrackBatchResult(
                ok=False,
                fail_reason="capture_failed",
                results=[],
                duration_ms=0,
                found_count=0,
                coord_updates=0,
            )
        return self.track_locals_frame(frame, roi, track_snapshots)

    def track_locals_frame(
        self,
        frame: np.ndarray | None,
        roi: HuntRoi,
        track_snapshots: list[StateTrackSnapshot],
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
                if snapshot.discovery_obs_tick > 0:
                    track["discoveryObsX"] = snapshot.discovery_obs_x - roi.x
                    track["discoveryObsY"] = snapshot.discovery_obs_y - roi.y
                    track["discoveryObsTick"] = snapshot.discovery_obs_tick
                results.append(
                    self._detector.track_local(
                        frame,
                        self._mob_name,
                        track,
                        offset_x=roi.x,
                        offset_y=roi.y,
                        skip_opacity=True,
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
