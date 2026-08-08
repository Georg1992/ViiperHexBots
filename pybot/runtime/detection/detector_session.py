"""Detector session — bridges hunt workers to pybot.recognition."""

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
        self._lock = threading.RLock()

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
        """Drop all temporal local-track patches at an area boundary."""
        clear_track_templates(self._detector)

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
            locked_at = time.perf_counter()
            # Collect all track positions (ROI-relative) so each track's peak search
            # can suppress heat near other tracks — prevents track swapping when mobs
            # are close together.
            all_roi_positions: list[tuple[int, int]] = [
                (snapshot.x - roi.x, snapshot.y - roi.y)
                for snapshot in track_snapshots
            ]
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
                # Other tracks' positions for heatmap suppression
                other_positions = [
                    pos for j, pos in enumerate(all_roi_positions) if j != i
                ]
                results.append(
                    self._detector.track_local(
                        frame,
                        self._mob_name,
                        track,
                        offset_x=roi.x,
                        offset_y=roi.y,
                        suppress_positions=other_positions if other_positions else None,
                    )
                )
        end = time.perf_counter()
        duration_ms = int((end - start) * 1000)
        lock_wait_ms = int((locked_at - start) * 1000)
        compute_ms = duration_ms - lock_wait_ms
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
        )

