"""Simple descriptor heatmap detector."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from descriptors.descriptor import SimpleMobDescriptor
from descriptors.descriptor_builder import DESCRIPTOR_VERSION
from scoring.heatmap_detector import HeatmapDetector, Heatmaps
from scoring.region_scorer import RegionScore, SimpleRegionScorer


REQUIRED_CONFIG_KEYS = {
    "minColorPurity",
    "minBodyPaletteScore",
    "minAccentScore",
    "minLocalPatternScore",
    "minDiscoveryHeatmapScore",

    "discoveryHeatmapDownscale",
    "discoveryHeatmapDownscaleMinSide",
    "watchDriftRadiusPx",
    "watchDriftStepPx",
    "localTrackSearchRadiusPx",
    "minDescriptorColorMatch",
    "maxSpritePaletteDistance",
    "minSpritePaletteMatch",
    "maxRareToBodyRatio",
    "minInformativePixelFraction",
    "maxDescriptorPixelFraction",
    "minDiscoverySizeScore",
    "minObjectSizeScore",
    "enforceObjectSizeGate",
    "topCandidateCenters",
    "minCenterDistancePx",
    "minCenterHeat",
    "nmsDistancePx",
    "matchRadiusPx",
    "maxCandidates",
    "smallScaleMinFrameWidth",
    "selfExclusionWidthRatio",
    "selfExclusionHeightRatio",
    "centerScales",
    "scales",
    "playfieldTopRatio",
    "playfieldBottomRatio",
    "playfieldLeftRatio",
    "playfieldRightRatio",
    "debugOutputDir",
    "centerWeights",
    "minDominantPixelFraction",
    "dominantPixelDistance",
}

REQUIRED_CENTER_WEIGHT_KEYS = {"bodyPalette", "accent", "rareColor", "localPattern"}


@dataclass(frozen=True)
class StateSearchProfile:
    """Search geometry for canonical state evaluation."""

    drift_radius_px: int | None = None
    drift_step_px: int | None = None
    single_scale: bool = False
    early_exit_at_center: bool = True


STATE_PROFILE_FULL = StateSearchProfile()
STATE_PROFILE_DIRECT = StateSearchProfile(
    drift_radius_px=15,
    single_scale=True,
    early_exit_at_center=False,
)


@dataclass
class SimpleCandidate:
    mob_name: str
    center_x: int
    center_y: int
    bbox: tuple[int, int, int, int]
    final_score: float
    body_palette_score: float
    accent_score: float
    rare_color_score: float
    local_pattern_score: float
    color_purity_score: float
    size_score: float
    candidate_scale: float
    accepted: bool
    rejection_reason: str
    heatmap_score: float

    def to_dict(self) -> dict:
        x, y, w, h = self.bbox
        return {
            "mobName": self.mob_name,
            "center": [self.center_x, self.center_y],
            "centerX": self.center_x,
            "centerY": self.center_y,
            "bbox": [x, y, w, h],
            "finalScore": round(self.final_score, 4),
            "bodyPaletteScore": round(self.body_palette_score, 4),
            "accentScore": round(self.accent_score, 4),
            "rareColorScore": round(self.rare_color_score, 4),
            "localPatternScore": round(self.local_pattern_score, 4),
            "colorPurityScore": round(self.color_purity_score, 4),
            "sizeScore": round(self.size_score, 4),
            "candidateScale": round(self.candidate_scale, 4),
            "heatmapScore": round(self.heatmap_score, 4),
            "accepted": self.accepted,
            "rejectionReason": self.rejection_reason,
        }


@dataclass
class SimpleDetectionResult:
    mob_name: str
    descriptor: SimpleMobDescriptor
    heatmaps: Heatmaps | None
    candidates: list[SimpleCandidate]
    accepted: list[SimpleCandidate]
    elapsed_s: float
    timing: dict[str, float]


def load_simple_config(path: Optional[Path] = None) -> dict:
    config_path = path or (Path(__file__).resolve().parent / "config_simple.json")
    import json

    config = json.loads(config_path.read_text(encoding="utf-8"))
    missing = sorted(REQUIRED_CONFIG_KEYS - set(config))
    if missing:
        raise ValueError(f"missing detector config keys: {', '.join(missing)}")
    center_weight_missing = sorted(REQUIRED_CENTER_WEIGHT_KEYS - set(config["centerWeights"]))
    if center_weight_missing:
        raise ValueError(f"missing center weight keys: {', '.join(center_weight_missing)}")
    return config


class SimpleMobDetector:
    def __init__(self, project_root: Path, config: Optional[dict] = None):
        self.project_root = project_root
        self.config = load_simple_config() if config is None else config
        self.heatmap_detector = HeatmapDetector(self.config)
        self.region_scorer = SimpleRegionScorer(self.config)
        self._descriptor_cache: dict[str, SimpleMobDescriptor] = {}
        self.self_exclusion_width_ratio = float(self.config["selfExclusionWidthRatio"])
        self.self_exclusion_height_ratio = float(self.config["selfExclusionHeightRatio"])
        self.small_scale_min_frame_width = int(self.config["smallScaleMinFrameWidth"])
        self.min_discovery_heatmap_score = float(self.config["minDiscoveryHeatmapScore"])

        self.discovery_heatmap_downscale = int(self.config["discoveryHeatmapDownscale"])
        self.discovery_heatmap_downscale_min_side = int(self.config["discoveryHeatmapDownscaleMinSide"])
        self.watch_drift_radius_px = int(self.config["watchDriftRadiusPx"])
        self.watch_drift_step_px = int(self.config["watchDriftStepPx"])
        self.local_track_search_radius_px = int(self.config["localTrackSearchRadiusPx"])

    def apply_runtime_config(self, config: dict) -> None:
        prior = self.config
        self.config = dict(config)
        scale_keys = ("scales", "centerScales")
        if any(prior.get(key) != self.config.get(key) for key in scale_keys):
            self.heatmap_detector = HeatmapDetector(self.config)
        self.region_scorer = SimpleRegionScorer(self.config)
        self.self_exclusion_width_ratio = float(self.config["selfExclusionWidthRatio"])
        self.self_exclusion_height_ratio = float(self.config["selfExclusionHeightRatio"])
        self.small_scale_min_frame_width = int(self.config["smallScaleMinFrameWidth"])
        self.min_discovery_heatmap_score = float(self.config["minDiscoveryHeatmapScore"])

        self.discovery_heatmap_downscale = int(self.config["discoveryHeatmapDownscale"])
        self.discovery_heatmap_downscale_min_side = int(self.config["discoveryHeatmapDownscaleMinSide"])
        self.watch_drift_radius_px = int(self.config["watchDriftRadiusPx"])
        self.watch_drift_step_px = int(self.config["watchDriftStepPx"])
        self.local_track_search_radius_px = int(self.config["localTrackSearchRadiusPx"])

    def descriptor_path(self, mob_name: str) -> Path:
        return self.project_root / "assets" / "generated_descriptors" / mob_name.lower() / "simple" / "descriptor.json"

    def ensure_descriptor(self, mob_name: str) -> SimpleMobDescriptor:
        mob_name = mob_name.lower()
        if mob_name in self._descriptor_cache:
            return self._descriptor_cache[mob_name]
        path = self.descriptor_path(mob_name)
        if not path.exists():
            raise FileNotFoundError(f"descriptor not found for mob '{mob_name}': {path}")
        descriptor = SimpleMobDescriptor.load(path)
        if descriptor.version < DESCRIPTOR_VERSION:
            raise RuntimeError(
                f"descriptor for mob '{mob_name}' is version {descriptor.version}; "
                f"rebuild descriptor version {DESCRIPTOR_VERSION} before detection"
            )
        self._descriptor_cache[mob_name] = descriptor
        return descriptor

    def detect(
        self,
        frame_bgr: np.ndarray,
        mob_name: str,
    ) -> SimpleDetectionResult:
        start = time.perf_counter()
        descriptor = self.ensure_descriptor(mob_name)
        hsv_start = time.perf_counter()
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        heatmap_start = time.perf_counter()
        frame_height, frame_width = frame_bgr.shape[:2]
        min_side = min(frame_width, frame_height)
        downscale = self.discovery_heatmap_downscale
        if (
            downscale > 1
            and (
                min_side < self.discovery_heatmap_downscale_min_side
                or abs(frame_width - frame_height) > 64
            )
        ):
            downscale = 1
        heatmaps = self.heatmap_detector.build_heatmaps(
            frame_bgr,
            hsv,
            descriptor,
            downscale=downscale,
        )
        centers_start = time.perf_counter()
        centers = self.heatmap_detector.top_centers(heatmaps.final_center)
        score_start = time.perf_counter()
        candidates: list[SimpleCandidate] = []
        for cx, cy, heat_score in centers:
            candidates.extend(self._evaluate_discovery_center(frame_bgr, hsv, descriptor, cx, cy, heat_score))
        nms_start = time.perf_counter()
        accepted = self._finalize_accepted(candidates)
        accepted_ids = {id(candidate) for candidate in accepted}
        final_candidates = []
        for candidate in candidates:
            if candidate.accepted and id(candidate) not in accepted_ids:
                candidate.accepted = False
                candidate.rejection_reason = "nms_suppressed"
            final_candidates.append(candidate)
        elapsed = time.perf_counter() - start
        timing = {
            "hsv": hsv_start - start,
            "heatmaps": centers_start - heatmap_start,
            "centers": score_start - centers_start,
            "scoring": nms_start - score_start,
            "nms": time.perf_counter() - nms_start,
            "total": elapsed,
        }
        return SimpleDetectionResult(
            mob_name=mob_name.lower(),
            descriptor=descriptor,
            heatmaps=heatmaps,
            candidates=final_candidates[: int(self.config["maxCandidates"])],
            accepted=accepted,
            elapsed_s=elapsed,
            timing=timing,
        )

    def _evaluate_discovery_center(
        self,
        frame_bgr: np.ndarray,
        hsv: np.ndarray,
        descriptor: SimpleMobDescriptor,
        cx: int,
        cy: int,
        heat_score: float,
    ) -> list[SimpleCandidate]:
        """Discovery scan: score one heatmap peak center. Returns accepted candidate or empty list."""
        if self._is_self_center(cx, cy, frame_bgr.shape):
            return []
        if heat_score < self.min_discovery_heatmap_score:
            return []

        scales = self._candidate_scales(frame_bgr.shape[1])
        living = self._score_point_at(frame_bgr, hsv, descriptor, cx, cy, scales=scales)
        if living and living.accepted:
            # Dominant pixel gate: discovery-only filter. Rejects candidates where
            # too few pixels match the exact dominant sprite pixel color.
            if descriptor.dominant_pixel_bgr is not None:
                x, y, w, h = living.bbox
                region = frame_bgr[y:y+h, x:x+w]
                if region.size > 0:
                    dominant = np.array(descriptor.dominant_pixel_bgr, dtype=np.float32).reshape(1, 1, 3)
                    diff = region.astype(np.float32) - dominant
                    dist = np.sqrt(np.sum(diff * diff, axis=2))
                    frac = float(np.mean(dist <= float(self.config.get("dominantPixelDistance", 12))))
                    min_frac = float(self.config.get("minDominantPixelFraction", 0.02))
                    if frac < min_frac:
                        return []
            living.heatmap_score = heat_score
            return [living]
        return []

    def _score_point_at(
        self,
        frame_bgr: np.ndarray,
        hsv: np.ndarray,
        descriptor: SimpleMobDescriptor,
        cx: int,
        cy: int,
        scales: list[float] | None = None,
    ) -> SimpleCandidate | None:
        """Score living signal at one point. Returns the best accepted candidate or None."""
        if cx < 0 or cy < 0 or cx >= frame_bgr.shape[1] or cy >= frame_bgr.shape[0]:
            return None

        best: tuple[float, tuple[int, int, int, int], RegionScore] | None = None

        if scales is None:
            scales = self._candidate_scales(frame_bgr.shape[1])

        for scale in scales:
            living_bbox = self._bbox_for_size(
                cx,
                cy,
                int(round(descriptor.avg_width * scale)),
                int(round(descriptor.avg_height * scale)),
                frame_bgr.shape,
            )
            if living_bbox is None:
                continue

            living_score = self.region_scorer.score(
                frame_bgr, hsv, descriptor, living_bbox, expected_scale=float(scale)
            )

            if living_score.accepted and (
                best is None or living_score.final_score > best[2].final_score
            ):
                best = (float(scale), living_bbox, living_score)
                if living_score.final_score >= 0.30:
                    break

        if best is not None:
            scale, bbox, score = best
            bx, by, bw, bh = bbox
            cx = bx + bw // 2
            cy = by + bh // 2
            return self._living_candidate(
                descriptor.mob_name,
                cx,
                cy,
                bbox,
                score,
                1.0,
                scale,
            )
        return None

    def _score_living_only_at(
        self,
        frame_bgr: np.ndarray,
        hsv: np.ndarray,
        descriptor: SimpleMobDescriptor,
        cx: int,
        cy: int,
        scale: float,
    ) -> tuple[RegionScore | None, tuple[int, int, int, int] | None]:
        """Living region score at one point — single scale."""
        living_bbox = self._bbox_for_size(
            cx,
            cy,
            int(round(descriptor.avg_width * scale)),
            int(round(descriptor.avg_height * scale)),
            frame_bgr.shape,
        )
        if living_bbox is None:
            return None, None
        score = self.region_scorer.score(
            frame_bgr,
            hsv,
            descriptor,
            living_bbox,
            expected_scale=float(scale),
        )
        return score, living_bbox

    def track_local(
        self,
        frame_bgr: np.ndarray,
        mob_name: str,
        track: dict,
        *,
        offset_x: int = 0,
        offset_y: int = 0,
        search_radius_px: int | None = None,
    ):
        from tracking.local_tracker import track_local as run_track_local

        return run_track_local(
            self,
            frame_bgr,
            mob_name,
            track,
            offset_x=offset_x,
            offset_y=offset_y,
            search_radius_px=search_radius_px,
        )

    def _evaluate_track_point(
        self,
        frame_bgr: np.ndarray,
        hsv: np.ndarray,
        descriptor: SimpleMobDescriptor,
        cx: int,
        cy: int,
        scales: list[float] | None = None,
    ) -> list[SimpleCandidate]:
        """Track state: living position at a known track point."""
        living = self._score_point_at(frame_bgr, hsv, descriptor, cx, cy, scales=scales)
        if living is not None and living.accepted:
            return [living]
        return []

    def _state_update_from_candidate(
        self,
        track_id: int,
        cx: int,
        cy: int,
        best: SimpleCandidate | None,
        *,
        offset_x: int,
        offset_y: int,
    ) -> dict:
        if best is None:
            return {
                "trackId": track_id,
                "state": "unreachable",
                "confidence": 0.0,
                "x": cx + offset_x,
                "y": cy + offset_y,
            }
        return {
            "trackId": track_id,
            "state": "alive",
            "confidence": round(best.final_score, 4),
            "x": best.center_x + offset_x,
            "y": best.center_y + offset_y,
            "candidateScale": round(best.candidate_scale, 4),
        }

    def _evaluate_one_track(
        self,
        frame_bgr: np.ndarray,
        hsv: np.ndarray,
        descriptor: SimpleMobDescriptor,
        track_id: int,
        cx: int,
        cy: int,
        *,
        offset_x: int = 0,
        offset_y: int = 0,
        scale_hint: float | None = None,
        profile: StateSearchProfile = STATE_PROFILE_FULL,
    ) -> dict:
        if profile.single_scale:
            track_scales = [self._direct_track_scale(frame_bgr.shape[1], scale_hint=scale_hint)]
        else:
            track_scales = self._scales_for_track(
                frame_bgr.shape[1],
                float(scale_hint) if scale_hint is not None else None,
            )
        point_candidates: list[SimpleCandidate] = []
        search_centers = self._track_search_centers(
            cx,
            cy,
            frame_bgr.shape,
            radius_px=profile.drift_radius_px,
            step_px=profile.drift_step_px,
        )
        for index, (center_x, center_y) in enumerate(search_centers):
            point_candidates.extend(
                self._evaluate_track_point(
                    frame_bgr,
                    hsv,
                    descriptor,
                    center_x,
                    center_y,
                    scales=track_scales,
                )
            )
            if profile.early_exit_at_center and index == 0:
                center_best, _ = self._select_track_state_candidate(point_candidates)
                if center_best is not None and center_best.accepted:
                    break
        best, arbitration = self._select_track_state_candidate(point_candidates)
        arbitration["scales"] = track_scales
        self._log_state_arbitration(track_id, arbitration)
        return self._state_update_from_candidate(
            track_id,
            cx,
            cy,
            best,
            offset_x=offset_x,
            offset_y=offset_y,
        )

    def evaluate_track_state(
        self,
        frame_bgr: np.ndarray,
        mob_name: str,
        track_id: int,
        cx: int,
        cy: int,
        *,
        offset_x: int = 0,
        offset_y: int = 0,
        scale_hint: float | None = None,
        profile: StateSearchProfile = STATE_PROFILE_FULL,
    ) -> dict:
        """Canonical single-track state: alive or unreachable."""
        descriptor = self.ensure_descriptor(mob_name)
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        return self._evaluate_one_track(
            frame_bgr,
            hsv,
            descriptor,
            track_id,
            cx,
            cy,
            offset_x=offset_x,
            offset_y=offset_y,
            scale_hint=scale_hint,
            profile=profile,
        )

    def evaluate_track_states(
        self,
        frame_bgr: np.ndarray,
        mob_name: str,
        tracks: list[dict],
        *,
        offset_x: int = 0,
        offset_y: int = 0,
        profile: StateSearchProfile = STATE_PROFILE_FULL,
    ) -> list[dict]:
        """Evaluate known tracks by id (ROI-local x,y). Returns trackUpdates with screen coordinates."""
        if not tracks:
            return []

        descriptor = self.ensure_descriptor(mob_name)
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        updates: list[dict] = []
        for track in tracks:
            scale_hint = track.get("scale")
            updates.append(
                self._evaluate_one_track(
                    frame_bgr,
                    hsv,
                    descriptor,
                    int(track["trackId"]),
                    int(track["x"]),
                    int(track["y"]),
                    offset_x=offset_x,
                    offset_y=offset_y,
                    scale_hint=float(scale_hint) if scale_hint is not None else None,
                    profile=profile,
                )
            )
        return updates

    def _direct_track_scale(self, frame_width: int, scale_hint: float | None = None) -> float:
        if scale_hint is not None:
            track_scales = self._scales_for_track(frame_width, float(scale_hint))
            return track_scales[len(track_scales) // 2]
        track_scales = self._candidate_scales(frame_width)
        return track_scales[len(track_scales) // 2]

    def evaluate_track_state_direct(
        self,
        frame_bgr: np.ndarray,
        mob_name: str,
        track_id: int,
        cx: int,
        cy: int,
        *,
        offset_x: int = 0,
        offset_y: int = 0,
        scale_hint: float | None = None,
    ) -> dict:
        """Post-attack / urgent state: lightweight search profile."""
        return self.evaluate_track_state(
            frame_bgr,
            mob_name,
            track_id,
            cx,
            cy,
            offset_x=offset_x,
            offset_y=offset_y,
            scale_hint=scale_hint,
            profile=STATE_PROFILE_DIRECT,
        )

    @staticmethod
    def _bbox_for_size(
        cx: int,
        cy: int,
        width: int,
        height: int,
        frame_shape: tuple[int, ...],
    ) -> tuple[int, int, int, int] | None:
        w = max(8, width)
        h = max(8, height)
        x = int(round(cx - w / 2))
        y = int(round(cy - h / 2))
        frame_h, frame_w = frame_shape[:2]
        if x < 0 or y < 0 or x + w > frame_w or y + h > frame_h:
            return None
        return x, y, w, h

    def _track_search_centers(
        self,
        cx: int,
        cy: int,
        frame_shape: tuple[int, ...],
        *,
        radius_px: int | None = None,
        step_px: int | None = None,
    ) -> list[tuple[int, int]]:
        radius = self.watch_drift_radius_px if radius_px is None else radius_px
        step = self.watch_drift_step_px if step_px is None else step_px
        step = max(8, step)
        if radius <= 0:
            return [(cx, cy)]

        frame_h, frame_w = frame_shape[:2]
        points: list[tuple[int, int]] = []
        for dx in range(-radius, radius + 1, step):
            for dy in range(-radius, radius + 1, step):
                nx = cx + dx
                ny = cy + dy
                if 0 <= nx < frame_w and 0 <= ny < frame_h:
                    points.append((nx, ny))
        return points or [(cx, cy)]

    def _select_track_state_candidate(
        self, candidates: list[SimpleCandidate]
    ) -> tuple[SimpleCandidate | None, dict[str, object]]:
        if not candidates:
            return None, {
                "bestScore": 0.0,
                "selectedState": "unreachable",
                "reason": "unreachable_no_candidate",
            }

        best = max(candidates, key=lambda c: c.final_score)
        if best.accepted:
            return best, {
                "bestScore": best.final_score,
                "selectedState": "alive",
                "reason": "living_found",
            }

        return None, {
            "bestScore": best.final_score,
            "selectedState": "unreachable",
            "reason": "unreachable_no_accepted",
        }

    @staticmethod
    def _log_state_arbitration(track_id: int, arbitration: dict[str, object]) -> None:
        scales = arbitration.get("scales", [])
        scale_text = ",".join(f"{float(value):.3f}" for value in scales) if scales else "-"
        print(
            (
                f"state_arbitration trackId={track_id} "
                f"bestScore={float(arbitration['bestScore']):.4f} "
                f"selectedState={arbitration['selectedState']} "
                f"reason={arbitration['reason']} "
                f"scales={scale_text}"
            ),
            file=sys.stderr,
            flush=True,
        )

    def _is_self_center(self, cx: int, cy: int, frame_shape: tuple[int, ...]) -> bool:
        height, width = frame_shape[:2]
        half_width = width * self.self_exclusion_width_ratio * 0.5
        half_height = height * self.self_exclusion_height_ratio * 0.5
        return abs(cx - width / 2) <= half_width and abs(cy - height / 2) <= half_height

    def _candidate_scales(self, frame_width: int) -> list[float]:
        scales = [
            float(scale)
            for scale in self.config["scales"]
            if float(scale) >= 0.75 or frame_width >= self.small_scale_min_frame_width
        ]
        if scales:
            return scales
        return [float(self.config["scales"][0])]

    def _scales_for_track(self, frame_width: int, scale_hint: float | None) -> list[float]:
        available = self._candidate_scales(frame_width)
        if scale_hint is None:
            return available
        hint = float(scale_hint)
        neighbors = sorted(
            [scale for scale in available if abs(scale - hint) <= 0.12],
            key=lambda scale: abs(scale - hint),
        )
        if neighbors:
            return neighbors
        return [min(available, key=lambda scale: abs(scale - hint))]

    def _finalize_accepted(self, candidates: list[SimpleCandidate]) -> list[SimpleCandidate]:
        candidates.sort(key=lambda c: c.final_score, reverse=True)
        return self._nms([candidate for candidate in candidates if candidate.accepted])

    @staticmethod
    def _living_candidate(
        mob_name: str,
        cx: int,
        cy: int,
        bbox: tuple[int, int, int, int],
        score: RegionScore,
        heat_score: float,
        candidate_scale: float,
    ) -> SimpleCandidate:
        return SimpleCandidate(
            mob_name=mob_name,
            center_x=cx,
            center_y=cy,
            bbox=bbox,
            final_score=score.final_score,
            body_palette_score=score.body_palette_score,
            accent_score=score.accent_score,
            rare_color_score=score.rare_color_score,
            local_pattern_score=score.local_pattern_score,
            color_purity_score=score.color_purity_score,
            size_score=score.size_score,
            candidate_scale=candidate_scale,
            accepted=score.accepted,
            rejection_reason=score.rejection_reason,
            heatmap_score=heat_score,
        )

    def _nms(self, candidates: list[SimpleCandidate]) -> list[SimpleCandidate]:
        kept: list[SimpleCandidate] = []
        min_dist = int(self.config["nmsDistancePx"])
        min_dist_sq = min_dist * min_dist
        for candidate in sorted(candidates, key=lambda c: c.final_score, reverse=True):
            if all(
                (candidate.center_x - kept_candidate.center_x) ** 2
                + (candidate.center_y - kept_candidate.center_y) ** 2
                >= min_dist_sq
                for kept_candidate in kept
            ):
                kept.append(candidate)
        return kept
