"""Simple descriptor heatmap detector."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from death_validator import DeathValidation, DeathValidator
from descriptor import SimpleMobDescriptor
from descriptor_builder import DESCRIPTOR_VERSION
from heatmap_detector import HeatmapDetector, Heatmaps
from region_scorer import RegionScore, SimpleRegionScorer


REQUIRED_CONFIG_KEYS = {
    "acceptThreshold",
    "minColorPurity",
    "minBodyPaletteScore",
    "minAccentScore",
    "minLocalPatternScore",
    "minDiscoveryHeatmapScore",
    "watchDriftRadiusPx",
    "watchDriftStepPx",
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
    "weights",
    "centerWeights",
    "deadValidationThreshold",
    "deadWatchPointThreshold",
    "minDeadMobPresence",
    "maxFullOpacity",
    "minOpacitySpriteFraction",
    "minOpacitySamplePixels",
    "deadValidationWeights",
}

REQUIRED_DEAD_VALIDATION_WEIGHT_KEYS = {"pose", "sizeGap", "histogram", "opacity"}

REQUIRED_WEIGHT_KEYS = {"bodyPalette", "accent", "rareColor", "localPattern", "colorPurity", "size"}
REQUIRED_CENTER_WEIGHT_KEYS = {"bodyPalette", "accent", "rareColor", "localPattern"}


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
    is_dead: bool
    dead_score: float
    mean_opacity: float
    opacity_confirmed: bool
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
            "dead": self.is_dead,
            "living": self.accepted and not self.is_dead,
            "deadScore": round(self.dead_score, 4),
            "meanOpacity": round(self.mean_opacity, 4),
            "opacityConfirmed": self.opacity_confirmed,
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
    weight_missing = sorted(REQUIRED_WEIGHT_KEYS - set(config["weights"]))
    if weight_missing:
        raise ValueError(f"missing detector weight keys: {', '.join(weight_missing)}")
    center_weight_missing = sorted(REQUIRED_CENTER_WEIGHT_KEYS - set(config["centerWeights"]))
    if center_weight_missing:
        raise ValueError(f"missing center weight keys: {', '.join(center_weight_missing)}")
    dead_weight_missing = sorted(REQUIRED_DEAD_VALIDATION_WEIGHT_KEYS - set(config["deadValidationWeights"]))
    if dead_weight_missing:
        raise ValueError(f"missing dead validation weight keys: {', '.join(dead_weight_missing)}")
    return config


class SimpleMobDetector:
    def __init__(self, project_root: Path, config: Optional[dict] = None):
        self.project_root = project_root
        self.config = load_simple_config() if config is None else config
        self.heatmap_detector = HeatmapDetector(self.config)
        self.region_scorer = SimpleRegionScorer(self.config)
        self.death_validator = DeathValidator(self.config, self.region_scorer)
        self._descriptor_cache: dict[str, SimpleMobDescriptor] = {}
        self.self_exclusion_width_ratio = float(self.config["selfExclusionWidthRatio"])
        self.self_exclusion_height_ratio = float(self.config["selfExclusionHeightRatio"])
        self.small_scale_min_frame_width = int(self.config["smallScaleMinFrameWidth"])
        self.min_discovery_heatmap_score = float(self.config["minDiscoveryHeatmapScore"])
        self.watch_drift_radius_px = int(self.config["watchDriftRadiusPx"])
        self.watch_drift_step_px = int(self.config["watchDriftStepPx"])

    def apply_runtime_config(self, config: dict) -> None:
        prior = self.config
        self.config = dict(config)
        scale_keys = ("scales", "centerScales")
        if any(prior.get(key) != self.config.get(key) for key in scale_keys):
            self.heatmap_detector = HeatmapDetector(self.config)
        self.region_scorer = SimpleRegionScorer(self.config)
        self.death_validator = DeathValidator(self.config, self.region_scorer)
        self.self_exclusion_width_ratio = float(self.config["selfExclusionWidthRatio"])
        self.self_exclusion_height_ratio = float(self.config["selfExclusionHeightRatio"])
        self.small_scale_min_frame_width = int(self.config["smallScaleMinFrameWidth"])
        self.min_discovery_heatmap_score = float(self.config["minDiscoveryHeatmapScore"])
        self.watch_drift_radius_px = int(self.config["watchDriftRadiusPx"])
        self.watch_drift_step_px = int(self.config["watchDriftStepPx"])

    def descriptor_path(self, mob_name: str) -> Path:
        return self.project_root / "generated_descriptors" / mob_name.lower() / "simple" / "descriptor.json"

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
        heatmaps = self.heatmap_detector.build_heatmaps(frame_bgr, hsv, descriptor)
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
        """Discovery scan: shared point scorer; living only when it beats corpse signal."""
        if self._is_self_center(cx, cy, frame_bgr.shape):
            return []
        if heat_score < self.min_discovery_heatmap_score:
            return []

        scales = self._candidate_scales(frame_bgr.shape[1], track_point=False)
        living, dead = self._score_point_at(frame_bgr, hsv, descriptor, cx, cy, scales=scales)
        if dead and not living:
            return []
        if living and living.accepted:
            if dead and dead.final_score >= living.final_score:
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
    ) -> tuple[SimpleCandidate | None, SimpleCandidate | None]:
        """Score living and corpse signals at one point. Returns (living, dead) candidates."""
        if cx < 0 or cy < 0 or cx >= frame_bgr.shape[1] or cy >= frame_bgr.shape[0]:
            return None, None
        if descriptor.dead is None:
            return None, None

        best_living: tuple[float, tuple[int, int, int, int], tuple[int, int, int, int], RegionScore, DeathValidation] | None = None
        best_dead: tuple[float, tuple[int, int, int, int], RegionScore | None, DeathValidation] | None = None
        presence_gate = self.death_validator.min_mob_presence * 0.55

        if scales is None:
            scales = self._candidate_scales(frame_bgr.shape[1], track_point=True)

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

            dead_bbox = self._bbox_for_size(
                cx,
                cy,
                int(round(descriptor.dead.size.avg_width * scale)),
                int(round(descriptor.dead.size.avg_height * scale)),
                frame_bgr.shape,
            )
            if dead_bbox is None:
                continue

            dead_score = self.death_validator.score_dead_region(frame_bgr, hsv, descriptor, dead_bbox, float(scale))
            living_at_dead_score = self.region_scorer.score(
                frame_bgr, hsv, descriptor, dead_bbox, expected_scale=float(scale)
            )
            mob_signal = max(
                living_score.final_score,
                living_at_dead_score.final_score,
                dead_score.final_score if dead_score is not None else 0.0,
            )
            if mob_signal < presence_gate:
                continue

            validation = self.death_validator.validate(
                frame_bgr,
                hsv,
                descriptor,
                living_bbox,
                dead_bbox,
                living_score,
                dead_score,
                living_at_dead_score,
                watch_point=True,
            )

            if validation.is_dead and (
                best_dead is None or validation.confidence > best_dead[3].confidence
            ):
                best_dead = (float(scale), dead_bbox, dead_score, validation)
            elif living_score.accepted and (
                best_living is None or living_score.final_score > best_living[3].final_score
            ):
                best_living = (float(scale), living_bbox, dead_bbox, living_score, validation)

        dead_candidate: SimpleCandidate | None = None
        if best_dead is not None:
            scale, bbox, score, validation = best_dead
            region_score = score if score is not None else RegionScore(0, 0, 0, 0, 0, 0, 0, False, "missing_dead_score")
            dead_candidate = self._track_candidate(
                descriptor.mob_name,
                cx,
                cy,
                bbox,
                region_score,
                scale,
                is_dead=True,
                death_validation=validation,
            )

        living_candidate: SimpleCandidate | None = None
        if best_living is not None:
            scale, living_bbox, dead_bbox, score, validation = best_living
            bx, by, bw, bh = living_bbox
            living_candidate = self._track_candidate(
                descriptor.mob_name,
                bx + bw // 2,
                by + bh // 2,
                living_bbox,
                score,
                scale,
                is_dead=False,
                death_validation=validation,
            )

        return living_candidate, dead_candidate

    def _evaluate_track_point(
        self,
        frame_bgr: np.ndarray,
        hsv: np.ndarray,
        descriptor: SimpleMobDescriptor,
        cx: int,
        cy: int,
        scales: list[float] | None = None,
    ) -> list[SimpleCandidate]:
        """Track state: death validation and living position at a known track point."""
        living, dead = self._score_point_at(frame_bgr, hsv, descriptor, cx, cy, scales=scales)
        if dead:
            return [dead]
        if living:
            return [living]
        return []

    def evaluate_track_states(
        self,
        frame_bgr: np.ndarray,
        mob_name: str,
        tracks: list[dict],
        *,
        offset_x: int = 0,
        offset_y: int = 0,
    ) -> list[dict]:
        """Evaluate known tracks by id (ROI-local x,y). Returns trackUpdates with screen coordinates."""
        if not tracks:
            return []

        descriptor = self.ensure_descriptor(mob_name)
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        updates: list[dict] = []

        for track in tracks:
            track_id = int(track["trackId"])
            cx = int(track["x"])
            cy = int(track["y"])
            point_candidates: list[SimpleCandidate] = []
            for center_x, center_y in self._track_search_centers(cx, cy, frame_bgr.shape):
                point_candidates.extend(
                    self._evaluate_track_point(frame_bgr, hsv, descriptor, center_x, center_y)
                )
            best = self._best_track_point_candidate(point_candidates)
            if best is None:
                updates.append(
                    {
                        "trackId": track_id,
                        "state": "gone",
                        "confidence": 0.0,
                        "x": cx + offset_x,
                        "y": cy + offset_y,
                    }
                )
            elif best.is_dead:
                updates.append(
                    {
                        "trackId": track_id,
                        "state": "dead",
                        "confidence": round(best.final_score, 4),
                        "x": best.center_x + offset_x,
                        "y": best.center_y + offset_y,
                    }
                )
            else:
                updates.append(
                    {
                        "trackId": track_id,
                        "state": "alive",
                        "confidence": round(best.final_score, 4),
                        "x": best.center_x + offset_x,
                        "y": best.center_y + offset_y,
                    }
                )

        return updates

    def _direct_track_scale(self, frame_width: int) -> float:
        track_scales = self._candidate_scales(frame_width, track_point=True)
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
    ) -> dict:
        """Fast post-attack state: one center point, single scale, no drift."""
        descriptor = self.ensure_descriptor(mob_name)
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        scale = self._direct_track_scale(frame_bgr.shape[1])
        point_candidates = self._evaluate_track_point(
            frame_bgr, hsv, descriptor, cx, cy, scales=[scale]
        )
        best = self._best_track_point_candidate(point_candidates)
        if best is None:
            return {
                "trackId": track_id,
                "state": "unknown",
                "confidence": 0.0,
                "x": cx + offset_x,
                "y": cy + offset_y,
            }
        if best.is_dead:
            return {
                "trackId": track_id,
                "state": "dead",
                "confidence": round(best.final_score, 4),
                "x": best.center_x + offset_x,
                "y": best.center_y + offset_y,
            }
        return {
            "trackId": track_id,
            "state": "alive",
            "confidence": round(best.final_score, 4),
            "x": best.center_x + offset_x,
            "y": best.center_y + offset_y,
        }

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

    def _track_search_centers(self, cx: int, cy: int, frame_shape: tuple[int, ...]) -> list[tuple[int, int]]:
        radius = self.watch_drift_radius_px
        step = max(8, self.watch_drift_step_px)
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

    def _best_track_point_candidate(self, candidates: list[SimpleCandidate]) -> SimpleCandidate | None:
        if not candidates:
            return None
        dead_candidates = [candidate for candidate in candidates if candidate.is_dead]
        if dead_candidates:
            return max(dead_candidates, key=lambda candidate: candidate.final_score)
        living_candidates = [candidate for candidate in candidates if not candidate.is_dead]
        if living_candidates:
            return max(living_candidates, key=lambda candidate: candidate.final_score)
        return None

    def _is_self_center(self, cx: int, cy: int, frame_shape: tuple[int, ...]) -> bool:
        height, width = frame_shape[:2]
        half_width = width * self.self_exclusion_width_ratio * 0.5
        half_height = height * self.self_exclusion_height_ratio * 0.5
        return abs(cx - width / 2) <= half_width and abs(cy - height / 2) <= half_height

    def _candidate_scales(self, frame_width: int, *, track_point: bool = False) -> list[float]:
        if track_point:
            return [0.45, 0.55, 0.65]
        return [
            float(scale)
            for scale in self.config["scales"]
            if float(scale) >= 0.75 or frame_width >= self.small_scale_min_frame_width
        ]

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
            is_dead=False,
            dead_score=0.0,
            mean_opacity=1.0,
            opacity_confirmed=False,
            rejection_reason=score.rejection_reason,
            heatmap_score=heat_score,
        )

    @staticmethod
    def _track_candidate(
        mob_name: str,
        cx: int,
        cy: int,
        bbox: tuple[int, int, int, int],
        score: RegionScore,
        candidate_scale: float,
        *,
        is_dead: bool,
        death_validation: DeathValidation,
    ) -> SimpleCandidate:
        region_score = score if score is not None else RegionScore(0, 0, 0, 0, 0, 0, 0, False, "missing_score")
        accepted = (not is_dead) or death_validation.is_dead
        return SimpleCandidate(
            mob_name=mob_name,
            center_x=cx,
            center_y=cy,
            bbox=bbox,
            final_score=death_validation.confidence if is_dead else region_score.final_score,
            body_palette_score=region_score.body_palette_score,
            accent_score=region_score.accent_score,
            rare_color_score=region_score.rare_color_score,
            local_pattern_score=region_score.local_pattern_score,
            color_purity_score=region_score.color_purity_score,
            size_score=region_score.size_score,
            candidate_scale=candidate_scale,
            accepted=accepted,
            is_dead=is_dead,
            dead_score=death_validation.confidence if is_dead else 0.0,
            mean_opacity=death_validation.mean_opacity,
            opacity_confirmed=death_validation.opacity_fade_score > 0.0,
            rejection_reason="" if is_dead else region_score.rejection_reason,
            heatmap_score=1.0,
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
