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
    "deadAttackSlotThreshold",
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
    heatmaps: Heatmaps
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
        attack_slots: list[tuple[int, int]] | None = None,
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
            candidates.extend(self._evaluate_center(frame_bgr, hsv, descriptor, cx, cy, heat_score))
        if attack_slots:
            for slot_x, slot_y in attack_slots:
                candidates.extend(
                    self._evaluate_center(frame_bgr, hsv, descriptor, slot_x, slot_y, 1.0, attack_slot=True)
                )
        nms_start = time.perf_counter()
        candidates.sort(key=lambda c: c.final_score, reverse=True)
        accepted = self._nms([candidate for candidate in candidates if candidate.accepted])
        if attack_slots:
            for candidate in candidates:
                if candidate.accepted and candidate.is_dead and candidate not in accepted:
                    accepted.append(candidate)
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

    def _evaluate_center(
        self,
        frame_bgr: np.ndarray,
        hsv: np.ndarray,
        descriptor: SimpleMobDescriptor,
        cx: int,
        cy: int,
        heat_score: float,
        *,
        attack_slot: bool = False,
    ) -> list[SimpleCandidate]:
        if not attack_slot and self._is_self_center(cx, cy, frame_bgr.shape):
            return []
        if attack_slot and (cx < 0 or cy < 0 or cx >= frame_bgr.shape[1] or cy >= frame_bgr.shape[0]):
            return []
        if attack_slot and descriptor.dead is None:
            return []

        best_living: tuple[float, tuple[int, int, int, int], RegionScore, DeathValidation] | None = None
        best_dead: tuple[float, tuple[int, int, int, int], RegionScore | None, DeathValidation] | None = None
        presence_gate = self.death_validator.min_mob_presence * (0.55 if attack_slot else 0.5)

        for scale in self._candidate_scales(frame_bgr.shape[1]):
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
            if descriptor.dead is None:
                if living_score.accepted and (
                    best_living is None or living_score.final_score > best_living[2].final_score
                ):
                    empty_validation = DeathValidation(False, 0.0, living_score.final_score, 0.0, 0.0, 0.0, 0.0, 1.0)
                    best_living = (float(scale), living_bbox, living_score, empty_validation)
                continue

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
                attack_slot=attack_slot,
            )

            if validation.is_dead and (best_dead is None or validation.confidence > best_dead[3].confidence):
                best_dead = (float(scale), dead_bbox, dead_score, validation)
            elif living_score.accepted and not validation.is_dead and (
                best_living is None or living_score.final_score > best_living[2].final_score
            ):
                best_living = (float(scale), living_bbox, living_score, validation)

        if best_dead is not None:
            scale, bbox, score, validation = best_dead
            region_score = score if score is not None else RegionScore(0, 0, 0, 0, 0, 0, 0, False, "missing_dead_score")
            return [
                self._candidate(
                    descriptor.mob_name,
                    cx,
                    cy,
                    bbox,
                    region_score,
                    heat_score,
                    scale,
                    is_dead=True,
                    death_validation=validation,
                )
            ]
        if best_living is not None:
            scale, bbox, score, validation = best_living
            return [
                self._candidate(
                    descriptor.mob_name,
                    cx,
                    cy,
                    bbox,
                    score,
                    heat_score,
                    scale,
                    is_dead=False,
                    death_validation=validation,
                )
            ]
        return []

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

    def _is_self_center(self, cx: int, cy: int, frame_shape: tuple[int, ...]) -> bool:
        height, width = frame_shape[:2]
        half_width = width * self.self_exclusion_width_ratio * 0.5
        half_height = height * self.self_exclusion_height_ratio * 0.5
        return abs(cx - width / 2) <= half_width and abs(cy - height / 2) <= half_height

    def _candidate_scales(self, frame_width: int) -> list[float]:
        return [
            float(scale)
            for scale in self.config["scales"]
            if float(scale) >= 0.75 or frame_width >= self.small_scale_min_frame_width
        ]

    @staticmethod
    def _candidate(
        mob_name: str,
        cx: int,
        cy: int,
        bbox: tuple[int, int, int, int],
        score: RegionScore,
        heat_score: float,
        candidate_scale: float,
        *,
        is_dead: bool,
        death_validation: DeathValidation,
    ) -> SimpleCandidate:
        region_score = score if score is not None else RegionScore(0, 0, 0, 0, 0, 0, 0, False, "missing_score")
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
            accepted=True,
            is_dead=is_dead,
            dead_score=death_validation.confidence if is_dead else 0.0,
            mean_opacity=death_validation.mean_opacity,
            opacity_confirmed=death_validation.opacity_fade_score > 0.0,
            rejection_reason="" if is_dead else region_score.rejection_reason,
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
