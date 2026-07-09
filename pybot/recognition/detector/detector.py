"""Simple descriptor heatmap detector."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from pybot.recognition.detector.descriptors.descriptor import MobDescriptor
from pybot.recognition.detector.descriptors.descriptor_builder import DESCRIPTOR_VERSION
from pybot.recognition.detector.scoring.heatmap_detector import HeatmapDetector, Heatmaps
from pybot.recognition.detector.scoring.region_scorer import RegionScore, RegionScorer


REQUIRED_CONFIG_KEYS = {
    "minColorPurity",
    "minBodyPaletteScore",
    "minAccentScore",
    "minLocalPatternScore",
    "minHistogramCorrelation",
    "minDiscoveryHeatmapScore",
    "discoveryHeatmapDownscale",
    "discoveryHeatmapDownscaleMinSide",
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
    "peakRelativeThreshold",
    "nmsDistancePx",
    "matchRadiusPx",
    "discoveryClusterRadiusPx",
    "trackDedupRadiusPx",
    "maxCandidates",
    "smallScaleMinFrameWidth",
    "smallScaleCutoff",
    "centerScales",
    "scales",
    "playfieldTopRatio",
    "playfieldBottomRatio",
    "playfieldLeftRatio",
    "playfieldRightRatio",
    "debugOutputDir",
    "centerWeights",
    "structuralPixelDistance",
    "accentStructuralPixelDistance",
    "structuralDiscoveryMinScore",
    "minStructuralAcceptScore",
    "minDominantPixelFraction",
    "minAccentPixelFraction",
    "deathOpacityBaselineSamples",
    "deathOpacityMinBaseline",
    "deathOpacityDecayRatio",
    "deathOpacityConfirmTicks",
    "deathRediscoveryCooldownMs",
    "deathOpacityMoveThresholdPx",
    "deathOpacityStopThresholdPx",
    "deathOpacityMinTrackAgeMs",
    "defaultAverageAttacksTillDeath",
    "attacksTillDeathHistoryWindow",
}

REQUIRED_CENTER_WEIGHT_KEYS = {"bodyPalette", "accent", "rareColor", "localPattern"}


@dataclass
class DetectionCandidate:
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
class DetectionResult:
    mob_name: str
    descriptor: MobDescriptor
    heatmaps: Heatmaps | None
    candidates: list[DetectionCandidate]
    accepted: list[DetectionCandidate]
    elapsed_s: float
    timing: dict[str, float]


def load_detector_config(path: Optional[Path] = None) -> dict:
    config_path = path or (Path(__file__).resolve().parent / "detector_config.json")
    import json

    config = json.loads(config_path.read_text(encoding="utf-8"))
    missing = sorted(REQUIRED_CONFIG_KEYS - set(config))
    if missing:
        raise ValueError(f"missing detector config keys: {', '.join(missing)}")
    center_weight_missing = sorted(REQUIRED_CENTER_WEIGHT_KEYS - set(config["centerWeights"]))
    if center_weight_missing:
        raise ValueError(f"missing center weight keys: {', '.join(center_weight_missing)}")
    return config


class MobDetector:
    def __init__(
        self,
        project_root: Path,
        config: Optional[dict] = None,
        *,
        use_modified_descriptor: bool = False,
    ):
        self.project_root = project_root
        self.use_modified_descriptor = use_modified_descriptor
        self.config = load_detector_config() if config is None else config
        self.heatmap_detector = HeatmapDetector(self.config)
        self.region_scorer = RegionScorer(self.config)
        self._descriptor_cache: dict[str, MobDescriptor] = {}
        self.small_scale_min_frame_width = int(self.config["smallScaleMinFrameWidth"])
        self.small_scale_cutoff = float(self.config["smallScaleCutoff"])
        self.min_discovery_heatmap_score = float(self.config["minDiscoveryHeatmapScore"])
        self.discovery_heatmap_downscale = int(self.config["discoveryHeatmapDownscale"])
        self.discovery_heatmap_downscale_min_side = int(self.config["discoveryHeatmapDownscaleMinSide"])
        self.local_track_search_radius_px = int(self.config["localTrackSearchRadiusPx"])
        self.structural_discovery_min_score = float(self.config["structuralDiscoveryMinScore"])
        self.min_structural_accept_score = float(self.config["minStructuralAcceptScore"])

    def apply_runtime_config(self, config: dict) -> None:
        prior = self.config
        self.config = dict(config)
        scale_keys = ("scales", "centerScales")
        if any(prior.get(key) != self.config.get(key) for key in scale_keys):
            self.heatmap_detector = HeatmapDetector(self.config)
        self.region_scorer = RegionScorer(self.config)
        self.small_scale_min_frame_width = int(self.config["smallScaleMinFrameWidth"])
        self.small_scale_cutoff = float(self.config["smallScaleCutoff"])
        self.min_discovery_heatmap_score = float(self.config["minDiscoveryHeatmapScore"])
        self.discovery_heatmap_downscale = int(self.config["discoveryHeatmapDownscale"])
        self.discovery_heatmap_downscale_min_side = int(self.config["discoveryHeatmapDownscaleMinSide"])
        self.local_track_search_radius_px = int(self.config["localTrackSearchRadiusPx"])
        self.structural_discovery_min_score = float(self.config["structuralDiscoveryMinScore"])
        self.min_structural_accept_score = float(self.config["minStructuralAcceptScore"])

    def descriptor_path(self, mob_name: str) -> Path:
        base = self.project_root / "assets" / "generated_descriptors"
        stem = mob_name.lower()
        if self.use_modified_descriptor:
            return base / "modified" / stem / "descriptor.json"
        return base / stem / "descriptor.json"

    def ensure_descriptor(self, mob_name: str) -> MobDescriptor:
        mob_name = mob_name.lower()
        if mob_name in self._descriptor_cache:
            return self._descriptor_cache[mob_name]
        path = self.descriptor_path(mob_name)
        if not path.exists():
            raise FileNotFoundError(f"descriptor not found for mob '{mob_name}': {path}")
        descriptor = MobDescriptor.load(path)
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
    ) -> DetectionResult:
        start = time.perf_counter()
        descriptor = self.ensure_descriptor(mob_name)
        hsv_start = time.perf_counter()
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        heatmap_start = time.perf_counter()
        frame_height, frame_width = frame_bgr.shape[:2]
        downscale = 1
        if self.discovery_heatmap_downscale > 1 and min(frame_width, frame_height) >= 1600:
            downscale = self.discovery_heatmap_downscale
        heatmaps = self.heatmap_detector.build_heatmaps(
            frame_bgr,
            None,
            descriptor,
            downscale=downscale,
            discovery_only=True,
        )
        centers_start = time.perf_counter()
        sprite_centers, structural_centers = self._discovery_centers(heatmaps)
        score_start = time.perf_counter()
        candidates: list[DetectionCandidate] = []
        for cx, cy, heat_score in sprite_centers:
            candidates.extend(
                self._evaluate_discovery_center(
                    frame_bgr,
                    hsv,
                    descriptor,
                    cx,
                    cy,
                    heat_score,
                    min_heatmap_score=self.min_discovery_heatmap_score,
                )
            )
        for cx, cy, heat_score in structural_centers:
            candidates.extend(
                self._evaluate_discovery_center(
                    frame_bgr,
                    hsv,
                    descriptor,
                    cx,
                    cy,
                    heat_score,
                    min_heatmap_score=self.structural_discovery_min_score,
                )
            )
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
        return DetectionResult(
            mob_name=mob_name.lower(),
            descriptor=descriptor,
            heatmaps=heatmaps,
            candidates=final_candidates[: int(self.config["maxCandidates"])],
            accepted=accepted,
            elapsed_s=elapsed,
            timing=timing,
        )

    def _discovery_centers(
        self,
        heatmaps: Heatmaps,
    ) -> tuple[list[tuple[int, int, float]], list[tuple[int, int, float]]]:
        offset_x, offset_y = heatmaps.playfield_offset
        sprite_centers = self._offset_centers(
            self.heatmap_detector.top_centers(heatmaps.final_center),
            offset_x,
            offset_y,
        )
        if float(heatmaps.structural_center.max()) <= 0.0:
            return sprite_centers, []

        saved_min_heat = self.heatmap_detector.min_center_heat
        saved_peak_threshold = self.heatmap_detector.peak_relative_threshold
        self.heatmap_detector.min_center_heat = self.structural_discovery_min_score
        self.heatmap_detector.peak_relative_threshold = 0.10
        structural_centers = self._offset_centers(
            self.heatmap_detector.top_centers(heatmaps.structural_center),
            offset_x,
            offset_y,
        )
        self.heatmap_detector.min_center_heat = saved_min_heat
        self.heatmap_detector.peak_relative_threshold = saved_peak_threshold

        min_dist_sq = self.heatmap_detector.min_center_distance ** 2
        extra_structural: list[tuple[int, int, float]] = []
        for cx, cy, score in structural_centers:
            if all((cx - px) ** 2 + (cy - py) ** 2 >= min_dist_sq for px, py, _ in sprite_centers):
                extra_structural.append((cx, cy, score))
        extra_structural.sort(key=lambda item: item[2], reverse=True)
        return sprite_centers, extra_structural[: self.heatmap_detector.max_centers]

    @staticmethod
    def _offset_centers(
        centers: list[tuple[int, int, float]],
        offset_x: int,
        offset_y: int,
    ) -> list[tuple[int, int, float]]:
        if offset_x == 0 and offset_y == 0:
            return centers
        return [(cx + offset_x, cy + offset_y, heat) for cx, cy, heat in centers]

    def _evaluate_discovery_center(
        self,
        frame_bgr: np.ndarray,
        hsv: np.ndarray,
        descriptor: MobDescriptor,
        cx: int,
        cy: int,
        heat_score: float,
        min_heatmap_score: float | None = None,
    ) -> list[DetectionCandidate]:
        """Discovery scan: score one heatmap peak center. Returns accepted candidate or empty list."""
        threshold = self.min_discovery_heatmap_score if min_heatmap_score is None else min_heatmap_score
        if heat_score < threshold:
            return []

        scales = self._candidate_scales(frame_bgr.shape[1])
        living = self._score_point_at(frame_bgr, hsv, descriptor, cx, cy, scales=scales)
        if living and living.accepted:
            if (
                threshold == self.structural_discovery_min_score
                and living.final_score < self.min_structural_accept_score
            ):
                return []
            if not self._passes_structural_pixel_gate(frame_bgr, descriptor, living.bbox):
                return []
            living.heatmap_score = heat_score
            return [living]
        return []

    def _passes_structural_pixel_gate(
        self,
        frame_bgr: np.ndarray,
        descriptor: MobDescriptor,
        bbox: tuple[int, int, int, int],
    ) -> bool:
        x, y, w, h = bbox
        region = frame_bgr[y : y + h, x : x + w]
        if region.size == 0:
            return False

        distance = float(self.config["structuralPixelDistance"])
        if descriptor.dominant_pixel_bgr is not None:
            dominant = np.array(descriptor.dominant_pixel_bgr, dtype=np.float32).reshape(1, 1, 3)
            dominant_dist = np.sqrt(np.sum((region.astype(np.float32) - dominant) ** 2, axis=2))
            dominant_fraction = float(np.mean(dominant_dist <= distance))
            if dominant_fraction < float(self.config["minDominantPixelFraction"]):
                return False

        if descriptor.accent_pixel_bgr is not None:
            accent = np.array(descriptor.accent_pixel_bgr, dtype=np.float32).reshape(1, 1, 3)
            accent_distance = float(self.config["accentStructuralPixelDistance"])
            accent_dist = np.sqrt(np.sum((region.astype(np.float32) - accent) ** 2, axis=2))
            accent_fraction = float(np.mean(accent_dist <= accent_distance))
            if accent_fraction < float(self.config["minAccentPixelFraction"]):
                return False

        return True

    def _score_point_at(
        self,
        frame_bgr: np.ndarray,
        hsv: np.ndarray,
        descriptor: MobDescriptor,
        cx: int,
        cy: int,
        scales: list[float] | None = None,
    ) -> DetectionCandidate | None:
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
        descriptor: MobDescriptor,
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
        death_detection_enabled: bool = False,
    ):
        from pybot.recognition.detector.tracking.local_tracker import track_local as run_track_local

        return run_track_local(
            self,
            frame_bgr,
            mob_name,
            track,
            offset_x=offset_x,
            offset_y=offset_y,
            search_radius_px=search_radius_px,
            death_detection_enabled=death_detection_enabled,
        )

    def _direct_track_scale(self, frame_width: int, scale_hint: float | None = None) -> float:
        if scale_hint is not None:
            track_scales = self._scales_for_track(frame_width, float(scale_hint))
            return track_scales[len(track_scales) // 2]
        track_scales = self._candidate_scales(frame_width)
        return track_scales[len(track_scales) // 2]

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

    def _candidate_scales(self, frame_width: int) -> list[float]:
        scales = [
            float(scale)
            for scale in self.config["scales"]
            if float(scale) >= self.small_scale_cutoff or frame_width >= self.small_scale_min_frame_width
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

    def _finalize_accepted(self, candidates: list[DetectionCandidate]) -> list[DetectionCandidate]:
        accepted = [candidate for candidate in candidates if candidate.accepted]
        accepted.sort(key=lambda candidate: candidate.final_score, reverse=True)
        return self._nms(accepted)

    @staticmethod
    def _living_candidate(
        mob_name: str,
        cx: int,
        cy: int,
        bbox: tuple[int, int, int, int],
        score: RegionScore,
        heat_score: float,
        candidate_scale: float,
    ) -> DetectionCandidate:
        return DetectionCandidate(
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

    def _nms(self, candidates: list[DetectionCandidate]) -> list[DetectionCandidate]:
        kept: list[DetectionCandidate] = []
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
