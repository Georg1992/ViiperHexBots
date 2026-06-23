"""Simple descriptor heatmap detector."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from descriptor import SimpleMobDescriptor
from descriptor_builder import SimpleDescriptorBuilder
from heatmap_detector import HeatmapDetector, Heatmaps
from region_scorer import RegionScore, SimpleRegionScorer


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
    size_score: float
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
            "sizeScore": round(self.size_score, 4),
            "heatmapScore": round(self.heatmap_score, 4),
            "accepted": self.accepted,
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
    return json.loads(config_path.read_text(encoding="utf-8"))


class SimpleMobDetector:
    def __init__(self, project_root: Path, config: Optional[dict] = None):
        self.project_root = project_root
        self.config = config or load_simple_config()
        self.heatmap_detector = HeatmapDetector(self.config)
        self.region_scorer = SimpleRegionScorer(self.config)
        self._descriptor_cache: dict[str, SimpleMobDescriptor] = {}

    def descriptor_path(self, mob_name: str) -> Path:
        return self.project_root / "generated_descriptors" / mob_name.lower() / "simple" / "descriptor.json"

    def ensure_descriptor(self, mob_name: str) -> SimpleMobDescriptor:
        mob_name = mob_name.lower()
        if mob_name in self._descriptor_cache:
            return self._descriptor_cache[mob_name]
        path = self.descriptor_path(mob_name)
        if not path.exists():
            SimpleDescriptorBuilder(self.project_root).build(mob_name)
        descriptor = SimpleMobDescriptor.load(path)
        self._descriptor_cache[mob_name] = descriptor
        return descriptor

    def detect(self, frame_bgr: np.ndarray, mob_name: str) -> SimpleDetectionResult:
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
            candidates.extend(self._score_center(frame_bgr, hsv, descriptor, cx, cy, heat_score))
        nms_start = time.perf_counter()
        candidates.sort(key=lambda c: c.final_score, reverse=True)
        accepted = self._nms([candidate for candidate in candidates if candidate.accepted])
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
            candidates=final_candidates[: int(self.config.get("maxCandidates", 100))],
            accepted=accepted,
            elapsed_s=elapsed,
            timing=timing,
        )

    def _score_center(
        self,
        frame_bgr: np.ndarray,
        hsv: np.ndarray,
        descriptor: SimpleMobDescriptor,
        cx: int,
        cy: int,
        heat_score: float,
    ) -> list[SimpleCandidate]:
        candidates: list[SimpleCandidate] = []
        scales = self.config.get("scales", [0.9, 1.0, 1.1])
        for scale in scales:
            w = max(8, int(round(descriptor.avg_width * float(scale))))
            h = max(8, int(round(descriptor.avg_height * float(scale))))
            x = int(round(cx - w / 2))
            y = int(round(cy - h / 2))
            if x < 0 or y < 0 or x + w > frame_bgr.shape[1] or y + h > frame_bgr.shape[0]:
                continue
            score = self.region_scorer.score(frame_bgr, hsv, descriptor, (x, y, w, h))
            candidates.append(self._candidate(descriptor.mob_name, cx, cy, (x, y, w, h), score, heat_score))
        if not candidates:
            return []
        return [max(candidates, key=lambda c: c.final_score)]

    @staticmethod
    def _candidate(
        mob_name: str,
        cx: int,
        cy: int,
        bbox: tuple[int, int, int, int],
        score: RegionScore,
        heat_score: float,
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
            size_score=score.size_score,
            accepted=score.accepted,
            rejection_reason=score.rejection_reason,
            heatmap_score=heat_score,
        )

    def _nms(self, candidates: list[SimpleCandidate]) -> list[SimpleCandidate]:
        kept: list[SimpleCandidate] = []
        min_dist = int(self.config.get("nmsDistancePx", 35))
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
