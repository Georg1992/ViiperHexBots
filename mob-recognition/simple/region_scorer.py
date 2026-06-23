"""Fixed-window region scoring for the simple detector."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from descriptor import SimpleMobDescriptor
from heatmap_detector import palette_heatmap


@dataclass
class RegionScore:
    final_score: float
    body_palette_score: float
    accent_score: float
    rare_color_score: float
    local_pattern_score: float
    size_score: float
    accepted: bool
    rejection_reason: str


class SimpleRegionScorer:
    def __init__(self, config: dict):
        self.threshold = float(config.get("acceptThreshold", 0.46))
        weights = config.get("weights", {})
        self.weights = {
            "body": float(weights.get("bodyPalette", 0.30)),
            "accent": float(weights.get("accent", 0.35)),
            "rare": float(weights.get("rareColor", 0.10)),
            "pattern": float(weights.get("localPattern", 0.15)),
            "size": float(weights.get("size", 0.10)),
        }

    def score(
        self,
        frame_bgr: np.ndarray,
        hsv: np.ndarray,
        descriptor: SimpleMobDescriptor,
        bbox: tuple[int, int, int, int],
    ) -> RegionScore:
        x, y, w, h = bbox
        region_bgr = frame_bgr[y : y + h, x : x + w]
        region_hsv = hsv[y : y + h, x : x + w]
        if region_bgr.size == 0:
            return RegionScore(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False, "empty_region")

        body = self._top_match_score(palette_heatmap(region_hsv, descriptor.body_colors), 0.22)
        accent = self._top_match_score(palette_heatmap(region_hsv, descriptor.accent_colors), 0.12)
        rare = self._top_match_score(palette_heatmap(region_hsv, descriptor.rare_colors), 0.08)
        pattern = self._local_pattern(region_bgr, region_hsv, descriptor)
        size = self._size_score(w, h, descriptor)
        final = (
            self.weights["body"] * body
            + self.weights["accent"] * accent
            + self.weights["rare"] * rare
            + self.weights["pattern"] * pattern
            + self.weights["size"] * size
        )
        accepted = final >= self.threshold
        return RegionScore(
            final_score=float(np.clip(final, 0.0, 1.0)),
            body_palette_score=body,
            accent_score=accent,
            rare_color_score=rare,
            local_pattern_score=pattern,
            size_score=size,
            accepted=accepted,
            rejection_reason="" if accepted else "below_threshold",
        )

    @staticmethod
    def _local_pattern(region_bgr: np.ndarray, region_hsv: np.ndarray, descriptor: SimpleMobDescriptor) -> float:
        accent = palette_heatmap(region_hsv, descriptor.accent_colors)
        body = palette_heatmap(region_hsv, descriptor.body_colors)
        gray = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2GRAY)
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        edge = cv2.magnitude(grad_x, grad_y)
        if float(edge.max()) > 0:
            edge = edge / float(edge.max())
        pattern = np.maximum(accent * 0.75, body * edge)
        return SimpleRegionScorer._top_match_score(pattern, 0.12)

    @staticmethod
    def _top_match_score(heatmap: np.ndarray, fraction: float) -> float:
        if heatmap.size == 0:
            return 0.0
        flat = heatmap.reshape(-1)
        keep = max(1, int(round(len(flat) * fraction)))
        top = np.partition(flat, len(flat) - keep)[-keep:]
        return float(np.clip(top.mean(), 0.0, 1.0))

    @staticmethod
    def _size_score(w: int, h: int, descriptor: SimpleMobDescriptor) -> float:
        width_ratio = min(w, descriptor.avg_width) / max(w, descriptor.avg_width, 1)
        height_ratio = min(h, descriptor.avg_height) / max(h, descriptor.avg_height, 1)
        return float(np.sqrt(width_ratio * height_ratio))
