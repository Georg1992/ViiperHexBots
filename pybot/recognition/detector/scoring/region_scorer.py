"""Fixed-window region scoring for the mob detector."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from pybot.recognition.detector.descriptors.descriptor import MobDescriptor
from pybot.recognition.detector.scoring.heatmap_detector import palette_heatmap, sprite_palette_heatmap


@dataclass
class RegionScore:
    final_score: float
    body_palette_score: float
    accent_score: float
    rare_color_score: float
    local_pattern_score: float
    color_purity_score: float
    size_score: float
    accepted: bool
    rejection_reason: str


class RegionScorer:
    def __init__(self, config: dict):
        self.min_color_purity = float(config["minColorPurity"])
        self.min_descriptor_color_match = float(config["minDescriptorColorMatch"])
        self.max_sprite_palette_distance = float(config["maxSpritePaletteDistance"])
        self.min_sprite_palette_match = float(config["minSpritePaletteMatch"])
        self.max_rare_to_body_ratio = float(config["maxRareToBodyRatio"])
        self.min_informative_fraction = float(config["minInformativePixelFraction"])
        self.min_discovery_size_score = float(config["minDiscoverySizeScore"])
        self.min_object_size_score = float(config["minObjectSizeScore"])
        self.enforce_object_size_gate = bool(config["enforceObjectSizeGate"])
        self.min_body_palette_score = float(config["minBodyPaletteScore"])
        self.min_accent_score = float(config["minAccentScore"])
        self.min_local_pattern_score = float(config["minLocalPatternScore"])
        self.min_histogram_correlation = float(config["minHistogramCorrelation"])

    def score(
        self,
        frame_bgr: np.ndarray,
        hsv: np.ndarray,
        descriptor: MobDescriptor,
        bbox: tuple[int, int, int, int],
        expected_scale: float = 1.0,
    ) -> RegionScore:
        x, y, w, h = bbox
        region_bgr = frame_bgr[y : y + h, x : x + w]
        region_hsv = hsv[y : y + h, x : x + w]
        if region_bgr.size == 0:
            return RegionScore(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False, "empty_region")

        sprite_palette_heat = sprite_palette_heatmap(
            region_bgr,
            descriptor.match_palette_bgr,
            self.max_sprite_palette_distance,
        )
        body_heat = palette_heatmap(region_hsv, descriptor.body_palette)
        accent_heat = palette_heatmap(region_hsv, descriptor.accent_colors)
        rare_heat = palette_heatmap(region_hsv, descriptor.rare_colors)

        sprite_score = self._top_match_score(sprite_palette_heat, 0.22)
        body = self._top_match_score(body_heat, 0.22)
        accent = self._top_match_score(accent_heat, 0.12)
        rare = self._top_match_score(rare_heat, 0.08)
        pattern = self._local_pattern(region_bgr, region_hsv, descriptor)
        sprite_pixels = sprite_palette_heat >= self.min_sprite_palette_match
        informative_fraction = float(sprite_pixels.mean()) if sprite_pixels.size else 0.0
        histogram = self._histogram_correlation(region_hsv, descriptor, sprite_pixels)
        size = self._object_size_score(sprite_palette_heat, descriptor, expected_scale)

        gates = []
        gates.append((sprite_score >= self.min_sprite_palette_match, "foreign_colors"))
        gates.append((body >= self.min_body_palette_score, "weak_body_palette"))
        gates.append((accent >= self.min_accent_score, "weak_accent"))
        gates.append((pattern >= self.min_local_pattern_score, "weak_pattern"))
        gates.append((histogram >= self.min_histogram_correlation, "histogram_mismatch"))
        gates.append((rare <= max(body * self.max_rare_to_body_ratio, 0.05), "rare_color_imbalance"))
        gates.append((informative_fraction >= self.min_informative_fraction, "insufficient_sprite_pixels"))
        gates.append((size >= self.min_discovery_size_score, "wrong_size"))
        if self.enforce_object_size_gate:
            gates.append((size >= self.min_object_size_score, "wrong_size"))

        accepted = all(gate[0] for gate in gates)
        rejection_reason = ""
        if not accepted:
            for gate_pass, gate_reason in gates:
                if not gate_pass:
                    rejection_reason = gate_reason
                    break

        return RegionScore(
            final_score=float(np.clip(sprite_score, 0.0, 1.0)),
            body_palette_score=body,
            accent_score=accent,
            rare_color_score=rare,
            local_pattern_score=pattern,
            color_purity_score=informative_fraction,
            size_score=size,
            accepted=accepted,
            rejection_reason=rejection_reason,
        )

    @staticmethod
    def _local_pattern(region_bgr: np.ndarray, region_hsv: np.ndarray, descriptor: MobDescriptor) -> float:
        accent = palette_heatmap(region_hsv, descriptor.accent_colors)
        body = palette_heatmap(region_hsv, descriptor.body_palette)
        gray = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2GRAY)
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        edge = cv2.magnitude(grad_x, grad_y)
        if float(edge.max()) > 0:
            edge = edge / float(edge.max())
        pattern = np.maximum(accent * 0.75, body * edge)
        return RegionScorer._top_match_score(pattern, 0.12)

    @staticmethod
    def _top_match_score(heatmap: np.ndarray, fraction: float) -> float:
        if heatmap.size == 0:
            return 0.0
        flat = heatmap.reshape(-1)
        keep = max(1, int(round(len(flat) * fraction)))
        top = np.partition(flat, len(flat) - keep)[-keep:]
        return float(np.clip(top.mean(), 0.0, 1.0))

    @staticmethod
    def _histogram_correlation(
        region_hsv: np.ndarray,
        descriptor: MobDescriptor,
        sprite_pixels: np.ndarray,
    ) -> float:
        if not descriptor.hsv_histogram or not np.any(sprite_pixels):
            return 0.0
        matched = region_hsv[sprite_pixels]
        if matched.size == 0:
            return 0.0
        strip = matched.astype(np.uint8).reshape(1, -1, 3)
        hist = cv2.calcHist([strip], [0, 1], None, [24, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        reference = np.asarray(descriptor.hsv_histogram, dtype=np.float32).reshape(24, 16)
        return float(cv2.compareHist(hist, reference, cv2.HISTCMP_CORREL))

    def _object_size_score(
        self,
        sprite_palette_heat: np.ndarray,
        descriptor: MobDescriptor,
        expected_scale: float,
    ) -> float:
        sprite_pixels = sprite_palette_heat >= self.min_sprite_palette_match
        if not np.any(sprite_pixels):
            return 0.0

        ys, xs = np.where(sprite_pixels)
        object_width = int(xs.max() - xs.min() + 1)
        object_height = int(ys.max() - ys.min() + 1)
        expected_width = max(1, descriptor.avg_width * expected_scale)
        expected_height = max(1, descriptor.avg_height * expected_scale)
        width_ratio = min(object_width, expected_width) / max(object_width, expected_width)
        height_ratio = min(object_height, expected_height) / max(object_height, expected_height)
        return float(np.sqrt(width_ratio * height_ratio))
