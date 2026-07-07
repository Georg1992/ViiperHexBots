"""Fixed-window region scoring for the simple detector."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from descriptors.descriptor import SimpleMobDescriptor
from scoring.heatmap_detector import palette_heatmap


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


class SimpleRegionScorer:
    def __init__(self, config: dict):
        self.min_color_purity = float(config["minColorPurity"])
        self.min_descriptor_color_match = float(config["minDescriptorColorMatch"])
        self.max_sprite_palette_distance = float(config["maxSpritePaletteDistance"])
        self.min_sprite_palette_match = float(config["minSpritePaletteMatch"])
        self.max_rare_to_body_ratio = float(config["maxRareToBodyRatio"])
        self.min_informative_fraction = float(config["minInformativePixelFraction"])
        self.max_descriptor_pixel_fraction = float(config["maxDescriptorPixelFraction"])
        self.min_discovery_size_score = float(config["minDiscoverySizeScore"])
        self.min_object_size_score = float(config["minObjectSizeScore"])
        self.enforce_object_size_gate = bool(config["enforceObjectSizeGate"])
        self.min_body_palette_score = float(config["minBodyPaletteScore"])
        self.min_accent_score = float(config["minAccentScore"])
        self.min_local_pattern_score = float(config["minLocalPatternScore"])

    def score(
        self,
        frame_bgr: np.ndarray,
        hsv: np.ndarray,
        descriptor: SimpleMobDescriptor,
        bbox: tuple[int, int, int, int],
        expected_scale: float = 1.0,
    ) -> RegionScore:
        x, y, w, h = bbox
        region_bgr = frame_bgr[y : y + h, x : x + w]
        region_hsv = hsv[y : y + h, x : x + w]
        if region_bgr.size == 0:
            return RegionScore(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False, "empty_region")

        body_heat = palette_heatmap(region_hsv, descriptor.body_palette)
        accent_heat = palette_heatmap(region_hsv, descriptor.accent_colors)
        rare_heat = palette_heatmap(region_hsv, descriptor.rare_colors)
        descriptor_heat = np.maximum.reduce([body_heat, accent_heat, rare_heat])
        sprite_palette_heat = self._sprite_palette_heatmap(region_bgr, descriptor)

        body = self._top_match_score(body_heat, 0.22)
        accent = self._top_match_score(accent_heat, 0.12)
        rare = self._top_match_score(rare_heat, 0.08)
        pattern = self._local_pattern(region_bgr, region_hsv, descriptor)
        purity, informative_fraction, descriptor_fraction = self._color_purity(
            region_bgr,
            region_hsv,
            descriptor_heat,
            sprite_palette_heat,
        )
        size = self._object_size_score(descriptor_heat, descriptor, expected_scale)

        # Binary pass/fail gates — each property is checked independently
        gates = []
        gates.append((body >= self.min_body_palette_score, "weak_body_palette"))
        gates.append((accent >= self.min_accent_score, "weak_accent"))
        gates.append((pattern >= self.min_local_pattern_score, "weak_pattern"))
        gates.append((purity >= self.min_color_purity, "foreign_colors"))
        gates.append((rare <= max(body * self.max_rare_to_body_ratio, 0.05), "rare_color_imbalance"))
        gates.append((informative_fraction >= self.min_informative_fraction, "insufficient_sprite_pixels"))
        gates.append((descriptor_fraction <= self.max_descriptor_pixel_fraction, "too_much_descriptor_color"))
        gates.append((size >= self.min_discovery_size_score, "wrong_size"))
        if self.enforce_object_size_gate:
            gates.append((size >= self.min_object_size_score, "wrong_size"))

        accepted = all(gate[0] for gate in gates)
        rejection_reason = ""
        if not accepted:
            # Report the first failing gate
            for gate_pass, gate_reason in gates:
                if not gate_pass:
                    rejection_reason = gate_reason
                    break

        return RegionScore(
            final_score=float(np.clip(body, 0.0, 1.0)),
            body_palette_score=body,
            accent_score=accent,
            rare_color_score=rare,
            local_pattern_score=pattern,
            color_purity_score=purity,
            size_score=size,
            accepted=accepted,
            rejection_reason=rejection_reason,
        )

    @staticmethod
    def _local_pattern(region_bgr: np.ndarray, region_hsv: np.ndarray, descriptor: SimpleMobDescriptor) -> float:
        accent = palette_heatmap(region_hsv, descriptor.accent_colors)
        body = palette_heatmap(region_hsv, descriptor.body_palette)
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

    def _color_purity(
        self,
        region_bgr: np.ndarray,
        region_hsv: np.ndarray,
        descriptor_heat: np.ndarray,
        sprite_palette_heat: np.ndarray,
    ) -> tuple[float, float, float]:
        gray = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2GRAY)
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        edge = cv2.magnitude(grad_x, grad_y)
        if float(edge.max()) > 0:
            edge = edge / float(edge.max())

        saturation = region_hsv[:, :, 1].astype(np.float32)
        value = region_hsv[:, :, 2].astype(np.float32)
        descriptor_pixels = descriptor_heat >= self.min_descriptor_color_match
        if not np.any(descriptor_pixels):
            return 0.0, 0.0, 0.0
        descriptor_fraction = float(descriptor_pixels.mean())

        object_zone = cv2.dilate(
            descriptor_pixels.astype(np.uint8),
            np.ones((3, 3), dtype=np.uint8),
            iterations=1,
        ).astype(bool)
        informative = ((saturation >= 35.0) | (edge >= 0.18)) & (value >= 35.0) & object_zone
        if not np.any(informative):
            return 0.0, 0.0, descriptor_fraction

        matches = sprite_palette_heat[informative] >= self.min_sprite_palette_match
        purity = float(matches.mean())
        informative_fraction = float(informative.sum() / max(1, descriptor_heat.size))
        return purity, informative_fraction, descriptor_fraction

    def _sprite_palette_heatmap(self, region_bgr: np.ndarray, descriptor: SimpleMobDescriptor) -> np.ndarray:
        if not descriptor.sprite_palette_bgr:
            return np.zeros(region_bgr.shape[:2], dtype=np.float32)

        pixels = region_bgr.reshape(-1, 3).astype(np.float32)
        palette = np.asarray(descriptor.sprite_palette_bgr, dtype=np.float32)
        min_dist_sq = np.full(pixels.shape[0], np.inf, dtype=np.float32)
        for start in range(0, len(palette), 128):
            chunk = palette[start : start + 128]
            diff = pixels[:, None, :] - chunk[None, :, :]
            dist_sq = np.sum(diff * diff, axis=2)
            min_dist_sq = np.minimum(min_dist_sq, dist_sq.min(axis=1))

        max_dist = max(self.max_sprite_palette_distance, 1.0)
        heat = 1.0 - (np.sqrt(min_dist_sq) / max_dist)
        return np.clip(heat, 0.0, 1.0).reshape(region_bgr.shape[:2]).astype(np.float32)

    def _object_size_score(
        self,
        descriptor_heat: np.ndarray,
        descriptor: SimpleMobDescriptor,
        expected_scale: float,
    ) -> float:
        descriptor_pixels = descriptor_heat >= self.min_descriptor_color_match
        if not np.any(descriptor_pixels):
            return 0.0

        ys, xs = np.where(descriptor_pixels)
        object_width = int(xs.max() - xs.min() + 1)
        object_height = int(ys.max() - ys.min() + 1)
        expected_width = max(1, descriptor.avg_width * expected_scale)
        expected_height = max(1, descriptor.avg_height * expected_scale)
        width_ratio = min(object_width, expected_width) / max(object_width, expected_width)
        height_ratio = min(object_height, expected_height) / max(object_height, expected_height)
        return float(np.sqrt(width_ratio * height_ratio))
