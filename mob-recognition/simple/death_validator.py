"""Unified death validation using pose, size, histogram, and opacity fade signals."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from descriptor import SimpleMobDescriptor
from heatmap_detector import palette_heatmap
from region_scorer import RegionScore, SimpleRegionScorer


@dataclass(frozen=True)
class DeathValidation:
    is_dead: bool
    confidence: float
    mob_presence: float
    pose_score: float
    size_gap_score: float
    histogram_score: float
    opacity_fade_score: float
    mean_opacity: float


class DeathValidator:
    def __init__(self, config: dict, region_scorer: SimpleRegionScorer):
        self.region_scorer = region_scorer
        self.threshold = float(config["deadValidationThreshold"])
        self.watch_point_threshold = float(config["deadWatchPointThreshold"])
        self.min_mob_presence = float(config["minDeadMobPresence"])
        self.min_descriptor_color_match = float(config["minDescriptorColorMatch"])
        self.max_full_opacity = float(config["maxFullOpacity"])
        self.min_opacity_sprite_fraction = float(config["minOpacitySpriteFraction"])
        self.min_opacity_sample_pixels = int(config["minOpacitySamplePixels"])
        weights = config["deadValidationWeights"]
        self.weights = {
            "pose": float(weights["pose"]),
            "sizeGap": float(weights["sizeGap"]),
            "histogram": float(weights["histogram"]),
            "opacity": float(weights["opacity"]),
        }
        weight_total = sum(self.weights.values())
        if weight_total <= 0:
            raise ValueError("deadValidationWeights must sum to a positive value")
        self.weights = {key: value / weight_total for key, value in self.weights.items()}

    def score_dead_region(
        self,
        frame_bgr: np.ndarray,
        hsv: np.ndarray,
        descriptor: SimpleMobDescriptor,
        bbox: tuple[int, int, int, int],
        expected_scale: float,
    ) -> RegionScore | None:
        if descriptor.dead is None:
            return None
        return self.region_scorer.score(
            frame_bgr,
            hsv,
            descriptor.dead_scoring_view(),
            bbox,
            expected_scale=expected_scale,
        )

    def validate(
        self,
        frame_bgr: np.ndarray,
        hsv: np.ndarray,
        descriptor: SimpleMobDescriptor,
        living_bbox: tuple[int, int, int, int],
        dead_bbox: tuple[int, int, int, int],
        living_score: RegionScore,
        dead_score: RegionScore | None,
        living_at_dead_score: RegionScore,
        *,
        watch_point: bool = False,
    ) -> DeathValidation:
        signals = self._collect_signals(
            frame_bgr,
            hsv,
            descriptor,
            dead_bbox,
            living_score,
            dead_score,
            living_at_dead_score,
        )
        threshold = self.watch_point_threshold if watch_point else self.threshold
        is_dead = self._is_dead(signals, threshold, watch_point=watch_point)
        return DeathValidation(
            is_dead=is_dead,
            confidence=signals["confidence"],
            mob_presence=signals["mob_presence"],
            pose_score=signals["pose_score"],
            size_gap_score=signals["size_gap_score"],
            histogram_score=signals["histogram_score"],
            opacity_fade_score=signals["opacity_fade_score"],
            mean_opacity=signals["mean_opacity"],
        )

    def _collect_signals(
        self,
        frame_bgr: np.ndarray,
        hsv: np.ndarray,
        descriptor: SimpleMobDescriptor,
        dead_bbox: tuple[int, int, int, int],
        living_score: RegionScore,
        dead_score: RegionScore | None,
        living_at_dead_score: RegionScore,
    ) -> dict[str, float]:
        pose_score = dead_score.final_score if dead_score is not None else 0.0
        size_gap_score = self._size_gap_score(dead_score, living_at_dead_score)
        histogram_score = self._histogram_score(hsv, descriptor, dead_bbox)
        mean_opacity, opacity_fade_score = self._opacity_fade_score(frame_bgr, hsv, descriptor, dead_bbox)
        mob_presence = max(living_score.final_score, pose_score, living_at_dead_score.final_score)

        confidence = (
            self.weights["pose"] * pose_score
            + self.weights["sizeGap"] * size_gap_score
            + self.weights["histogram"] * histogram_score
            + self.weights["opacity"] * opacity_fade_score
        )

        return {
            "confidence": float(np.clip(confidence, 0.0, 1.0)),
            "mob_presence": float(np.clip(mob_presence, 0.0, 1.0)),
            "living_score": living_score.final_score,
            "pose_score": pose_score,
            "size_gap_score": size_gap_score,
            "histogram_score": histogram_score,
            "opacity_fade_score": opacity_fade_score,
            "mean_opacity": mean_opacity,
        }

    def _is_dead(self, signals: dict[str, float], threshold: float, *, watch_point: bool = False) -> bool:
        min_presence = self.min_mob_presence * (0.70 if watch_point else 1.0)
        if signals["mob_presence"] < min_presence:
            return False

        pose = signals["pose_score"]
        living = signals["living_score"]
        gap = signals["size_gap_score"]
        confidence = signals["confidence"]
        fade = signals["opacity_fade_score"]

        if fade > 0.0 and confidence >= self.watch_point_threshold:
            return True

        if watch_point:
            if gap >= 0.36 and pose >= 0.26:
                return True
            pose_living_ratio = pose / max(living, 1e-6)
            if pose >= 0.24 and pose_living_ratio <= 0.84:
                return True
            if confidence >= threshold and pose >= living and pose >= 0.22:
                return True
            return False

        if pose >= living and confidence >= threshold:
            return True

        if living >= 0.62 or pose < 0.30 or gap < 0.39:
            return False

        pose_living_ratio = pose / max(living, 1e-6)
        mob_presence = signals["mob_presence"]
        if mob_presence >= 0.535:
            return pose_living_ratio <= 0.82
        return pose_living_ratio <= 0.65

    @staticmethod
    def _size_gap_score(dead_score: RegionScore | None, living_at_dead_score: RegionScore) -> float:
        if dead_score is None:
            return 0.0
        gap = dead_score.size_score - living_at_dead_score.size_score
        return float(np.clip((gap + 1.0) / 2.0, 0.0, 1.0))

    @staticmethod
    def _histogram_similarity(region_hsv: np.ndarray, reference_hist: list[float]) -> float:
        if region_hsv.size == 0 or not reference_hist:
            return 0.0
        strip = region_hsv.astype(np.uint8).reshape(1, -1, 3)
        hist = cv2.calcHist([strip], [0, 1], None, [24, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        reference = np.asarray(reference_hist, dtype=np.float32).reshape(24, 16)
        correlation = cv2.compareHist(hist, reference, cv2.HISTCMP_CORREL)
        return float(np.clip((correlation + 1.0) / 2.0, 0.0, 1.0))

    def _histogram_score(
        self,
        hsv: np.ndarray,
        descriptor: SimpleMobDescriptor,
        bbox: tuple[int, int, int, int],
    ) -> float:
        if descriptor.dead is None:
            return 0.0
        x, y, w, h = bbox
        region_hsv = hsv[y : y + h, x : x + w]
        dead_hist = self._histogram_similarity(region_hsv, descriptor.dead.hsv_histogram)
        living_hist = self._histogram_similarity(region_hsv, descriptor.hsv_histogram)
        return float(np.clip((dead_hist - living_hist + 1.0) / 2.0, 0.0, 1.0))

    def _opacity_fade_score(
        self,
        frame_bgr: np.ndarray,
        hsv: np.ndarray,
        descriptor: SimpleMobDescriptor,
        bbox: tuple[int, int, int, int],
    ) -> tuple[float, float]:
        x, y, w, h = bbox
        region_bgr = frame_bgr[y : y + h, x : x + w]
        region_hsv = hsv[y : y + h, x : x + w]
        if region_bgr.size == 0 or not descriptor.sprite_palette_bgr:
            return 1.0, 0.0

        body_heat = palette_heatmap(region_hsv, descriptor.body_colors)
        accent_heat = palette_heatmap(region_hsv, descriptor.accent_colors)
        object_mask = np.maximum(body_heat, accent_heat) >= self.min_descriptor_color_match
        sample_count = int(object_mask.sum())
        if sample_count < self.min_opacity_sample_pixels:
            return 1.0, 0.0
        if float(object_mask.mean()) < self.min_opacity_sprite_fraction:
            return 1.0, 0.0

        background_bgr = self._estimate_background_bgr(region_bgr, object_mask)
        if background_bgr is None:
            return 1.0, 0.0

        mean_opacity = self._estimate_mean_opacity(
            region_bgr[object_mask].astype(np.float32),
            background_bgr,
            descriptor.sprite_palette_bgr,
        )
        if mean_opacity >= self.max_full_opacity:
            return mean_opacity, 0.0
        fade_score = float(np.clip((self.max_full_opacity - mean_opacity) / self.max_full_opacity, 0.0, 1.0))
        return mean_opacity, fade_score

    @staticmethod
    def _estimate_background_bgr(region_bgr: np.ndarray, object_mask: np.ndarray) -> np.ndarray | None:
        ring = cv2.dilate(object_mask.astype(np.uint8), np.ones((5, 5), dtype=np.uint8), iterations=1)
        ring = (ring > 0) & (~object_mask.astype(bool))
        if not np.any(ring):
            return None
        return np.median(region_bgr[ring].reshape(-1, 3), axis=0).astype(np.float32)

    @staticmethod
    def _estimate_mean_opacity(
        pixels: np.ndarray,
        background_bgr: np.ndarray,
        sprite_palette_bgr: list[tuple[int, int, int]],
    ) -> float:
        palette = np.asarray(sprite_palette_bgr, dtype=np.float32)
        diff = pixels[:, None, :] - palette[None, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        foreground = palette[dist_sq.argmin(axis=1)]

        background = np.broadcast_to(background_bgr, pixels.shape).astype(np.float32)
        denom = foreground - background
        numer = pixels - background

        channel_alphas: list[np.ndarray] = []
        for channel in range(3):
            channel_denom = denom[:, channel]
            valid = np.abs(channel_denom) > 8.0
            if not np.any(valid):
                continue
            alpha = numer[valid, channel] / channel_denom[valid]
            channel_alphas.append(np.clip(alpha, 0.0, 1.0))

        if not channel_alphas:
            return 1.0
        return float(np.median(np.concatenate(channel_alphas)))
