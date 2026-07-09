"""Vectorized descriptor heatmaps and center peak selection."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from pybot.recognition.detector.descriptors.descriptor import ColorCluster, MobDescriptor


@dataclass
class Heatmaps:
    body_palette: np.ndarray
    accent: np.ndarray
    rare_color: np.ndarray
    local_pattern: np.ndarray
    final_center: np.ndarray
    structural_center: np.ndarray
    ui_mask: np.ndarray
    playfield_offset: tuple[int, int] = (0, 0)


def _cluster_match(hsv: np.ndarray, cluster: ColorCluster) -> np.ndarray:
    center = np.array(cluster.hsv, dtype=np.float32)
    tol = np.array(cluster.tolerance, dtype=np.float32)
    hsv_f = hsv.astype(np.float32)
    hue_diff = np.abs(hsv_f[:, :, 0] - center[0])
    hue_diff = np.minimum(hue_diff, 180.0 - hue_diff)
    sat_diff = np.abs(hsv_f[:, :, 1] - center[1])
    val_diff = np.abs(hsv_f[:, :, 2] - center[2])
    hue_score = np.clip(1.0 - hue_diff / max(tol[0], 1.0), 0.0, 1.0)
    sat_score = np.clip(1.0 - sat_diff / max(tol[1], 1.0), 0.0, 1.0)
    val_score = np.clip(1.0 - val_diff / max(tol[2], 1.0), 0.0, 1.0)
    return (hue_score * sat_score * val_score).astype(np.float32)


def palette_heatmap(hsv: np.ndarray, clusters: list[ColorCluster]) -> np.ndarray:
    if not clusters:
        return np.zeros(hsv.shape[:2], dtype=np.float32)
    heat = np.zeros(hsv.shape[:2], dtype=np.float32)
    for cluster in clusters:
        heat = np.maximum(heat, _cluster_match(hsv, cluster))
    return heat


def sprite_palette_heatmap(
    frame_bgr: np.ndarray,
    palette_bgr: list[tuple[int, int, int]],
    max_distance: float,
) -> np.ndarray:
    if not palette_bgr:
        return np.zeros(frame_bgr.shape[:2], dtype=np.float32)

    pixels = frame_bgr.reshape(-1, 3).astype(np.float32)
    palette = np.asarray(palette_bgr, dtype=np.float32)
    min_dist_sq = np.full(pixels.shape[0], np.inf, dtype=np.float32)
    for start in range(0, len(palette), 128):
        chunk = palette[start : start + 128]
        diff = pixels[:, None, :] - chunk[None, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        min_dist_sq = np.minimum(min_dist_sq, dist_sq.min(axis=1))

    max_dist = max(max_distance, 1.0)
    heat = 1.0 - (np.sqrt(min_dist_sq) / max_dist)
    return np.clip(heat, 0.0, 1.0).reshape(frame_bgr.shape[:2]).astype(np.float32)


class HeatmapDetector:
    def __init__(self, config: dict):
        self.max_centers = int(config["topCandidateCenters"])
        self.min_center_distance = int(config["minCenterDistancePx"])
        self.ui_top_ratio = float(config["playfieldTopRatio"])
        self.ui_bottom_ratio = float(config["playfieldBottomRatio"])
        self.ui_left_ratio = float(config["playfieldLeftRatio"])
        self.ui_right_ratio = float(config["playfieldRightRatio"])
        self.min_center_heat = float(config["minCenterHeat"])
        self.peak_relative_threshold = float(config["peakRelativeThreshold"])
        self.center_scales = [float(scale) for scale in config["centerScales"]]
        self.small_scale_min_frame_width = int(config["smallScaleMinFrameWidth"])
        self.small_scale_cutoff = float(config["smallScaleCutoff"])
        self.max_sprite_palette_distance = float(config["maxSpritePaletteDistance"])
        self.accent_structural_distance = float(config["accentStructuralPixelDistance"])
        self.structural_discovery_min_score = float(config["structuralDiscoveryMinScore"])

    def playfield_bounds(self, shape: tuple[int, int]) -> tuple[int, int, int, int]:
        h, w = shape
        return (
            int(h * self.ui_top_ratio),
            int(h * self.ui_bottom_ratio),
            int(w * self.ui_left_ratio),
            int(w * self.ui_right_ratio),
        )

    def build_heatmaps(
        self,
        frame_bgr: np.ndarray,
        hsv: np.ndarray | None,
        descriptor: MobDescriptor,
        downscale: int = 1,
        *,
        discovery_only: bool = True,
    ) -> Heatmaps:
        y1, y2, x1, x2 = self.playfield_bounds(frame_bgr.shape[:2])
        crop_bgr = frame_bgr[y1:y2, x1:x2]
        crop_shape = crop_bgr.shape[:2]

        if discovery_only:
            body = accent = rare = local_pattern = np.zeros(crop_shape, dtype=np.float32)
        else:
            if hsv is None:
                raise ValueError("hsv is required when discovery_only is False")
            crop_hsv = hsv[y1:y2, x1:x2]
            body = palette_heatmap(crop_hsv, descriptor.body_palette)
            accent = palette_heatmap(crop_hsv, descriptor.accent_colors)
            rare = palette_heatmap(crop_hsv, descriptor.rare_colors)
            local_pattern = self._local_pattern(crop_bgr, accent, body)

        work_bgr = crop_bgr
        if downscale > 1:
            work_bgr = cv2.resize(
                crop_bgr,
                None,
                fx=1.0 / downscale,
                fy=1.0 / downscale,
                interpolation=cv2.INTER_AREA,
            )

        sprite = sprite_palette_heatmap(
            work_bgr,
            descriptor.match_palette_bgr,
            self.max_sprite_palette_distance,
        )
        body_sprite: np.ndarray | None = None
        accent_sprite: np.ndarray | None = None
        has_structural = (
            descriptor.dominant_pixel_bgr is not None and descriptor.accent_pixel_bgr is not None
        )
        if has_structural:
            body_sprite = sprite_palette_heatmap(
                work_bgr,
                [tuple(descriptor.dominant_pixel_bgr)],
                self.max_sprite_palette_distance,
            )
            accent_sprite = sprite_palette_heatmap(
                work_bgr,
                [tuple(descriptor.accent_pixel_bgr)],
                self.accent_structural_distance,
            )

        final = np.zeros(crop_shape, dtype=np.float32)
        structural_final = np.zeros(crop_shape, dtype=np.float32)
        for scale in self._center_scales(work_bgr.shape[1]):
            window = (
                max(3, int(round(descriptor.avg_width * scale / downscale)) | 1),
                max(3, int(round(descriptor.avg_height * scale / downscale)) | 1),
            )
            sprite_heat = cv2.blur(sprite, window)
            if downscale > 1:
                sprite_heat = cv2.resize(
                    sprite_heat,
                    (crop_shape[1], crop_shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
            final = np.maximum(final, sprite_heat.astype(np.float32))
            if body_sprite is not None and accent_sprite is not None:
                body_heat = cv2.blur(body_sprite, window)
                accent_heat = cv2.blur(accent_sprite, window)
                structural_heat = np.sqrt(body_heat * accent_heat).astype(np.float32)
                if downscale > 1:
                    structural_heat = cv2.resize(
                        structural_heat,
                        (crop_shape[1], crop_shape[0]),
                        interpolation=cv2.INTER_LINEAR,
                    )
                structural_final = np.maximum(structural_final, structural_heat.astype(np.float32))

        ui_mask = self._ui_mask(crop_shape)
        final[ui_mask == 0] = 0.0
        structural_final[ui_mask == 0] = 0.0
        return Heatmaps(
            body_palette=body,
            accent=accent,
            rare_color=rare,
            local_pattern=local_pattern,
            final_center=final,
            structural_center=structural_final,
            ui_mask=ui_mask,
            playfield_offset=(x1, y1),
        )

    def top_centers(self, heatmap: np.ndarray) -> list[tuple[int, int, float]]:
        if heatmap.size == 0:
            return []
        radius = max(3, self.min_center_distance // 2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (radius * 2 + 1, radius * 2 + 1))
        local_max = heatmap == cv2.dilate(heatmap, kernel)
        threshold = max(
            float(heatmap.max()) * self.peak_relative_threshold,
            self.min_center_heat,
        )
        ys, xs = np.where(local_max & (heatmap >= threshold))
        if len(xs) == 0:
            return []
        scores = heatmap[ys, xs]
        order = np.argsort(scores)[::-1]
        centers: list[tuple[int, int, float]] = []
        min_dist_sq = self.min_center_distance * self.min_center_distance
        for idx in order:
            x, y, score = int(xs[idx]), int(ys[idx]), float(scores[idx])
            if all((x - px) ** 2 + (y - py) ** 2 >= min_dist_sq for px, py, _ in centers):
                centers.append((x, y, score))
                if len(centers) >= self.max_centers:
                    break
        return centers

    @staticmethod
    def _local_pattern(frame_bgr: np.ndarray, accent: np.ndarray, body: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        edge = cv2.magnitude(grad_x, grad_y)
        edge = cv2.normalize(edge, None, 0.0, 1.0, cv2.NORM_MINMAX)
        pattern = np.maximum(accent * 0.8, body * edge)
        return np.clip(pattern, 0.0, 1.0).astype(np.float32)

    def _center_scales(self, frame_width: int) -> list[float]:
        return [
            scale
            for scale in self.center_scales
            if scale >= self.small_scale_cutoff or frame_width >= self.small_scale_min_frame_width
        ]

    def _ui_mask(self, shape: tuple[int, int]) -> np.ndarray:
        h, w = shape
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[:, :] = 255
        return mask
