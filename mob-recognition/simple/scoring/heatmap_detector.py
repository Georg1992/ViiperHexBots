"""Vectorized descriptor heatmaps and center peak selection."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from descriptors.descriptor import ColorCluster, SimpleMobDescriptor


@dataclass
class Heatmaps:
    body_palette: np.ndarray
    accent: np.ndarray
    rare_color: np.ndarray
    local_pattern: np.ndarray
    final_center: np.ndarray
    ui_mask: np.ndarray


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


class HeatmapDetector:
    def __init__(self, config: dict):
        self.max_centers = int(config["topCandidateCenters"])
        self.min_center_distance = int(config["minCenterDistancePx"])
        self.ui_top_ratio = float(config["playfieldTopRatio"])
        self.ui_bottom_ratio = float(config["playfieldBottomRatio"])
        self.ui_left_ratio = float(config["playfieldLeftRatio"])
        self.ui_right_ratio = float(config["playfieldRightRatio"])
        self.min_center_heat = float(config["minCenterHeat"])
        self.center_scales = [float(scale) for scale in config["centerScales"]]
        self.small_scale_min_frame_width = int(config["smallScaleMinFrameWidth"])
        weights = config["centerWeights"]
        self.center_weights = {
            "body": float(weights["bodyPalette"]),
            "accent": float(weights["accent"]),
            "rare": float(weights["rareColor"]),
            "pattern": float(weights["localPattern"]),
        }

    def build_heatmaps(
        self,
        frame_bgr: np.ndarray,
        hsv: np.ndarray,
        descriptor: SimpleMobDescriptor,
        downscale: int = 1,
    ) -> Heatmaps:
        body = palette_heatmap(hsv, descriptor.body_palette)
        accent = palette_heatmap(hsv, descriptor.accent_colors)
        rare = palette_heatmap(hsv, descriptor.rare_colors)
        local_pattern = self._local_pattern(frame_bgr, accent, body)
        final = np.zeros(body.shape, dtype=np.float32)
        blur_body = body
        blur_accent = accent
        blur_rare = rare
        blur_pattern = local_pattern
        if downscale > 1:
            blur_body = cv2.resize(body, None, fx=1.0 / downscale, fy=1.0 / downscale, interpolation=cv2.INTER_AREA)
            blur_accent = cv2.resize(accent, None, fx=1.0 / downscale, fy=1.0 / downscale, interpolation=cv2.INTER_AREA)
            blur_rare = cv2.resize(rare, None, fx=1.0 / downscale, fy=1.0 / downscale, interpolation=cv2.INTER_AREA)
            blur_pattern = cv2.resize(
                local_pattern,
                None,
                fx=1.0 / downscale,
                fy=1.0 / downscale,
                interpolation=cv2.INTER_AREA,
            )
        for scale in self._center_scales(blur_body.shape[1]):
            window = (
                max(3, int(round(descriptor.avg_width * scale)) | 1),
                max(3, int(round(descriptor.avg_height * scale)) | 1),
            )
            scale_window = (
                max(3, int(round(window[0] / downscale)) | 1),
                max(3, int(round(window[1] / downscale)) | 1),
            )
            center_body = cv2.blur(blur_body, scale_window)
            center_accent = cv2.blur(blur_accent, scale_window)
            center_rare = cv2.blur(blur_rare, scale_window)
            center_pattern = cv2.blur(blur_pattern, scale_window)
            scale_center = (
                center_body * self.center_weights["body"]
                + center_accent * self.center_weights["accent"]
                + center_rare * self.center_weights["rare"]
                + center_pattern * self.center_weights["pattern"]
            ).astype(np.float32)
            if downscale > 1:
                scale_center = cv2.resize(
                    scale_center,
                    (body.shape[1], body.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
            final = np.maximum(final, scale_center)
        ui_mask = self._ui_mask(frame_bgr.shape[:2])
        final[ui_mask == 0] = 0.0
        return Heatmaps(
            body_palette=body,
            accent=accent,
            rare_color=rare,
            local_pattern=local_pattern,
            final_center=final,
            ui_mask=ui_mask,
        )

    def top_centers(self, heatmap: np.ndarray) -> list[tuple[int, int, float]]:
        if heatmap.size == 0:
            return []
        radius = max(3, self.min_center_distance // 2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (radius * 2 + 1, radius * 2 + 1))
        local_max = heatmap == cv2.dilate(heatmap, kernel)
        threshold = max(float(heatmap.max()) * 0.45, self.min_center_heat)
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
            if scale >= 0.75 or frame_width >= self.small_scale_min_frame_width
        ]

    def _ui_mask(self, shape: tuple[int, int]) -> np.ndarray:
        h, w = shape
        mask = np.zeros((h, w), dtype=np.uint8)
        y1 = int(h * self.ui_top_ratio)
        y2 = int(h * self.ui_bottom_ratio)
        x1 = int(w * self.ui_left_ratio)
        x2 = int(w * self.ui_right_ratio)
        mask[y1:y2, x1:x2] = 255
        return mask
