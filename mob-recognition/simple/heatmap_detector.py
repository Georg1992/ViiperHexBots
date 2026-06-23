"""Vectorized descriptor heatmaps and center peak selection."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from descriptor import ColorCluster, SimpleMobDescriptor


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
        self.max_centers = int(config.get("topCandidateCenters", 50))
        self.min_center_distance = int(config.get("minCenterDistancePx", 35))
        self.ui_top_ratio = float(config.get("playfieldTopRatio", 0.08))
        self.ui_bottom_ratio = float(config.get("playfieldBottomRatio", 0.92))
        self.ui_left_ratio = float(config.get("playfieldLeftRatio", 0.03))
        self.ui_right_ratio = float(config.get("playfieldRightRatio", 0.97))
        self.min_center_heat = float(config.get("minCenterHeat", 0.02))

    def build_heatmaps(self, frame_bgr: np.ndarray, hsv: np.ndarray, descriptor: SimpleMobDescriptor) -> Heatmaps:
        body = palette_heatmap(hsv, descriptor.body_colors)
        accent = palette_heatmap(hsv, descriptor.accent_colors)
        rare = palette_heatmap(hsv, descriptor.rare_colors)
        local_pattern = self._local_pattern(frame_bgr, accent, body)
        window = (
            max(3, descriptor.avg_width | 1),
            max(3, descriptor.avg_height | 1),
        )
        center_body = cv2.blur(body, window)
        center_accent = cv2.blur(accent, window)
        center_rare = cv2.blur(rare, window)
        center_pattern = cv2.blur(local_pattern, window)
        final = (
            center_body * 0.30
            + center_accent * 0.35
            + center_rare * 0.10
            + center_pattern * 0.15
        ).astype(np.float32)
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

    def _ui_mask(self, shape: tuple[int, int]) -> np.ndarray:
        h, w = shape
        mask = np.zeros((h, w), dtype=np.uint8)
        y1 = int(h * self.ui_top_ratio)
        y2 = int(h * self.ui_bottom_ratio)
        x1 = int(w * self.ui_left_ratio)
        x2 = int(w * self.ui_right_ratio)
        mask[y1:y2, x1:x2] = 255
        return mask
