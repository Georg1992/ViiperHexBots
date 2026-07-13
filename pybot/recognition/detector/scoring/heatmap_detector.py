"""Vectorized descriptor sprite heatmap and blob finding.

Pipeline: weighted sprite palette heatmap → selectivity boost → edge-density boost
          → GaussianBlur → normalize → connected components → blob centers.
"""

from __future__ import annotations

import cv2
import numpy as np

from pybot.recognition.detector.descriptors.descriptor import ColorCluster, MobDescriptor


def _cluster_match(hsv_f: np.ndarray, cluster: ColorCluster) -> np.ndarray:
    center = np.array(cluster.hsv, dtype=np.float32)
    tol = np.array(cluster.tolerance, dtype=np.float32)
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
    hsv_f = hsv.astype(np.float32)
    heat = np.zeros(hsv.shape[:2], dtype=np.float32)
    for cluster in clusters:
        heat = np.maximum(heat, _cluster_match(hsv_f, cluster))
    return heat


def sprite_palette_heatmap(
    frame_bgr: np.ndarray,
    palette_bgr: list[tuple[int, int, int]],
    max_distance: float,
) -> np.ndarray:
    """Euclidean-distance heatmap: how close each pixel is to any palette color."""
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


def _palette_role_weights(descriptor: MobDescriptor) -> np.ndarray:
    peak_weight = max(descriptor.match_palette_weights) if descriptor.match_palette_weights else 1.0
    shade_cutoff = np.float32(0.20) * np.float32(peak_weight)

    accent_bgrs: set[tuple[int, int, int]] = set()
    for pixel in descriptor.accent_pixels_bgr or []:
        accent_bgrs.add(tuple(int(v) for v in pixel))
    for cluster in descriptor.accent_colors:
        accent_bgrs.add(tuple(int(v) for v in cluster.bgr))

    structural_bgrs: set[tuple[int, int, int]] = set()
    for pixel in descriptor.dominant_pixels_bgr or []:
        structural_bgrs.add(tuple(int(v) for v in pixel))
    if descriptor.dominant_pixel_bgr is not None:
        structural_bgrs.add(tuple(int(v) for v in descriptor.dominant_pixel_bgr))

    roles = np.empty(len(descriptor.match_palette_bgr), dtype=np.float32)
    for index, bgr in enumerate(descriptor.match_palette_bgr):
        key = tuple(int(v) for v in bgr)
        weight = (
            descriptor.match_palette_weights[index]
            if index < len(descriptor.match_palette_weights)
            else 1.0
        )
        if key in accent_bgrs:
            roles[index] = np.float32(1.10)
        elif key in structural_bgrs or np.float32(weight) >= shade_cutoff:
            roles[index] = np.float32(1.00)
        else:
            roles[index] = np.float32(0.45)
    return roles


def _palette_descriptor_weights(descriptor: MobDescriptor) -> np.ndarray:
    raw = descriptor.match_palette_weights
    if len(raw) != len(descriptor.match_palette_bgr):
        raw = [1.0] * len(descriptor.match_palette_bgr)
    return (
        np.float32(0.6) + np.float32(0.4) * np.sqrt(np.asarray(raw, dtype=np.float32))
    ).astype(np.float32)


def weighted_sprite_palette_heatmap(
    frame_bgr: np.ndarray,
    descriptor: MobDescriptor,
    max_distance: float,
) -> np.ndarray:
    """Palette heatmap with runtime rarity, descriptor, and role weighting.

    Distance computed via |p-c|² = |p|² + |c|² - 2p·c to avoid the (N,C,3)
    intermediate and to route through BLAS (numpy dot).  The single (N,C)
    buffer is reused in-place for distance, similarity, and weighting.
    """
    palette_bgr = descriptor.match_palette_bgr
    if not palette_bgr:
        return np.zeros(frame_bgr.shape[:2], dtype=np.float32)

    pixels = frame_bgr.reshape(-1, 3).astype(np.float32)
    palette = np.asarray(palette_bgr, dtype=np.float32)
    n_pixels = pixels.shape[0]
    n_colors = len(palette)
    max_dist = np.float32(max(max_distance, 1.0))

    # --- distance via expansion (avoids 3×-larger diff intermediate) ---
    p_norm = np.sum(pixels * pixels, axis=1, keepdims=True)            # (N, 1)
    c_norm = np.sum(palette * palette, axis=1, keepdims=True)          # (C, 1)
    dist_sq = np.dot(pixels, palette.T)                                # (N, C)
    dist_sq *= np.float32(-2.0)
    dist_sq += p_norm                                                  # broadcast (N, 1)
    dist_sq += c_norm.T                                                # broadcast (1, C)
    np.maximum(dist_sq, np.float32(0.0), out=dist_sq)  # clamp fp noise

    # --- nearest-color index → per-color rarity weights ---
    nearest_idx = dist_sq.argmin(axis=1)
    palette_match_count = np.bincount(nearest_idx, minlength=n_colors).astype(np.float32)
    scene_fraction = palette_match_count / np.float32(max(n_pixels, 1))
    rarity = np.float32(1.0) / np.sqrt(scene_fraction + np.float32(1e-6))
    median_rarity = float(np.median(rarity))
    if median_rarity > 0.0:
        rarity = (rarity / np.float32(median_rarity)).astype(np.float32)
    rarity = np.clip(rarity, np.float32(0.25), np.float32(2.0))

    combined_w = (
        rarity
        * _palette_descriptor_weights(descriptor)
        * _palette_role_weights(descriptor)
    ).astype(np.float32)

    # --- in-place: dist_sq → distance → similarity → weighted → max ---
    np.sqrt(dist_sq, out=dist_sq)                                       # now distances
    dist_sq /= max_dist
    np.subtract(np.float32(1.0), dist_sq, out=dist_sq)                  # similarity
    np.clip(dist_sq, np.float32(0.0), np.float32(1.0), out=dist_sq)
    np.multiply(dist_sq, combined_w, out=dist_sq)                      # weighted (guaranteed in-place)
    best_weighted = dist_sq.max(axis=1)

    return best_weighted.reshape(frame_bgr.shape[:2])


def _nearest_upscale(heatmap: np.ndarray, scale: int, out_h: int, out_w: int) -> np.ndarray:
    """Repeat each pooled cell to recover full-frame heatmap coordinates."""
    if scale <= 1:
        return heatmap.astype(np.float32)
    upscaled = np.repeat(np.repeat(heatmap, scale, axis=0), scale, axis=1)
    return upscaled[:out_h, :out_w].astype(np.float32)


def _local_peak_boost(heatmap: np.ndarray, factor: float = 1.08) -> np.ndarray:
    """Boost local maxima only — leaves broad background plateaus unchanged."""
    kernel = np.ones((3, 3), np.uint8)
    local_max = cv2.dilate(heatmap, kernel)
    peak_mask = heatmap >= local_max - np.float32(1e-6)
    boosted = heatmap.copy()
    boosted[peak_mask] *= np.float32(factor)
    return np.clip(boosted, np.float32(0.0), np.float32(1.0)).astype(np.float32)


class HeatmapDetector:
    """Builds a single sprite-matching heatmap and finds blob centers."""

    def __init__(self, config: dict):
        self.max_centers = int(config["topCandidateCenters"])
        self.min_center_heat = float(config["minCenterHeat"])
        self.peak_relative_threshold = float(config["peakRelativeThreshold"])
        self.center_scales = [float(scale) for scale in config["centerScales"]]
        self.small_scale_min_frame_width = int(config["smallScaleMinFrameWidth"])
        self.small_scale_cutoff = float(config["smallScaleCutoff"])
        self.max_sprite_palette_distance = float(config["maxSpritePaletteDistance"])

    def _center_scales(self, frame_width: int) -> list[float]:
        return [
            s for s in self.center_scales
            if s >= self.small_scale_cutoff or frame_width >= self.small_scale_min_frame_width
        ]

    def build_sprite_heatmap(
        self,
        frame_bgr: np.ndarray,
        hsv: np.ndarray,
        descriptor: MobDescriptor,
        downscale: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build sprite palette heatmap with selectivity and edge-density boosts.

        Returns (sprite_heatmap, accent_heatmap) at full frame resolution.
        accent_heatmap is the raw accent cluster heatmap reused by blob filters.
        """
        frame_shape = frame_bgr.shape[:2]

        # --- early downscale: run the expensive steps at discovery resolution ---
        if downscale > 1:
            ds_h, ds_w = frame_shape[0] // downscale, frame_shape[1] // downscale
            work_bgr = cv2.resize(frame_bgr, (ds_w, ds_h), interpolation=cv2.INTER_NEAREST)
            work_hsv = cv2.resize(hsv, (ds_w, ds_h), interpolation=cv2.INTER_NEAREST)
        else:
            work_bgr = frame_bgr
            work_hsv = hsv

        # --- 1. Weighted sprite-palette-distance heatmap ---
        sprite = weighted_sprite_palette_heatmap(
            work_bgr,
            descriptor,
            self.max_sprite_palette_distance,
        )

        # --- 2. Selectivity: body × accent → suppress non-mob colour combos ---
        accent = palette_heatmap(work_hsv, descriptor.accent_colors)
        body = palette_heatmap(work_hsv, descriptor.body_palette)
        if descriptor.accent_colors:
            joint = np.sqrt(body * accent).astype(np.float32)
            selectivity = np.float32(0.18) + np.float32(0.82) * joint
        else:
            selectivity = np.float32(0.18) + np.float32(0.82) * body.astype(np.float32)
        sprite *= selectivity

        # --- 3. Edge-density: mobs have edges, flat terrain doesn't ---
        gray = cv2.cvtColor(work_bgr, cv2.COLOR_BGR2GRAY)
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        edge_mag = cv2.magnitude(grad_x, grad_y)
        edge_density = cv2.blur(edge_mag, (7, 7))
        p95 = float(np.percentile(edge_density, 95))
        if p95 > 1e-6:
            edge_density = np.clip(edge_density / p95, 0.0, 1.0).astype(np.float32)
        else:
            edge_density = np.zeros_like(edge_density, dtype=np.float32)
        sprite *= np.float32(0.5) + np.float32(0.5) * edge_density

        # --- 4. GaussianBlur + normalize — every mob gets full contrast ---
        w = max(3, int(round(descriptor.avg_width * 0.8 / downscale)) | 1)
        h = max(3, int(round(descriptor.avg_height * 0.8 / downscale)) | 1)
        blurred = cv2.GaussianBlur(sprite, (w, h), 0)
        final = cv2.normalize(blurred, None, 0.0, 1.0, cv2.NORM_MINMAX)

        # --- 5. Upscale back to full frame with local peak recovery ---
        if downscale > 1:
            final = _local_peak_boost(final)
            final = _nearest_upscale(final, downscale, frame_shape[0], frame_shape[1])
            accent = _nearest_upscale(accent, downscale, frame_shape[0], frame_shape[1])

        return final, accent

    def top_centers(
        self, heatmap: np.ndarray,
    ) -> list[tuple[int, int, float, tuple[int, int, int, int]]]:
        """Find distinct hot regions via connected components, no merge."""
        if heatmap.size == 0:
            return []

        threshold = max(
            float(heatmap.max()) * self.peak_relative_threshold,
            self.min_center_heat,
        )
        binary = (heatmap >= threshold).astype(np.uint8)
        if not np.any(binary):
            return []

        num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8,
        )
        if num_labels <= 1:
            return []

        raw: list[tuple[int, int, float, tuple[int, int, int, int]]] = []
        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if area < 12:
                continue

            mask = labels == label
            vals = heatmap[mask]
            peak_score = float(vals.max())

            weights = vals.astype(np.float32)
            if weights.sum() > 0:
                ys, xs = np.where(mask)
                cx = int(np.average(xs, weights=weights))
                cy = int(np.average(ys, weights=weights))
            else:
                r = _centroids[label]
                cx, cy = int(round(r[0])), int(round(r[1]))

            comp_bbox = (
                stats[label, cv2.CC_STAT_LEFT],
                stats[label, cv2.CC_STAT_TOP],
                stats[label, cv2.CC_STAT_WIDTH],
                stats[label, cv2.CC_STAT_HEIGHT],
            )
            raw.append((cx, cy, peak_score, comp_bbox))

        raw.sort(key=lambda item: item[2], reverse=True)
        return raw[: self.max_centers]
