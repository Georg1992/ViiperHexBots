"""Vectorized descriptor sprite heatmap and blob finding.

Pipeline: sprite palette heatmap → selectivity boost → edge-density boost
          → multi-scale blur → connected components → blob centers.
"""

from __future__ import annotations

import cv2
import numpy as np

from pybot.recognition.detector.descriptors.descriptor import ColorCluster, MobDescriptor


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
        *,
        exclude_palette_bgr: list[tuple[int, int, int]] | None = None,
        texture_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Build sprite palette heatmap with selectivity and edge-density boosts.

        Returns a float32 heatmap at full frame resolution. Each pixel ≈ how likely
        it is part of the mob based on palette match, colour selectivity, and edges.
        """
        frame_shape = frame_bgr.shape[:2]

        # --- downscale work images if needed ---
        work_bgr = frame_bgr
        work_hsv = hsv
        if downscale > 1:
            work_bgr = cv2.resize(
                frame_bgr, None,
                fx=1.0 / downscale, fy=1.0 / downscale,
                interpolation=cv2.INTER_AREA,
            )
            work_hsv = cv2.resize(
                hsv, None,
                fx=1.0 / downscale, fy=1.0 / downscale,
                interpolation=cv2.INTER_AREA,
            )

        # --- 1. Pure sprite-palette-distance heatmap ---
        sprite = sprite_palette_heatmap(
            work_bgr,
            descriptor.match_palette_bgr,
            self.max_sprite_palette_distance,
        )

        if exclude_palette_bgr:
            background_heat = sprite_palette_heatmap(
                work_bgr,
                exclude_palette_bgr,
                self.max_sprite_palette_distance,
            )
            sprite = np.where(background_heat >= np.float32(0.55), np.float32(0.0), sprite)

        if texture_mask is not None:
            work_mask = texture_mask
            if downscale > 1:
                work_mask = cv2.resize(
                    texture_mask,
                    (work_bgr.shape[1], work_bgr.shape[0]),
                    interpolation=cv2.INTER_AREA,
                )
            sprite *= work_mask.astype(np.float32)

        # --- 2. Selectivity: body × accent → suppress non-mob colour combos ---
        accent = palette_heatmap(work_hsv, descriptor.accent_colors)
        body = palette_heatmap(work_hsv, descriptor.body_palette)
        selectivity = np.sqrt(body * accent).astype(np.float32)
        sprite *= np.float32(0.5) + np.float32(0.5) * selectivity

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

        # --- 4. Multi-scale blur → aggregate hot-spots at mob-sized scales ---
        final = np.zeros(frame_shape, dtype=np.float32)
        for scale in self._center_scales(work_bgr.shape[1]):
            window = (
                max(3, int(round(descriptor.avg_width * scale / downscale)) | 1),
                max(3, int(round(descriptor.avg_height * scale / downscale)) | 1),
            )
            blurred = cv2.blur(sprite, window)
            if downscale > 1:
                blurred = cv2.resize(
                    blurred, (frame_shape[1], frame_shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
            final = np.maximum(final, blurred.astype(np.float32))

        return final

    def top_centers(
        self, heatmap: np.ndarray,
    ) -> list[tuple[int, int, float, tuple[int, int, int, int]]]:
        """Find distinct hot regions via connected components.

        Thresholds the heatmap, finds contiguous blobs above threshold,
        merges nearby fragments, and returns heat-weighted centroids with
        bounding boxes.  Yields exactly 1 center per visually distinct
        hot region, typically 3–5 per frame at most.
        """
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

        # Merge fragments within MERGE_DIST that belong to the same mob.
        MERGE_DIST = 40  # pixels in heatmap space
        MERGE_DIST_SQ = MERGE_DIST * MERGE_DIST
        merged: list[tuple[int, int, float, tuple[int, int, int, int]]] = []
        for cx, cy, heat, (left, top, w, h) in raw:
            merged_flag = False
            for mi, (mx, my, mheat, mbbox) in enumerate(merged):
                if (cx - mx) ** 2 + (cy - my) ** 2 < MERGE_DIST_SQ:
                    ml, mt, mw, mh = mbbox
                    nleft = min(ml, left)
                    ntop = min(mt, top)
                    nright = max(ml + mw, left + w)
                    nbottom = max(mt + mh, top + h)
                    merged[mi] = (
                        (nleft + nright) // 2,
                        (ntop + nbottom) // 2,
                        max(mheat, heat),
                        (nleft, ntop, nright - nleft, nbottom - ntop),
                    )
                    merged_flag = True
                    break
            if not merged_flag:
                merged.append((cx, cy, heat, (left, top, w, h)))

        return merged[: self.max_centers]
