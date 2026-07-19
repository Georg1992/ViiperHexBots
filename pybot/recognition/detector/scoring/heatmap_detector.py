"""Vectorized descriptor sprite heatmap and blob finding.

Pipeline: weighted sprite palette heatmap → optional palette diversity
          → edge boost → GaussianBlur → connected components → blob centers.
"""

from __future__ import annotations

import cv2
import numpy as np

from pybot.recognition.detector.descriptors.descriptor import ColorCluster, MobDescriptor

_EDGE_BLUR_KSIZE = (7, 7)

# Local palette-diversity: mono-family → 0 heat; full heat only when
# every Lab group is present in the coverage window.
_PRESENCE_SIMILARITY_LOW = np.float32(0.35)
_PRESENCE_SIMILARITY_HIGH = np.float32(0.75)
_MIN_GROUP_AREA_FRACTION = np.float32(0.01)
_DIVERSITY_FLOOR = np.float32(0.0)
_DIVERSITY_POWER = np.float32(2.0)
_COVERAGE_SIZE_FRAC = 0.6
# Near-duplicate blob suppress radius as a fraction of min(sprite w, h).
_BLOB_DEDUP_SIZE_FRAC = 0.85


def _cluster_match(bgr_f: np.ndarray, cluster: ColorCluster) -> np.ndarray:
    center = np.array(cluster.bgr, dtype=np.float32)
    diff = bgr_f - center
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    max_d = max(float(cluster.max_distance), 1.0)
    return np.clip(1.0 - dist / max_d, 0.0, 1.0).astype(np.float32)


def palette_heatmap(frame_bgr: np.ndarray, clusters: list[ColorCluster]) -> np.ndarray:
    """BGR Euclidean heatmap against ColorCluster centers (tracking/opacity)."""
    if not clusters:
        return np.zeros(frame_bgr.shape[:2], dtype=np.float32)
    bgr_f = frame_bgr.astype(np.float32)
    heat = np.zeros(frame_bgr.shape[:2], dtype=np.float32)
    for cluster in clusters:
        heat = np.maximum(heat, _cluster_match(bgr_f, cluster))
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


def _palette_descriptor_weights(descriptor: MobDescriptor) -> np.ndarray:
    raw = np.asarray(descriptor.match_palette_weights, dtype=np.float32)
    return (np.float32(0.6) + np.float32(0.4) * np.sqrt(raw)).astype(np.float32)


def _coverage_window(avg_width: float, avg_height: float, downscale: int) -> tuple[int, int]:
    """Odd local support ≈ 0.6 × mob size at discovery resolution."""
    w = max(3, int(round(avg_width * _COVERAGE_SIZE_FRAC / max(downscale, 1))) | 1)
    h = max(3, int(round(avg_height * _COVERAGE_SIZE_FRAC / max(downscale, 1))) | 1)
    return (w, h)


def weighted_sprite_palette_heatmap(
    frame_bgr: np.ndarray,
    descriptor: MobDescriptor,
    max_distance: float,
    *,
    return_similarity: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Palette heatmap with runtime rarity and descriptor frequency weights.

    Distance computed via |p-c|² = |p|² + |c|² - 2p·c to avoid the (N,C,3)
    intermediate and to route through BLAS (numpy dot).

    When ``return_similarity`` is True, also returns the unweighted per-color
    similarity map shaped (H, W, C) for palette-group coverage.
    """
    palette_bgr = descriptor.match_palette_bgr
    shape_hw = frame_bgr.shape[:2]
    if not palette_bgr:
        empty = np.zeros(shape_hw, dtype=np.float32)
        if return_similarity:
            return empty, np.zeros((*shape_hw, 0), dtype=np.float32)
        return empty

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

    combined_w = (rarity * _palette_descriptor_weights(descriptor)).astype(np.float32)

    # --- distance → similarity (keep unweighted for group coverage) ---
    np.sqrt(dist_sq, out=dist_sq)
    dist_sq /= max_dist
    np.subtract(np.float32(1.0), dist_sq, out=dist_sq)
    np.clip(dist_sq, np.float32(0.0), np.float32(1.0), out=dist_sq)
    similarity = dist_sq  # (N, C) unweighted

    best_weighted = (similarity * combined_w).max(axis=1)
    base_sprite = best_weighted.reshape(shape_hw)

    if return_similarity:
        return base_sprite, similarity.reshape(*shape_hw, n_colors).astype(np.float32)
    return base_sprite


def apply_palette_diversity(
    base_sprite: np.ndarray,
    similarity_hwc: np.ndarray,
    palette_groups: list[list[int]],
    *,
    avg_width: float,
    avg_height: float,
    downscale: int = 1,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Soft local multi-family coverage factor (never hard-rejects).

    Returns ``(sprite_with_diversity, debug_maps)``.
    """
    h, w = base_sprite.shape[:2]
    n_groups = len(palette_groups)
    if n_groups == 0 or similarity_hwc.size == 0:
        ones = np.ones((h, w), dtype=np.float32)
        return base_sprite.copy(), {
            "group_similarity": [],
            "group_present": [],
            "effective_groups": np.zeros((h, w), dtype=np.float32),
            "diversity_factor": ones,
        }

    denom = _PRESENCE_SIMILARITY_HIGH - _PRESENCE_SIMILARITY_LOW
    ksize = _coverage_window(avg_width, avg_height, downscale)
    required_groups = float(n_groups)

    group_similarity: list[np.ndarray] = []
    group_present: list[np.ndarray] = []
    effective = np.zeros((h, w), dtype=np.float32)

    for indices in palette_groups:
        idx = np.asarray(indices, dtype=np.int32)
        g_sim = similarity_hwc[:, :, idx].max(axis=2).astype(np.float32)
        group_similarity.append(g_sim)

        matched = np.clip(
            (g_sim - _PRESENCE_SIMILARITY_LOW) / denom,
            0.0,
            1.0,
        ).astype(np.float32)
        local_presence = cv2.boxFilter(
            matched, ddepth=-1, ksize=ksize, normalize=True,
        )
        present = np.clip(
            local_presence / _MIN_GROUP_AREA_FRACTION, 0.0, 1.0,
        ).astype(np.float32)
        group_present.append(present)
        effective += present

    diversity = np.clip(
        (effective - np.float32(1.0)) / np.float32(max(required_groups - 1.0, 1.0)),
        0.0,
        1.0,
    ).astype(np.float32)
    np.power(diversity, _DIVERSITY_POWER, out=diversity)
    diversity_factor = (
        _DIVERSITY_FLOOR + (np.float32(1.0) - _DIVERSITY_FLOOR) * diversity
    ).astype(np.float32)

    sprite = (base_sprite * diversity_factor).astype(np.float32)
    return sprite, {
        "group_similarity": group_similarity,
        "group_present": group_present,
        "effective_groups": effective,
        "diversity_factor": diversity_factor,
    }


def _p95_normalize(field: np.ndarray) -> np.ndarray:
    """Frame-relative normalize: p95 → 1.0, clip to [0, 1]."""
    p95 = float(np.percentile(field, 95))
    if p95 > 1e-6:
        return np.clip(field / p95, 0.0, 1.0).astype(np.float32)
    return np.zeros(field.shape[:2], dtype=np.float32)


def box_blurred_edge_density(gray: np.ndarray) -> np.ndarray:
    """Sobel magnitude + 7×7 box blur, p95-normalized."""
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge_mag = cv2.magnitude(gx, gy)
    return _p95_normalize(cv2.blur(edge_mag, _EDGE_BLUR_KSIZE))


def _nearest_upscale(heatmap: np.ndarray, scale: int, out_h: int, out_w: int) -> np.ndarray:
    """Repeat each pooled cell to recover full-frame heatmap coordinates."""
    if scale <= 1:
        return heatmap.astype(np.float32)
    upscaled = np.repeat(np.repeat(heatmap, scale, axis=0), scale, axis=1)
    return upscaled[:out_h, :out_w].astype(np.float32)


def _dedup_blobs_by_sprite_size(
    blobs: list[tuple[int, int, float, tuple[int, int, int, int]]],
    avg_width: int,
    avg_height: int,
) -> list[tuple[int, int, float, tuple[int, int, int, int]]]:
    """Keep strongest peak when centers fall within ~sprite size of each other."""
    min_dist = max(1.0, min(avg_width, avg_height) * _BLOB_DEDUP_SIZE_FRAC)
    min_dist_sq = min_dist * min_dist
    kept: list[tuple[int, int, float, tuple[int, int, int, int]]] = []
    for blob in sorted(blobs, key=lambda item: item[2], reverse=True):
        cx, cy, _score, _bbox = blob
        if all(
            (cx - kx) * (cx - kx) + (cy - ky) * (cy - ky) >= min_dist_sq
            for kx, ky, _ks, _kb in kept
        ):
            kept.append(blob)
    return kept


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
        # Debug A/B: False restores production path without diversity.
        self.use_palette_diversity = bool(config.get("usePaletteDiversity", True))

    def _center_scales(self, frame_width: int) -> list[float]:
        return [
            s for s in self.center_scales
            if s >= self.small_scale_cutoff or frame_width >= self.small_scale_min_frame_width
        ]

    def _work_bgr(self, frame_bgr: np.ndarray, downscale: int) -> np.ndarray:
        if downscale > 1:
            fh, fw = frame_bgr.shape[:2]
            return cv2.resize(
                frame_bgr,
                (fw // downscale, fh // downscale),
                interpolation=cv2.INTER_NEAREST,
            )
        return frame_bgr

    def _finish_heatmap(
        self,
        sprite: np.ndarray,
        work_bgr: np.ndarray,
        descriptor: MobDescriptor,
        downscale: int,
        frame_shape: tuple[int, int],
    ) -> np.ndarray:
        gray = cv2.cvtColor(work_bgr, cv2.COLOR_BGR2GRAY)
        edge_density = box_blurred_edge_density(gray)
        sprite = sprite * (np.float32(0.5) + np.float32(0.5) * edge_density)

        w = max(3, int(round(descriptor.avg_width * 0.8 / downscale)) | 1)
        h = max(3, int(round(descriptor.avg_height * 0.8 / downscale)) | 1)
        final = cv2.GaussianBlur(sprite, (w, h), 0)

        if downscale > 1:
            final = _nearest_upscale(final, downscale, frame_shape[0], frame_shape[1])
        return final

    def build_sprite_heatmap(
        self,
        frame_bgr: np.ndarray,
        descriptor: MobDescriptor,
        downscale: int = 1,
    ) -> np.ndarray:
        """Build sprite palette heatmap with edge-density boost.

        Returns sprite_heatmap at full frame resolution.
        """
        frame_shape = frame_bgr.shape[:2]
        work_bgr = self._work_bgr(frame_bgr, downscale)

        # --- 1. Weighted sprite-palette-distance heatmap ---
        if self.use_palette_diversity:
            base_sprite, similarity = weighted_sprite_palette_heatmap(
                work_bgr,
                descriptor,
                self.max_sprite_palette_distance,
                return_similarity=True,
            )
            sprite, _div_maps = apply_palette_diversity(
                base_sprite,
                similarity,
                descriptor.match_palette_groups,
                avg_width=descriptor.size.avg_width,
                avg_height=descriptor.size.avg_height,
                downscale=downscale,
            )
        else:
            sprite = weighted_sprite_palette_heatmap(
                work_bgr,
                descriptor,
                self.max_sprite_palette_distance,
            )

        # --- 2. Edge-density boost ---
        # --- 3. GaussianBlur ---
        # --- 4. Upscale ---
        return self._finish_heatmap(sprite, work_bgr, descriptor, downscale, frame_shape)

    def palette_diversity_debug(
        self,
        frame_bgr: np.ndarray,
        descriptor: MobDescriptor,
        downscale: int = 1,
    ) -> dict[str, object]:
        """Intermediate diversity / heatmap maps for visualization (full-frame)."""
        frame_shape = frame_bgr.shape[:2]
        work_bgr = self._work_bgr(frame_bgr, downscale)

        base_sprite, similarity = weighted_sprite_palette_heatmap(
            work_bgr,
            descriptor,
            self.max_sprite_palette_distance,
            return_similarity=True,
        )
        sprite_div, div_maps = apply_palette_diversity(
            base_sprite,
            similarity,
            descriptor.match_palette_groups,
            avg_width=descriptor.size.avg_width,
            avg_height=descriptor.size.avg_height,
            downscale=downscale,
        )

        final_new = self._finish_heatmap(
            sprite_div, work_bgr, descriptor, downscale, frame_shape,
        )
        final_old = self._finish_heatmap(
            base_sprite, work_bgr, descriptor, downscale, frame_shape,
        )

        def up(field: np.ndarray) -> np.ndarray:
            if downscale <= 1:
                return field.astype(np.float32)
            return _nearest_upscale(field, downscale, frame_shape[0], frame_shape[1])

        return {
            "base_sprite": up(base_sprite),
            "group_similarity": [up(g) for g in div_maps["group_similarity"]],
            "group_present": [up(g) for g in div_maps["group_present"]],
            "effective_groups": up(div_maps["effective_groups"]),
            "diversity_factor": up(div_maps["diversity_factor"]),
            "heatmap_after_diversity": up(sprite_div),
            "final_heatmap_new": final_new,
            "final_heatmap_old": final_old,
            "downscale": downscale,
        }

    def top_centers(
        self,
        heatmap: np.ndarray,
        descriptor: MobDescriptor,
    ) -> list[tuple[int, int, float, tuple[int, int, int, int]]]:
        """Find distinct hot regions via connected components.

        Near-duplicate peaks within ~0.85× min(sprite dims) are suppressed.
        """
        if heatmap.size == 0:
            return []

        avg_width = int(descriptor.avg_width)
        avg_height = int(descriptor.avg_height)

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
            if stats[label, cv2.CC_STAT_AREA] < 6:
                continue
            blob = self._blob_from_mask(heatmap, labels == label)
            if blob is not None:
                raw.append(blob)

        kept = _dedup_blobs_by_sprite_size(raw, avg_width, avg_height)
        return kept[: self.max_centers]

    def _blob_from_mask(
        self,
        heatmap: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[int, int, float, tuple[int, int, int, int]] | None:
        area = int(mask.sum())
        if area < 6:
            return None

        vals = heatmap[mask]
        peak_score = float(vals.max())
        weights = vals.astype(np.float32)
        ys, xs = np.where(mask)
        if float(weights.sum()) > 0.0:
            cx = int(np.average(xs, weights=weights))
            cy = int(np.average(ys, weights=weights))
        else:
            cx = int(round(float(xs.mean())))
            cy = int(round(float(ys.mean())))

        x0 = int(xs.min())
        y0 = int(ys.min())
        comp_bbox = (x0, y0, int(xs.max()) - x0 + 1, int(ys.max()) - y0 + 1)
        return (cx, cy, peak_score, comp_bbox)
