"""Vectorized descriptor sprite heatmap and blob finding.

Pipeline: weighted sprite palette heatmap → body-cluster diversity
          (required groups + optional-group boost) → edge boost →
          GaussianBlur → connected components → blob centers.
"""

from __future__ import annotations

import cv2
import numpy as np

from pybot.recognition.detector.descriptors.descriptor import ColorCluster, MobDescriptor

_EDGE_BLUR_KSIZE = (7, 7)

# Local body-cluster diversity (soft heatmap reshape after palette heat).
# Coverage window ≈ 0.6 × mob size; same window used for group presence in
# the hard color-structure gate.
_PRESENCE_SIMILARITY_LOW = np.float32(0.35)
_PRESENCE_SIMILARITY_HIGH = np.float32(0.75)
_MIN_GROUP_AREA_FRACTION = np.float32(0.01)
_COVERAGE_SIZE_FRAC = 0.6
# Body-strong pixel (sim >= this) feeds local body density for soft diversity.
_BODY_STRONG_SIM = 0.5
# Soft diversity break-even for local strong-body density. Kept above the hard
# gate floor (minBodyClusterStrong) so weak body-tinted fringes stay suppressed
# and impostors near the floor do not get a free boost.
_BODY_DIVERSITY_BREAK_EVEN = np.float32(0.07)
_BODY_DIVERSITY_BOOST_SLOPE = np.float32(5.0)
_BODY_DIVERSITY_MAX_FACTOR = np.float32(1.75)
_BODY_DIVERSITY_SUPPRESS_POWER = np.float32(2.0)
# Optional Lab groups (eyes / intermittents) never raise the diversity bar,
# but their local presence multiplies heat up to this extra gain when the
# region already clears body + required-group bars.
_OPTIONAL_GROUP_BOOST = np.float32(0.35)
# Body density uses mass clusters only (fraction >= this). Low-mass cream /
# highlight accents are shared by impostors and must not inflate body_strong.
_BODY_MASS_MIN_FRACTION = 0.15
# Near-duplicate blob suppress radius as a fraction of min(sprite w, h).
_BLOB_DEDUP_SIZE_FRAC = 0.85
# A close pair can form one connected heat component after blur/upscale even
# though the sprites are separate. Only components clearly larger than one
# descriptor-sized sprite are eligible for the narrow two-peak recovery below.
_OVERSIZED_SPLIT_DIM_RATIO = 1.65
_OVERSIZED_SPLIT_PEAK_RATIO = 0.78
_OVERSIZED_SPLIT_NMS_RADIUS_FRAC = 0.55
_OVERSIZED_SPLIT_MIN_SEPARATION_FRAC = 0.70
# Ignore heat CCs smaller than this many pixels (noise speckles).
_MIN_BLOB_COMPONENT_AREA = 6
# Gaussian blur kernel ≈ this fraction of sprite size at work resolution.
_GAUSSIAN_BLUR_SIZE_FRAC = 0.8
# Cap kernel at this fraction of work-res sprite size so small sprites
# (Creamy 48 px → 24 px at downscale 2) are not over-blurred. A 19 px
# kernel on a 24 px field smears heat across the full frame.
_GAUSSIAN_BLUR_MAX_WORK_SIZE_FRAC = 0.40
# Edge-density mixes as 0.5 + 0.5 * normalized edge map.
_EDGE_DENSITY_BASE = np.float32(0.5)
_EDGE_DENSITY_WEIGHT = np.float32(0.5)

# Palette-distance chunk budget in output elements. Row chunks of ~1 MiB of
# output plus small per-chunk temporaries keep the peak working set tiny so
# concurrent numpy callers (discovery + tracking + OCR) do not thrash cache on
# giant (N, C) temporaries — the post-sit spike that made single scans 30–50×
# slower on the live 16-core machine.
_PALETTE_DIST_CHUNK_ELEMENTS = 8192 * 32  # 1 MiB of float32 output


def _pixel_dot(pixels: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Dot each BGR pixel with a 3-vector without entering BLAS.

    ``numpy.dot`` on a very wide pixel matrix can hand each tiny 3-column
    multiply to the process BLAS pool. Discovery, tracking, and OCR can then
    all trigger native pools at once even though OpenCV itself is configured
    single-threaded. The explicit three-channel expression has bounded,
    deterministic native work and avoids that post-stand oversubscription.
    """
    return (
        pixels[:, 0] * vector[0]
        + pixels[:, 1] * vector[1]
        + pixels[:, 2] * vector[2]
    )


def _pixel_palette_dot(
    pixels: np.ndarray,
    palette: np.ndarray,
) -> np.ndarray:
    """Return ``pixels @ palette.T`` without invoking the BLAS thread pool."""
    return (
        pixels[:, None, 0] * palette[None, :, 0]
        + pixels[:, None, 1] * palette[None, :, 1]
        + pixels[:, None, 2] * palette[None, :, 2]
    )


def _palette_dist_sq(pixels: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Squared Euclidean distances (N, C) via |p-c|² = |p|² + |c|² - 2p·c.

    Uses a bounded three-channel multiply and avoids a (N, C, 3)
    intermediate. The matrix is built in bounded row chunks so the peak
    allocation is ~1 MiB regardless
    of frame size; per-row arithmetic is identical to the single-GEMM version.
    """
    n_pixels = pixels.shape[0]
    n_colors = palette.shape[0]
    out = np.empty((n_pixels, n_colors), dtype=np.float32)
    if n_pixels <= 0 or n_colors <= 0:
        return out
    c_norm = np.sum(palette * palette, axis=1, keepdims=True)  # (C, 1)
    chunk = max(1, _PALETTE_DIST_CHUNK_ELEMENTS // n_colors)
    neg2 = np.float32(-2.0)
    zero = np.float32(0.0)
    for start in range(0, n_pixels, chunk):
        pc = pixels[start : start + chunk]
        p_norm = np.sum(pc * pc, axis=1, keepdims=True)  # (chunk, 1)
        dist_sq = _pixel_palette_dot(pc, palette)  # (chunk, C)
        dist_sq *= neg2
        dist_sq += p_norm
        dist_sq += c_norm.T
        np.maximum(dist_sq, zero, out=dist_sq)  # clamp fp noise
        out[start : start + chunk] = dist_sq
    return out


def _palette_min_dist_sq_gemv(pixels: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Per-pixel min squared distance to any palette color, shape (N,).

    One GEMV per palette color instead of a single (N, C) GEMM: every
    intermediate is a contiguous (N,) vector, so concurrent callers never fight
    over a large shared temporary. Measured on the live 16-core machine: ~2×
    faster single-threaded and ~2.6× faster under 8-way concurrency than the
    materializing GEMM, with bit-identical results.
    """
    n_pixels = pixels.shape[0]
    n_colors = palette.shape[0]
    best = np.full(n_pixels, np.inf, dtype=np.float32)
    if n_pixels <= 0 or n_colors <= 0:
        return best
    p_sq = np.sum(pixels * pixels, axis=1)
    c_norms = np.sum(palette * palette, axis=1)
    neg2 = np.float32(-2.0)
    zero = np.float32(0.0)
    for j in range(n_colors):
        col = _pixel_dot(pixels, palette[j])
        col *= neg2
        col += c_norms[j]
        col += p_sq
        np.maximum(col, zero, out=col)
        np.minimum(best, col, out=best)
    return best


def _palette_argmin_dist_sq_gemv(
    pixels: np.ndarray,
    palette: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel nearest palette index and min squared distance.

    Strict less-than keeps the first (lowest-index) color on ties, matching
    ``argmin(axis=1)`` semantics. Same bounded-temporary profile as
    ``_palette_min_dist_sq_gemv``.
    """
    n_pixels = pixels.shape[0]
    n_colors = palette.shape[0]
    idx = np.zeros(n_pixels, dtype=np.int32)
    best = np.full(n_pixels, np.inf, dtype=np.float32)
    if n_pixels <= 0 or n_colors <= 0:
        return idx, best
    p_sq = np.sum(pixels * pixels, axis=1)
    c_norms = np.sum(palette * palette, axis=1)
    neg2 = np.float32(-2.0)
    zero = np.float32(0.0)
    for j in range(n_colors):
        col = _pixel_dot(pixels, palette[j])
        col *= neg2
        col += c_norms[j]
        col += p_sq
        np.maximum(col, zero, out=col)
        closer = col < best
        best[closer] = col[closer]
        idx[closer] = j
    return idx, best


def palette_heatmap(frame_bgr: np.ndarray, clusters: list[ColorCluster]) -> np.ndarray:
    """BGR Euclidean heatmap against ColorCluster centers (tracking/opacity)."""
    return _multi_cluster_match_max(frame_bgr.astype(np.float32), clusters)


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
    max_dist = np.float32(max(max_distance, 1.0))
    min_dist_sq = _palette_min_dist_sq_gemv(pixels, palette)
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

    Distance computed via |p-c|² = |p|² + |c|² - 2p·c. The common (non-
    similarity) path folds the weighted max with per-color GEMVs so no (N, C)
    matrix is materialized; ``return_similarity`` builds the (N, C) distance
    matrix in bounded chunks for palette-group coverage. Both keep peak
    temporaries small so concurrent detector/OCR workers do not thrash cache.

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

    # --- nearest-color index → per-color rarity weights ---
    # GEMV loop keeps the nearest-color pass free of a full (N, C) temporary;
    # bit-identical to ``dist_sq.argmin(axis=1)`` (first-wins on ties).
    nearest_idx = _palette_argmin_dist_sq_gemv(pixels, palette)[0]
    palette_match_count = np.bincount(nearest_idx, minlength=n_colors).astype(np.float32)
    scene_fraction = palette_match_count / np.float32(max(n_pixels, 1))
    rarity = np.float32(1.0) / np.sqrt(scene_fraction + np.float32(1e-6))
    median_rarity = float(np.median(rarity))
    if median_rarity > 0.0:
        rarity = (rarity / np.float32(median_rarity)).astype(np.float32)
    rarity = np.clip(rarity, np.float32(0.25), np.float32(2.0))

    combined_w = (rarity * _palette_descriptor_weights(descriptor)).astype(np.float32)

    if return_similarity:
        # Group coverage needs the full per-color similarity map: build the
        # (N, C) distance matrix in bounded chunks, then transform in place.
        dist_sq = _palette_dist_sq(pixels, palette)
        np.sqrt(dist_sq, out=dist_sq)
        dist_sq /= max_dist
        np.subtract(np.float32(1.0), dist_sq, out=dist_sq)
        np.clip(dist_sq, np.float32(0.0), np.float32(1.0), out=dist_sq)
        similarity = dist_sq  # (N, C) unweighted
        best_weighted = (similarity * combined_w).max(axis=1)
        base_sprite = best_weighted.reshape(shape_hw)
        return base_sprite, similarity.reshape(*shape_hw, n_colors).astype(np.float32)

    # Non-similarity path: fold the weighted max per palette color with GEMVs
    # so no (N, C) matrix is materialized at all.
    out = np.full(n_pixels, np.float32(-np.inf), dtype=np.float32)
    p_sq = np.sum(pixels * pixels, axis=1)
    c_norms = np.sum(palette * palette, axis=1)
    neg2 = np.float32(-2.0)
    zero = np.float32(0.0)
    inv_max_dist = np.float32(1.0) / max_dist
    one = np.float32(1.0)
    for j in range(n_colors):
        col = _pixel_dot(pixels, palette[j])
        col *= neg2
        col += c_norms[j]
        col += p_sq
        np.maximum(col, zero, out=col)
        np.sqrt(col, out=col)
        col *= inv_max_dist
        np.subtract(one, col, out=col)
        np.clip(col, np.float32(0.0), np.float32(1.0), out=col)
        col *= combined_w[j]
        np.maximum(out, col, out=out)
    return out.reshape(shape_hw)


# Hard color-structure gate: group counts as present when peak local presence
# reaches this (1.0 = coverage window meets _MIN_GROUP_AREA_FRACTION).
_GROUP_PRESENT_PEAK_MIN = 1.0


def _group_presence_maps(
    similarity_hwc: np.ndarray,
    groups: list[list[int]],
    ksize: tuple[int, int],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Per-group max-sim and local presence maps (hard color-structure gate)."""
    denom = _PRESENCE_SIMILARITY_HIGH - _PRESENCE_SIMILARITY_LOW
    group_present: list[np.ndarray] = []
    for indices in groups:
        idx = np.asarray(indices, dtype=np.int32)
        g_sim = similarity_hwc[:, :, idx].max(axis=2).astype(np.float32)
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
    return [], group_present


def _group_presence_maps_from_frame(
    frame_bgr: np.ndarray,
    descriptor: MobDescriptor,
    groups: list[list[int]],
    ksize: tuple[int, int],
    max_distance: float,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Build group presence maps without a full-frame ``(H, W, palette)`` map.

    The discovery diversity pass used to retain every palette similarity for the
    whole ROI. For a 1024x1024 Anubis frame that is a large temporary tensor and
    it competed with tracking immediately after sit recovery. Each group only
    needs its maximum similarity and local presence, so compute one 2-D map per
    group and release it before moving to the next group.
    """
    denom = _PRESENCE_SIMILARITY_HIGH - _PRESENCE_SIMILARITY_LOW
    group_present: list[np.ndarray] = []
    palette = descriptor.match_palette_bgr
    for indices in groups:
        group_palette = [palette[index] for index in indices]
        g_sim = sprite_palette_heatmap(
            frame_bgr,
            group_palette,
            max_distance,
        )
        matched = np.clip(
            (g_sim - _PRESENCE_SIMILARITY_LOW) / denom,
            0.0,
            1.0,
        ).astype(np.float32)
        local_presence = cv2.boxFilter(
            matched, ddepth=-1, ksize=ksize, normalize=True,
        )
        group_present.append(
            np.clip(
                local_presence / _MIN_GROUP_AREA_FRACTION,
                0.0,
                1.0,
            ).astype(np.float32)
        )
    return [], group_present


def mass_body_clusters(descriptor: MobDescriptor) -> list[ColorCluster]:
    """Dominant + supporting clusters with enough sprite mass.

    Low-fraction cream/highlight accents are excluded so body density tracks
    the mob's real body, not impostor-shared specular tones.
    """
    clusters = [descriptor.dominant_color, *descriptor.supporting_colors]
    mass = [c for c in clusters if float(c.fraction) >= _BODY_MASS_MIN_FRACTION]
    return mass if mass else clusters[:1]


def required_groups_structure(
    crop_bgr: np.ndarray,
    descriptor: MobDescriptor,
    max_distance: float,
    *,
    downscale: int = 1,
    presence_peak_min: float = float(_GROUP_PRESENT_PEAK_MIN),
    body_best_full: np.ndarray | None = None,
    body_best_downscale: int = 0,
    crop_x: int = 0,
    crop_y: int = 0,
) -> tuple[int, float, float, float]:
    """Palette structure in *crop_bgr*.

    Returns ``(present_count, second_share, match_coverage, body_strong)``.

    ``body_strong`` is always measured on the full-resolution *crop_bgr*.
    A downscaled ``body_best_full`` cache must not be used here: nearest-neighbour
    work-res sampling inflates the strong-pixel fraction (one hot work pixel
    covers a 2×2 full-res block) and lets gray-world impostors clear the
    per-mob floor (see 0WildRose_Gray).

    ``body_best_full`` / ``body_best_downscale`` / ``crop_x`` / ``crop_y`` are
    accepted for call-site compatibility but ignored for ``body_strong``.
    """
    del body_best_full, body_best_downscale, crop_x, crop_y
    empty = (0, 0.0, 0.0, 0.0)
    groups = list(descriptor.match_palette_required_groups)
    if (
        not groups
        or crop_bgr is None
        or crop_bgr.size == 0
        or not descriptor.match_palette_bgr
    ):
        return empty
    _base, similarity_hwc = weighted_sprite_palette_heatmap(
        crop_bgr,
        descriptor,
        max_distance,
        return_similarity=True,
    )
    if similarity_hwc.size == 0:
        return empty
    ksize = _coverage_window(
        float(descriptor.avg_width),
        float(descriptor.avg_height),
        downscale,
    )
    _sims, present_maps = _group_presence_maps(similarity_hwc, groups, ksize)
    present_count = sum(
        1 for present in present_maps if float(present.max()) >= presence_peak_min
    )

    # Derive group match masks directly from similarity_hwc (already computed
    # via weighted_sprite_palette_heatmap). similarity > 0.0 ⇔ pixel is within
    # max_distance of a palette color in that group.
    match_mats: list[np.ndarray] = []
    for indices in groups:
        idx_arr = np.asarray(indices, dtype=np.int32)
        group_max_sim = similarity_hwc[:, :, idx_arr].max(axis=2)
        match_mats.append((group_max_sim > 0.0).reshape(-1))
    matched = np.stack(match_mats, axis=1)
    any_match = matched.any(axis=1)
    match_coverage = float(any_match.mean()) if any_match.size else 0.0

    # Full-resolution crop measurement — matches descriptor build calibration
    # (opaque sprite pixels at native scale).
    body_clusters = mass_body_clusters(descriptor)
    if body_clusters:
        body_best = _multi_cluster_match_max(
            crop_bgr.astype(np.float32), body_clusters,
        )
        body_strong = float((body_best >= _BODY_STRONG_SIM).mean())
    else:
        body_strong = 0.0

    if int(any_match.sum()) <= 0:
        return present_count, 0.0, match_coverage, body_strong
    shares = matched[any_match].sum(axis=0).astype(np.float32)
    shares /= np.float32(any_match.sum())
    if shares.size < 2:
        return present_count, 0.0, match_coverage, body_strong
    ordered = np.sort(shares)[::-1]
    return present_count, float(ordered[1]), match_coverage, body_strong



def _multi_cluster_match_max(bgr_f: np.ndarray, clusters: list[ColorCluster]) -> np.ndarray:
    if not clusters:
        return np.zeros(bgr_f.shape[:2], dtype=np.float32)
    pixels = bgr_f.reshape(-1, 3).astype(np.float32, copy=False)
    centers = np.asarray([cluster.bgr for cluster in clusters], dtype=np.float32)
    max_dists = np.asarray(
        [max(float(cluster.max_distance), 1.0) for cluster in clusters],
        dtype=np.float32,
    )
    # Per-cluster GEMV + running max: same arithmetic as the (N, C) GEMM but
    # only contiguous (N,) intermediates (see _palette_min_dist_sq_gemv).
    n_pixels = pixels.shape[0]
    n_clusters = centers.shape[0]
    best = np.full(n_pixels, -np.inf, dtype=np.float32)
    if n_pixels <= 0 or n_clusters <= 0:
        return np.clip(best, 0.0, 1.0).reshape(bgr_f.shape[:2]).astype(np.float32)
    p_sq = np.sum(pixels * pixels, axis=1)
    c_norms = np.sum(centers * centers, axis=1)
    neg2 = np.float32(-2.0)
    zero = np.float32(0.0)
    one = np.float32(1.0)
    for j in range(n_clusters):
        col = _pixel_dot(pixels, centers[j])
        col *= neg2
        col += c_norms[j]
        col += p_sq
        np.maximum(col, zero, out=col)
        np.sqrt(col, out=col)
        col /= max_dists[j]
        col *= np.float32(-1.0)
        col += one
        np.maximum(best, col, out=best)
    return np.clip(best, 0.0, 1.0).reshape(bgr_f.shape[:2]).astype(np.float32)


def apply_body_cluster_diversity(
    base_sprite: np.ndarray,
    frame_bgr: np.ndarray,
    descriptor: MobDescriptor,
    *,
    similarity_hwc: np.ndarray | None = None,
    min_body_strong: float,
    min_required_groups: int = 2,
    avg_width: float,
    avg_height: float,
    downscale: int = 1,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Soft reshape: boost real body, press impostors and mono terrain.

    Local signals in the coverage window:

    - weak body density → suppress (Kobold-like impostors)
    - below required Lab-group majority → suppress (mono / body-tinted terrain)
    - strong body AND required-group majority → boost, gated by palette heat
    - optional groups (eyes / intermittents) never raise the bar; when the
      region already clears body+required bars, their presence multiplies heat

    Returns ``(sprite_with_diversity, debug_maps)``.
    """
    h, w = base_sprite.shape[:2]
    ones = np.ones((h, w), dtype=np.float32)
    body_clusters = mass_body_clusters(descriptor)
    required_groups = list(descriptor.match_palette_required_groups)
    optional_groups = list(descriptor.match_palette_optional_groups)
    if (not body_clusters and not required_groups) or frame_bgr.size == 0:
        return base_sprite.copy(), {
            "body_best": np.zeros((h, w), dtype=np.float32),
            "local_body_strong": np.zeros((h, w), dtype=np.float32),
            "effective_groups": np.zeros((h, w), dtype=np.float32),
            "optional_effective": np.zeros((h, w), dtype=np.float32),
            "diversity_factor": ones,
        }

    ksize = _coverage_window(avg_width, avg_height, downscale)

    if body_clusters:
        bgr_f = frame_bgr.astype(np.float32)
        body_best = _multi_cluster_match_max(bgr_f, body_clusters)
        strong = (body_best >= _BODY_STRONG_SIM).astype(np.float32)
        local_body = cv2.boxFilter(
            strong, ddepth=-1, ksize=ksize, normalize=True,
        ).astype(np.float32)
    else:
        body_best = np.zeros((h, w), dtype=np.float32)
        local_body = ones.copy()

    n_required = len(required_groups)
    if n_required > 0:
        if similarity_hwc is not None and similarity_hwc.size > 0:
            _sims, req_present = _group_presence_maps(
                similarity_hwc, required_groups, ksize,
            )
        else:
            _sims, req_present = _group_presence_maps_from_frame(
                frame_bgr,
                descriptor,
                required_groups,
                ksize,
                float(descriptor.max_sprite_palette_distance),
            )
        required_effective = np.zeros((h, w), dtype=np.float32)
        for present in req_present:
            required_effective += present
        majority = n_required // 2 + 1
        group_bar = np.float32(
            max(1, min(n_required, max(int(min_required_groups), majority)))
        )
    else:
        required_effective = np.full((h, w), np.float32(n_required or 1), dtype=np.float32)
        group_bar = np.float32(1.0)

    n_optional = len(optional_groups)
    optional_effective = np.zeros((h, w), dtype=np.float32)
    if n_optional > 0:
        if similarity_hwc is not None and similarity_hwc.size > 0:
            _opt_sims, opt_present = _group_presence_maps(
                similarity_hwc, optional_groups, ksize,
            )
        else:
            _opt_sims, opt_present = _group_presence_maps_from_frame(
                frame_bgr,
                descriptor,
                optional_groups,
                ksize,
                float(descriptor.max_sprite_palette_distance),
            )
        for present in opt_present:
            optional_effective += present

    body_bar = np.float32(
        max(float(min_body_strong), float(_BODY_DIVERSITY_BREAK_EVEN), 1e-6)
    )
    body_ok = local_body >= body_bar
    groups_ok = required_effective >= group_bar

    body_suppress = np.clip(local_body / body_bar, 0.0, 1.0).astype(np.float32)
    np.power(body_suppress, _BODY_DIVERSITY_SUPPRESS_POWER, out=body_suppress)
    group_suppress = np.clip(
        required_effective / group_bar, 0.0, 1.0,
    ).astype(np.float32)
    np.power(group_suppress, _BODY_DIVERSITY_SUPPRESS_POWER, out=group_suppress)

    boost = (
        np.float32(1.0)
        + _BODY_DIVERSITY_BOOST_SLOPE * (local_body - body_bar)
    ).astype(np.float32)
    np.clip(boost, 1.0, float(_BODY_DIVERSITY_MAX_FACTOR), out=boost)

    # Press if body weak or required groups incomplete; boost only when both pass.
    body_factor = np.where(
        body_ok & groups_ok,
        boost,
        np.minimum(body_suppress, group_suppress),
    ).astype(np.float32)

    # Palette gate: limit diversity boost to pixels with meaningful palette
    # heat in their local neighborhood. Uses the same coverage window for
    # a local-max baseline instead of a frame-global peak (which is fragile
    # when a single bright impostor inflates the threshold frame-wide).
    local_peak = cv2.boxFilter(
        base_sprite, ddepth=-1, ksize=ksize, normalize=True,
    )
    local_peak = np.maximum(local_peak, np.float32(1e-6))
    palette_gate = np.clip(
        base_sprite / (local_peak * np.float32(0.25)), 0.0, 1.0,
    ).astype(np.float32)
    diversity_factor = np.where(
        body_factor < np.float32(1.0),
        body_factor,
        np.float32(1.0) + (body_factor - np.float32(1.0)) * palette_gate,
    ).astype(np.float32)

    # Optional palette diversity: never suppress when absent; amplify real
    # candidates that already clear body + required-group bars. palette_gate
    # keeps bare optional-color terrain from inventing heat peaks.
    if n_optional > 0:
        optional_presence = (
            optional_effective / np.float32(n_optional)
        ).astype(np.float32)
        optional_boost = (
            np.float32(1.0)
            + _OPTIONAL_GROUP_BOOST * optional_presence * palette_gate
        ).astype(np.float32)
        diversity_factor = np.where(
            body_ok & groups_ok,
            diversity_factor * optional_boost,
            diversity_factor,
        ).astype(np.float32)

    sprite = (base_sprite * diversity_factor).astype(np.float32)
    return sprite, {
        "body_best": body_best,
        "local_body_strong": local_body,
        "effective_groups": required_effective,
        "optional_effective": optional_effective,
        "diversity_factor": diversity_factor,
        "palette_gate": palette_gate,
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
        self.min_body_cluster_strong = float(config["minBodyClusterStrong"])
        self.min_required_groups = int(config["minRequiredPaletteGroups"])
        # Cached full-frame body map from the last build_sprite_heatmap call
        # (at work resolution). Keyed by descriptor identity to avoid cross-mob
        # poisoning when detect() is called for different mobs on the same frame.
        self._last_body_best: np.ndarray | None = None
        self._last_body_downscale: int = 0
        self._last_body_descriptor_id: int = 0

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
                (max(1, fw // downscale), max(1, fh // downscale)),
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
        sprite = sprite * (_EDGE_DENSITY_BASE + _EDGE_DENSITY_WEIGHT * edge_density)

        work_w = descriptor.avg_width / max(downscale, 1)
        work_h = descriptor.avg_height / max(downscale, 1)
        w = max(3, int(round(work_w * _GAUSSIAN_BLUR_SIZE_FRAC)) | 1)
        h = max(3, int(round(work_h * _GAUSSIAN_BLUR_SIZE_FRAC)) | 1)
        # Cap at 50 % of work-res sprite size so small sprites are not
        # over-blurred into featureless smears.
        cap_w = max(3, int(round(work_w * _GAUSSIAN_BLUR_MAX_WORK_SIZE_FRAC)) | 1)
        cap_h = max(3, int(round(work_h * _GAUSSIAN_BLUR_MAX_WORK_SIZE_FRAC)) | 1)
        w = min(w, cap_w)
        h = min(h, cap_h)
        final = cv2.GaussianBlur(sprite, (w, h), 0)

        if downscale > 1:
            final = _nearest_upscale(final, downscale, frame_shape[0], frame_shape[1])
        return final

    def build_sprite_heatmap(
        self,
        frame_bgr: np.ndarray,
        descriptor: MobDescriptor,
        downscale: int = 1,
        *,
        fast_static: bool = False,
    ) -> np.ndarray:
        """Build sprite palette heatmap with edge-density boost.

        ``fast_static`` is used only for the modified, single-frame sprite
        assets. Their palette is already distinctive and deterministic, so
        rarity weighting and body-diversity maps add cost without adding an
        identity signal. The downstream geometry and silhouette gates still
        validate candidates.

        Returns sprite_heatmap at full frame resolution.
        """
        frame_shape = frame_bgr.shape[:2]
        work_bgr = self._work_bgr(frame_bgr, downscale)

        # --- 1. Sprite-palette-distance heatmap ---
        if fast_static:
            # Static GRF mode does not need scene-relative rarity or the
            # full-frame body/group diversity maps. Avoiding those extra
            # palette passes is important on the 1024x1024 hunt ROI.
            sprite = sprite_palette_heatmap(
                work_bgr,
                descriptor.match_palette_bgr,
                descriptor.max_sprite_palette_distance,
            )
            self._last_body_best = None
            self._last_body_downscale = 0
            self._last_body_descriptor_id = 0
        elif descriptor.use_body_cluster_diversity:
            # Keep the weighted base heatmap, but do not retain a full-frame
            # per-palette similarity tensor. Diversity only needs one 2-D
            # presence map per color group; building those maps on demand keeps
            # the 1024x1024 Anubis path bounded after sit recovery.
            base_sprite = weighted_sprite_palette_heatmap(
                work_bgr,
                descriptor,
                descriptor.max_sprite_palette_distance,
            )
            sprite, div_maps = apply_body_cluster_diversity(
                base_sprite,
                work_bgr,
                descriptor,
                similarity_hwc=None,
                min_body_strong=self.min_body_cluster_strong,
                min_required_groups=self.min_required_groups,
                avg_width=descriptor.size.avg_width,
                avg_height=descriptor.size.avg_height,
                downscale=downscale,
            )
            self._last_body_best = div_maps["body_best"]
            self._last_body_downscale = downscale
            self._last_body_descriptor_id = id(descriptor)
        else:
            sprite = weighted_sprite_palette_heatmap(
                work_bgr,
                descriptor,
                descriptor.max_sprite_palette_distance,
            )
            self._last_body_best = None
            self._last_body_downscale = 0
            self._last_body_descriptor_id = 0

        # --- 2. Edge-density boost ---
        # --- 3. GaussianBlur ---
        # --- 4. Upscale ---
        return self._finish_heatmap(sprite, work_bgr, descriptor, downscale, frame_shape)

    def top_centers(
        self,
        heatmap: np.ndarray,
        descriptor: MobDescriptor,
    ) -> list[tuple[int, int, float, tuple[int, int, int, int]]]:
        """Find distinct hot regions via connected components.

        Near-duplicate peaks within ~0.85× min(sprite dims) are suppressed.
        A single oversized component is split only when it contains two strong,
        well-separated heat peaks; ordinary components keep their original
        connected-component bbox and behavior.
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
        split_groups: list[
            list[tuple[int, int, float, tuple[int, int, int, int]]]
        ] = []
        for label in range(1, num_labels):
            if stats[label, cv2.CC_STAT_AREA] < _MIN_BLOB_COMPONENT_AREA:
                continue
            mask = labels == label
            bbox = (
                int(stats[label, cv2.CC_STAT_LEFT]),
                int(stats[label, cv2.CC_STAT_TOP]),
                int(stats[label, cv2.CC_STAT_WIDTH]),
                int(stats[label, cv2.CC_STAT_HEIGHT]),
            )
            split = self._split_oversized_component(
                heatmap,
                mask,
                bbox,
                avg_width,
                avg_height,
            )
            if split is not None:
                # These two peaks came from one intentionally split component.
                # They have already passed the stricter split separation check;
                # do not run them through the normal 0.85-sprite dedup radius,
                # which would merge a valid close pair again.
                split_groups.append(split)
                continue
            blob = self._blob_from_mask(heatmap, mask)
            if blob is not None:
                raw.append(blob)

        kept = _dedup_blobs_by_sprite_size(raw, avg_width, avg_height)
        split_kept: list[tuple[int, int, float, tuple[int, int, int, int]]] = []
        split_dedup_sq = (
            min(avg_width, avg_height) * _BLOB_DEDUP_SIZE_FRAC
        ) ** 2
        # Connected-component label order is spatial, not confidence order.
        # Resolve cross-component duplicates strongest-first so the retained
        # candidate is deterministic and has the best evidence.
        for split_group in sorted(
            split_groups,
            key=lambda group: max(blob[2] for blob in group),
            reverse=True,
        ):
            prior_kept = (*kept, *split_kept)
            for split_blob in split_group:
                sx, sy, _score, _bbox = split_blob
                # Preserve the two peaks from one explicitly validated
                # component, but suppress a split peak that duplicates an
                # ordinary component or a split peak from another component.
                # This keeps separate close pairs intact without allowing
                # unusual heatmaps to create duplicate candidates.
                if all(
                    (sx - kx) * (sx - kx) + (sy - ky) * (sy - ky)
                    >= split_dedup_sq
                    for kx, ky, _ks, _kb in prior_kept
                ):
                    split_kept.append(split_blob)
        kept.extend(split_kept)
        kept.sort(key=lambda item: item[2], reverse=True)
        return kept[: self.max_centers]

    def _split_oversized_component(
        self,
        heatmap: np.ndarray,
        mask: np.ndarray,
        bbox: tuple[int, int, int, int],
        avg_width: int,
        avg_height: int,
    ) -> list[tuple[int, int, float, tuple[int, int, int, int]]] | None:
        """Recover two nearby sprites from one clearly oversized heat blob.

        This is deliberately not a general segmentation algorithm. It only
        activates when a component is at least 1.65 descriptor dimensions on one
        axis, then keeps at most two strong peaks separated by most of one
        sprite width/height. Each recovered peak gets a descriptor-sized search
        box, which gives the downstream geometry/color/silhouette gates an
        individual sprite-sized crop instead of the merged component.
        """
        _x, _y, width, height = bbox
        width_ratio = width / max(avg_width, 1)
        height_ratio = height / max(avg_height, 1)
        if max(width_ratio, height_ratio) < _OVERSIZED_SPLIT_DIM_RATIO:
            return None

        component_heat = np.where(mask, heatmap, 0.0).astype(np.float32)
        component_peak = float(component_heat.max())
        if component_peak <= 0.0:
            return None

        work = component_heat.copy()
        nms_radius = max(
            3,
            int(round(min(avg_width, avg_height) * _OVERSIZED_SPLIT_NMS_RADIUS_FRAC)),
        )
        peaks: list[tuple[int, int, float]] = []
        for _ in range(2):
            peak_y, peak_x = np.unravel_index(int(np.argmax(work)), work.shape)
            peak_score = float(work[peak_y, peak_x])
            if peak_score < component_peak * _OVERSIZED_SPLIT_PEAK_RATIO:
                break
            peaks.append((int(peak_x), int(peak_y), peak_score))
            cv2.circle(work, (int(peak_x), int(peak_y)), nms_radius, 0.0, -1)

        if len(peaks) != 2:
            return None

        dominant_axis = 1 if height_ratio >= width_ratio else 0
        separation = abs(
            peaks[0][dominant_axis] - peaks[1][dominant_axis]
        )
        if separation < min(avg_width, avg_height) * _OVERSIZED_SPLIT_MIN_SEPARATION_FRAC:
            return None

        frame_height, frame_width = heatmap.shape[:2]
        split: list[tuple[int, int, float, tuple[int, int, int, int]]] = []
        for peak_x, peak_y, peak_score in peaks:
            left = max(0, min(frame_width - avg_width, peak_x - avg_width // 2))
            top = max(0, min(frame_height - avg_height, peak_y - avg_height // 2))
            split.append((
                peak_x,
                peak_y,
                peak_score,
                (int(left), int(top), avg_width, avg_height),
            ))
        return split

    def _blob_from_mask(
        self,
        heatmap: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[int, int, float, tuple[int, int, int, int]] | None:
        area = int(mask.sum())
        if area < _MIN_BLOB_COMPONENT_AREA:
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
