"""Sprite heatmap + silhouette-gate mob detector.

Pipeline: sprite heatmap → blobs → geometry pre-gate → color-structure
pre-gate → silhouette gate → accept by heat score.
No RegionScorer, no structural pixels, and no heavyweight global center search.
"""

from __future__ import annotations

import inspect
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from pybot.recognition.detector.descriptors.descriptor import MobDescriptor
from pybot.recognition.detector.descriptors.descriptor_builder import DESCRIPTOR_VERSION
from pybot.recognition.detector.descriptors.layout_utils import (
    HARD_OCCUPANCY,
    best_silhouette_match,
    candidate_silhouette,
)
from pybot.recognition.detector.scoring.heatmap_detector import (
    HeatmapDetector,
    required_groups_structure,
    sprite_palette_heatmap,
)


REQUIRED_CONFIG_KEYS = {
    "discoveryHeatmapDownscale",
    "discoveryHeatmapDownscaleMinSide",
    "maxSpritePaletteDistance",
    "silhouettePaletteDistanceScale",
    "silhouetteHorizontalBridgeCells",
    "minSpritePaletteMatch",
    "gateRefUniqueIoU",
    "minSilhouetteRecall",
    "minSilhouettePrecision",
    "grfMinSilhouetteRecall",
    "grfMinSilhouettePrecision",
    "grfLocalTrackSkipNativeGate",
    "grfAspectBandScale",
    "minRequiredPaletteGroups",
    "minSecondPaletteGroupShare",
    "minRequiredPaletteCoverage",
    "minBodyClusterStrong",
    "topCandidateCenters",
    "minCenterHeat",
    "peakRelativeThreshold",
    "maxCandidates",
    "smallScaleMinFrameWidth",
    "smallScaleCutoff",
    "centerScales",
    "localTrackSearchRadiusPx",
    "localTrackMovingSearchRadiusPx",
    "localTrackSpriteRadiusMultiplier",
    "localTrackMaxSearchRadiusPx",
    "discoveryClusterRadiusPx",
    "trackDedupRadiusPx",
    "debugOutputDir",
    # track-removal keys (movement + discovery blob stationary + opacity death)
    "movementMoveThresholdPx",
    "movementStopThresholdPx",
    "deathOpacityBaselineSamples",
    "deathOpacityMinBaseline",
    "deathOpacityDecayRatio",
    "deathOpacityConfirmTicks",
    "deathRediscoveryCooldownMs",
    "deathSiteRadiusPx",
}

# Geometry pre-gate: heat-CC area must sit in [min_area_ratio, max_area_ratio]
# vs sprite area. Aspect uses per-mob descriptor.min/max_aspect_ratio
# (build-time sprite tight-bbox band, floored by MIN_ASPECT_FLOOR).
_GEOMETRY_AREA_SIL_FRAC_DIVISOR = 5.0
_GEOMETRY_AREA_MAX_RATIO = 2.0
# Heat CCs smaller than this multiple of the geometry min-area floor get the
# conservative body path (descriptor-sized crop) and a relative-heat check.
# 2× min_area = 2/5 of stable silhouette fraction — still below a full body
# footprint, where heat-crop body density is no longer inflated by tiny crops.
_BODY_STRONG_SMALL_HEAT_AREA_MIN_AREA_MULT = 2.0
# Small-CC relative heat vs frame peak must clear this multiple of
# peakRelativeThreshold (blob-formation floor). 1.5× is a mild lift above the
# weakest admissible blob so gray-world fringe peaks (e.g. 0.28 of peak) drop
# while multi-mob secondary TPs (~0.47+) remain.
_SMALL_HEAT_RELATIVE_PEAK_MULT = 1.8
# Post-silhouette extract body floor as a fraction of descriptor.min_body_cluster_strong.
# Extract is sprite-scale and tighter than the heat CC; 0.5× still rejects wrong-fill
# shapes while leaving margin for patchy mobs (Creamy TP sits ~0.79× full floor).
_EXTRACT_BODY_STRONG_FLOOR_FRAC = 0.75



# Modified sprite.grf assets are rendered from a single deterministic sprite
# source, so discovery always uses this fixed work-scale reduction. The scale
# is selected from the rendering mode, never from the selected mob descriptor.
_SPRITE_GRF_HEATMAP_DOWNSCALE = 4
# Non-GRF discovery keeps the generic small-sprite safety floor. GRF mode is
# intentionally exempt: its fixed rendering mode is the contract, regardless
# of descriptor dimensions.
_DOWNSCALE_MIN_WORK_RESOLUTION_PX = 16.0

# Extract / content-noise thresholds shared by silhouette gate control flow
# and the post-gate noisy_extract cleanup hook.
_EXTRACT_BLOAT_AREA_RATIO = 2.0
_CONTENT_NOISE_SOFT_HARD_RATIO = 2.0
# Full 16x16 hard fill = palette smear in a desc-sized window, not a sprite body.
_SOLID_FILL_HARD_FRACTION = 0.85

# Silhouette crop / morph / deform sizing.
_MIN_DESCRIPTOR_PX = 8
_MIN_EXTRACT_COMPONENT_PX = 4
_MIN_HORIZONTAL_BRIDGE_PX = 3
_DEFORM_RADIUS_SILHOUETTE_CELLS = 2
_MORPH_NEIGHBORHOOD_PX = 3
# Fixed morph kernels — recreating these every score_at was pure overhead.
_MORPH_ELLIPSE_3 = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE, (_MORPH_NEIGHBORHOOD_PX, _MORPH_NEIGHBORHOOD_PX),
)
_MORPH_RECT_3 = cv2.getStructuringElement(
    cv2.MORPH_RECT, (_MORPH_NEIGHBORHOOD_PX, _MORPH_NEIGHBORHOOD_PX),
)


@lru_cache(maxsize=64)
def _horizontal_bridge_kernel(bridge_px: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_RECT, (bridge_px, 1))


@lru_cache(maxsize=16)
def _ellipse_kernel(ksize: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))


def _descriptor_sprite_size_px(descriptor: MobDescriptor) -> tuple[int, int]:
    return (
        max(_MIN_DESCRIPTOR_PX, int(round(descriptor.avg_width))),
        max(_MIN_DESCRIPTOR_PX, int(round(descriptor.avg_height))),
    )


def _occupancy_soft_hard_ratio(candidate: np.ndarray | None) -> float:
    """Soft-cell count / hard-cell count on a silhouette occupancy grid."""
    if candidate is None or candidate.size == 0:
        return 0.0
    hard = candidate >= HARD_OCCUPANCY
    soft = (candidate > 0) & ~hard
    hard_n = int(hard.sum())
    if hard_n <= 0:
        return 0.0
    return float(int(soft.sum())) / float(hard_n)


def _bbox_area_ratio(
    bbox: tuple[int, int, int, int] | None,
    descriptor: MobDescriptor,
) -> float:
    if bbox is None:
        return 0.0
    _x, _y, w, h = bbox
    if w < 1 or h < 1:
        return 0.0
    desc_area = float(descriptor.avg_width) * float(descriptor.avg_height)
    return (float(w) * float(h)) / desc_area


@dataclass
class DetectionCandidate:
    mob_name: str
    center_x: int
    center_y: int
    bbox: tuple[int, int, int, int]
    final_score: float
    heatmap_score: float
    accepted: bool
    rejection_reason: str
    candidate_scale: float = 1.0

    def to_dict(self) -> dict:
        x, y, w, h = self.bbox
        return {
            "mobName": self.mob_name,
            "center": [self.center_x, self.center_y],
            "centerX": self.center_x,
            "centerY": self.center_y,
            "bbox": [x, y, w, h],
            "finalScore": round(self.final_score, 4),
            "heatmapScore": round(self.heatmap_score, 4),
            "accepted": self.accepted,
            "rejectionReason": self.rejection_reason,
        }


@dataclass
class SilhouetteCheck:
    center_x: int
    center_y: int
    heat_score: float
    passed: bool
    similarity: float
    precision: float = 0.0
    recall: float = 0.0
    candidate_mask: list[float] | None = None
    matched_mask_index: int = 0
    mask_similarities: list[float] | None = None
    extract_bbox: tuple[int, int, int, int] | None = None
    # Cleanup hooks: bloated crop and/or noisy silhouette content.
    noisy_extract: bool = False
    extract_bloated: bool = False
    content_noisy: bool = False
    extract_area_ratio: float = 0.0
    soft_hard_ratio: float = 0.0


@dataclass
class DetectionResult:
    mob_name: str
    descriptor: MobDescriptor
    candidates: list[DetectionCandidate]
    accepted: list[DetectionCandidate]
    elapsed_s: float
    timing: dict[str, float]
    sprite_heatmap: np.ndarray
    silhouette_checks: list[SilhouetteCheck]



@lru_cache(maxsize=8)
def _load_detector_config_cached(path_str: str) -> dict:
    import json

    config = json.loads(Path(path_str).read_text(encoding="utf-8"))
    missing = sorted(REQUIRED_CONFIG_KEYS - set(config))
    if missing:
        raise ValueError(f"missing detector config keys: {', '.join(missing)}")
    return config


def load_detector_config(path: Optional[Path] = None) -> dict:
    config_path = path or (Path(__file__).resolve().parent / "detector_config.json")
    # Return a copy so callers cannot mutate the cached object.
    return dict(_load_detector_config_cached(str(Path(config_path).resolve())))


def clear_detector_config_cache() -> None:
    _load_detector_config_cached.cache_clear()


def configure_opencv_runtime() -> None:
    """Bound native OpenCV parallelism for the live observer runtime.

    Discovery and local tracking intentionally use separate detector sessions,
    so their Python workers can enter OpenCV at the same time. OpenCV's default
    native worker pool multiplies that concurrency and can turn a normal frame
    into a long CPU oversubscription spike. Configure this once at runtime
    startup, rather than as a side effect of constructing an individual
    detector (which would make unrelated detector users and tests order
    dependent).
    """
    cv2.setNumThreads(1)


class MobDetector:
    def __init__(
        self,
        project_root: Path,
        config: Optional[dict] = None,
        *,
        use_sprite_grf: bool = False,
    ):
        self.project_root = project_root
        self.config = load_detector_config() if config is None else config
        self.use_sprite_grf = use_sprite_grf
        # Native OpenCV parallelism is configured once by the live runtime
        # startup path (``configure_opencv_runtime``). Detector construction
        # itself must remain side-effect free because sessions are also used by
        # recognition tools and tests outside the hunt runtime.
        self.heatmap_detector = HeatmapDetector(self.config)
        self._descriptor_cache: dict[str, MobDescriptor] = {}
        # Local tracking state is kept by the single tracking session and
        # contains one stable visual anchor per active Track.
        self._silhouette_ref_cache: dict[
            tuple[int, ...], list[tuple[np.ndarray, np.ndarray]]
        ] = {}
        self._apply_config_values()

    def apply_runtime_config(self, config: dict) -> None:
        self.config = dict(config)
        self.heatmap_detector = HeatmapDetector(self.config)
        self._silhouette_ref_cache.clear()
        self._apply_config_values()

    def _apply_config_values(self) -> None:
        """Refresh scalar detector settings from the active config."""
        self.discovery_heatmap_downscale = int(self.config["discoveryHeatmapDownscale"])
        self.discovery_heatmap_downscale_min_side = int(
            self.config["discoveryHeatmapDownscaleMinSide"]
        )
        self.local_track_search_radius_px = int(
            self.config["localTrackSearchRadiusPx"]
        )
        self.local_track_moving_search_radius_px = int(
            self.config["localTrackMovingSearchRadiusPx"]
        )
        # Keep programmatic/older runtime configs compatible with the new
        # optional tracking tuning knobs; the checked-in detector config still
        # validates and documents the defaults.
        self.local_track_sprite_radius_multiplier = float(
            self.config.get("localTrackSpriteRadiusMultiplier", 1.5)
        )
        self.local_track_max_search_radius_px = int(
            self.config.get("localTrackMaxSearchRadiusPx", 360)
        )
        # GRF mode (modified sprite.grf) relaxes the silhouette gate and lets
        # static-descriptor local tracking skip the native verify. Defaults
        # keep programmatic/older configs working.
        self.grf_min_silhouette_recall = float(
            self.config.get("grfMinSilhouetteRecall", 0.32)
        )
        self.grf_min_silhouette_precision = float(
            self.config.get("grfMinSilhouettePrecision", 0.55)
        )
        self.grf_local_track_skip_native_gate = bool(
            self.config.get("grfLocalTrackSkipNativeGate", True)
        )
        # GRF mode widens the descriptor aspect band: a static red sprite's
        # palette CC is often clipped (head/feet shade outside the match radius),
        # which shifts the extract aspect beyond the build-time tight band.
        self.grf_aspect_band_scale = float(
            self.config.get("grfAspectBandScale", 1.3)
        )

    def descriptor_path(self, mob_name: str) -> Path:
        stem = mob_name.lower()
        filename = (
            "modified_sprite_descriptor.json"
            if self.use_sprite_grf
            else "descriptor.json"
        )
        return (
            self.project_root
            / "assets"
            / "generated_descriptors"
            / stem
            / filename
        )

    def ensure_descriptor(self, mob_name: str) -> MobDescriptor:
        mob_name = mob_name.lower()
        if mob_name in self._descriptor_cache:
            return self._descriptor_cache[mob_name]
        path = self.descriptor_path(mob_name)
        if not path.exists():
            raise FileNotFoundError(f"descriptor not found for mob '{mob_name}': {path}")
        descriptor = MobDescriptor.load(path)
        if descriptor.version < DESCRIPTOR_VERSION:
            raise RuntimeError(
                f"descriptor for mob '{mob_name}' is version {descriptor.version}; "
                f"rebuild descriptor version {DESCRIPTOR_VERSION} before detection"
            )
        self._descriptor_cache[mob_name] = descriptor
        return descriptor

    # ------------------------------------------------------------------
    #  Discovery: heatmap → blobs → geometry → color structure → silhouette
    # ------------------------------------------------------------------

    def _discovery_heatmap_downscale(self, frame_bgr: np.ndarray) -> int:
        """Choose discovery scale from frame size and rendering mode only.

        GRF mode deliberately uses a fixed 4× work scale for every mob. The
        detector must not silently undo that choice based on descriptor size:
        doing so makes performance and memory use depend on which mob is being
        hunted, and was the source of the Noxious native-resolution fallback.
        """
        if self.use_sprite_grf:
            return _SPRITE_GRF_HEATMAP_DOWNSCALE
        if (
            self.discovery_heatmap_downscale > 1
            and min(frame_bgr.shape[:2]) >= self.discovery_heatmap_downscale_min_side
        ):
            return self.discovery_heatmap_downscale
        return 1

    def detect(
        self,
        frame_bgr: np.ndarray,
        mob_name: str,
    ) -> DetectionResult:
        """Heatmap discovery with silhouette check.

        Order: heatmap → blobs → geometry → color structure → silhouette.
        All blobs go through every gate — dedup against existing tracks is
        handled by TrackReconciler after detection.

        When ``use_sprite_grf`` is True, uses a deterministic 4× heatmap
        downscale for every mob; no descriptor-size or mob-specific fallback is
        applied.
        """
        start = time.perf_counter()
        descriptor = self.ensure_descriptor(mob_name)

        # --- heatmap --------------------------------------------------
        heatmap_start = time.perf_counter()
        downscale = self._discovery_heatmap_downscale(frame_bgr)
        # Static modified sprites use the cheap palette-only discovery path.
        # Compute this before building the heatmap so the fast mode is also
        # applied to the first post-teleport scan.
        static_sprite_fast_path = self.use_sprite_grf and self.descriptor_is_static(
            descriptor,
        )
        if (
            not self.use_sprite_grf
            and downscale > 1
            and min(descriptor.avg_width, descriptor.avg_height) / downscale
            < _DOWNSCALE_MIN_WORK_RESOLUTION_PX
        ):
            downscale = 1

        build_heatmap = self.heatmap_detector.build_sprite_heatmap
        # Keep older detector doubles/extensions usable without catching an
        # internal TypeError from the actual heatmap implementation. Signature
        # dispatch is resolved before the call, so production failures remain
        # visible and static GRF mode cannot silently fall back to the slow path.
        try:
            heatmap_parameters = inspect.signature(build_heatmap).parameters
            supports_fast_static = (
                "fast_static" in heatmap_parameters
                or any(
                    parameter.kind is inspect.Parameter.VAR_KEYWORD
                    for parameter in heatmap_parameters.values()
                )
            )
        except (TypeError, ValueError):
            # Some extension callables do not expose a signature; the current
            # built-in implementation supports the keyword, so use it there.
            supports_fast_static = True
        if supports_fast_static:
            sprite_heatmap = build_heatmap(
                frame_bgr,
                descriptor,
                downscale=downscale,
                fast_static=static_sprite_fast_path,
            )
        else:
            sprite_heatmap = build_heatmap(
                frame_bgr,
                descriptor,
                downscale=downscale,
            )
        heatmap_end = time.perf_counter()

        # --- blobs ----------------------------------------------------
        blobs = self.heatmap_detector.top_centers(sprite_heatmap, descriptor)
        blobs_end = time.perf_counter()

        heatmap_peak = float(sprite_heatmap.max()) if sprite_heatmap.size else 0.0
        peak_rel = float(self.config["peakRelativeThreshold"])
        small_rel_heat = _SMALL_HEAT_RELATIVE_PEAK_MULT * peak_rel
        # The silhouette gate needs the unweighted palette mask, not the
        # weighted discovery heatmap. Build that mask once per frame and slice
        # it for every candidate. Recomputing the same palette-distance matrix
        # inside every gate made a frame with several plausible blobs spend
        # seconds in the gate, even though only one mob was ultimately accepted.
        # This is especially visible when the sitting character changes the
        # post-teleport frame and creates several Anubis-colored blobs.
        # A full-frame palette map pays off only when several candidates will
        # reuse it. Keep the one-candidate/common path local; this avoids adding
        # a large frame-wide allocation to normal scans while bounding the
        # repeated per-candidate work on a noisy post-transition frame. Static
        # GRF mode deliberately keeps this None: each local silhouette window
        # is much smaller than the 1024x1024 frame and uses the one exact sprite
        # palette directly.
        reuse_palette_heatmap = len(blobs) >= 2 and not static_sprite_fast_path
        palette_heatmap_started = time.perf_counter()
        palette_heatmap_full = (
            sprite_palette_heatmap(
                frame_bgr,
                descriptor.match_palette_bgr,
                float(descriptor.max_sprite_palette_distance),
            )
            if reuse_palette_heatmap
            else None
        )
        palette_heatmap_elapsed = time.perf_counter() - palette_heatmap_started

        # --- gates → silhouette (known tracks skip pre-gates) ----------
        candidates: list[DetectionCandidate] = []
        silhouette_checks: list[SilhouetteCheck] = []
        # Keep candidate-level timing so a pathological live frame can be
        # diagnosed as blob explosion versus one malformed silhouette crop.
        gate_elapsed_s = 0.0
        max_gate_elapsed_s = 0.0
        max_gate_bbox: tuple[int, int, int, int] = (0, 0, 0, 0)

        for cx, cy, heat_score, comp_bbox in blobs:
            bx, by, bw, bh = comp_bbox
            bbox = (bx, by, bw, bh)
            # All blobs must clear geometry + color structure pre-gates.
            # Dedup against existing tracks is handled post-detection by
            # TrackReconciler.match_and_absent().
            if not self._passes_discovery_geometry_gate(comp_bbox, descriptor):
                silhouette_checks.append(SilhouetteCheck(
                    center_x=cx,
                    center_y=cy,
                    heat_score=heat_score,
                    passed=False,
                    similarity=0.0,
                ))
                continue

            # Tiny heat CCs: require relative heat vs frame peak (config-derived).
            if self._is_small_heat_cc(comp_bbox, descriptor):
                if heatmap_peak <= 0.0 or (float(heat_score) / heatmap_peak) < small_rel_heat:
                    silhouette_checks.append(SilhouetteCheck(
                        center_x=cx,
                        center_y=cy,
                        heat_score=heat_score,
                        passed=False,
                        similarity=0.0,
                    ))
                    continue

            if (
                not static_sprite_fast_path
                and not self._passes_color_structure_gate(
                    frame_bgr, descriptor, comp_bbox,
                )
            ):
                silhouette_checks.append(SilhouetteCheck(
                    center_x=cx,
                    center_y=cy,
                    heat_score=heat_score,
                    passed=False,
                    similarity=0.0,
                ))
                continue


            gate_started = time.perf_counter()
            (
                passed,
                similarity,
                candidate,
                matched_idx,
                scores,
                extract_bbox,
                precision,
                recall,
                bridged_extract_area_ratio,
            ) = self._evaluate_silhouette_gate(
                frame_bgr,
                descriptor,
                bbox,
                comp_bbox=comp_bbox,
                palette_heatmap_full=palette_heatmap_full,
            )
            gate_elapsed = time.perf_counter() - gate_started
            gate_elapsed_s += gate_elapsed
            if gate_elapsed > max_gate_elapsed_s:
                max_gate_elapsed_s = gate_elapsed
                max_gate_bbox = comp_bbox
            # The generic extract body confirmation is useful for animated
            # sprites and gray-world impostors. In static GRF mode the local
            # palette-backed one-reference silhouette gate is the confirmation;
            # repeating a second full palette-group analysis only adds latency.
            if passed and not static_sprite_fast_path:
                if not self._passes_extract_body_gate(
                    frame_bgr, descriptor, extract_bbox,
                ):
                    passed = False
            candidate_mask = (
                candidate.reshape(-1).tolist() if candidate is not None else None
            )
            (
                noisy_extract,
                extract_bloated,
                content_noisy,
                extract_area_ratio,
                soft_hard_ratio,
            ) = self._noisy_extraction_signal(
                extract_bbox,
                descriptor,
                candidate,
                extract_area_ratio=bridged_extract_area_ratio,
            )
            # Drawn/accept box = heat CC bbox (a35ef47 tight blob box).
            silhouette_checks.append(SilhouetteCheck(
                center_x=cx,
                center_y=cy,
                heat_score=heat_score,
                passed=passed,
                similarity=similarity,
                precision=precision,
                recall=recall,
                candidate_mask=candidate_mask,
                matched_mask_index=matched_idx,
                mask_similarities=scores,
                extract_bbox=extract_bbox,
                noisy_extract=noisy_extract,
                extract_bloated=extract_bloated,
                content_noisy=content_noisy,
                extract_area_ratio=extract_area_ratio,
                soft_hard_ratio=soft_hard_ratio,
            ))

            if passed and extract_bbox is not None:
                # Use silhouette extract bbox center (refined by palette CC)
                # instead of the raw heatmap-blob center. The heatmap center
                # can be off for asymmetrical mobs or clustered blobs;
                # the extract bbox is the silhouette-matched palette region.
                ex, ey, ew, eh = extract_bbox
                refined_cx = ex + ew // 2
                refined_cy = ey + eh // 2
                candidates.append(DetectionCandidate(
                    mob_name=descriptor.mob_name,
                    center_x=refined_cx,
                    center_y=refined_cy,
                    bbox=bbox,
                    final_score=heat_score,
                    heatmap_score=heat_score,
                    accepted=True,
                    rejection_reason="",
                ))


        gate_end = time.perf_counter()
        max_candidates = int(self.config["maxCandidates"])
        accepted = self._finalize_accepted(candidates)[:max_candidates]

        elapsed = time.perf_counter() - start
        timing = {
            "descriptor": heatmap_start - start,
            "spriteHeatmap": heatmap_end - heatmap_start,
            "blobCenters": blobs_end - heatmap_end,
            "silhouettePaletteHeatmap": palette_heatmap_elapsed,
            # Candidate gate time excludes the optional shared palette-map
            # build above; ``silhouetteGate`` is therefore directly comparable
            # with ``gateTotal`` and no longer hides the precompute cost.
            "silhouetteGate": gate_end - (
                palette_heatmap_started + palette_heatmap_elapsed
            ),
            "total": elapsed,
            "blobCount": float(len(blobs)),
            "silhouetteCheckCount": float(len(silhouette_checks)),
            "gateTotal": gate_elapsed_s,
            "maxGate": max_gate_elapsed_s,
            "maxGateWidth": float(max_gate_bbox[2]),
            "maxGateHeight": float(max_gate_bbox[3]),
        }

        return DetectionResult(
            mob_name=mob_name.lower(),
            descriptor=descriptor,
            candidates=accepted,
            accepted=accepted,
            elapsed_s=elapsed,
            timing=timing,
            sprite_heatmap=sprite_heatmap,
            silhouette_checks=silhouette_checks,
        )

    # ------------------------------------------------------------------
    #  Geometry pre-gate + silhouette gate
    # ------------------------------------------------------------------

    def _heat_area_ratio(
        self,
        comp_bbox: tuple[int, int, int, int],
        descriptor: MobDescriptor,
    ) -> float:
        """Heat-CC area / descriptor sprite area."""
        _x, _y, bw, bh = comp_bbox
        desc_area = max(float(descriptor.avg_width) * float(descriptor.avg_height), 1.0)
        return (float(max(int(bw), 0)) * float(max(int(bh), 0))) / desc_area

    def _small_heat_area_cutoff(self, descriptor: MobDescriptor) -> float:
        """Area ratio below which heat-CC body density is treated as unreliable.

        ``2 × min_area_ratio`` = ``2/5`` of mean stable silhouette fraction —
        derived from the same silhouette occupancy that sets the geometry floor.
        """
        return (
            self._descriptor_min_area_ratio(descriptor)
            * _BODY_STRONG_SMALL_HEAT_AREA_MIN_AREA_MULT
        )

    def _is_small_heat_cc(
        self,
        comp_bbox: tuple[int, int, int, int],
        descriptor: MobDescriptor,
    ) -> bool:
        return self._heat_area_ratio(comp_bbox, descriptor) < self._small_heat_area_cutoff(
            descriptor,
        )

    def _passes_discovery_geometry_gate(
        self,
        comp_bbox: tuple[int, int, int, int],
        descriptor: MobDescriptor,
    ) -> bool:
        """Reject heat CCs whose size/aspect cannot plausibly match the mob.

        ``min_area_ratio = sil_frac / _GEOMETRY_AREA_SIL_FRAC_DIVISOR`` uses the
        descriptor's stable silhouette occupancy as a lower bound on heat-CC area
        vs sprite area. ``_GEOMETRY_AREA_MAX_RATIO`` caps terrain mega-blobs.
        Aspect uses the per-mob band ``descriptor.min_aspect_ratio`` /
        ``descriptor.max_aspect_ratio`` (sprite tight-bboxes at build time).
        """
        _x, _y, hw, hh = comp_bbox
        return self._passes_size_aspect_vs_descriptor(
            int(hw), int(hh), descriptor, require_min_area=True,
        )

    def _passes_color_structure_gate(
        self,
        frame_bgr: np.ndarray,
        descriptor: MobDescriptor,
        comp_bbox: tuple[int, int, int, int],
    ) -> bool:
        """Reject heat CCs that lack this mob's color structure / palette.

        Fail-closed before silhouette:
        - enough required groups present (diversity presence)
        - non-trivial second-group share (rejects mono-family, e.g. Poring)
        - enough crop pixels match required-group colors (coverage)
        - enough crop pixels strongly match mass body clusters
          (rejects obviously foreign palettes)

        Group presence / second-share / coverage always use the heat-CC crop.
        ``body_strong`` is full-resolution BGR only (never the downscaled
        diversity body map).  Normal heat CCs use the heat crop.  Small heat
        CCs (area < ``2 × descriptor min_area_ratio``) re-measure body on a
        descriptor-sized window so a few matching pixels cannot clear the
        per-mob floor (0WildRose_Gray).

        Skips when the descriptor has no required groups.
        """
        required_groups = descriptor.match_palette_required_groups
        if not required_groups:
            return True
        min_groups = int(self.config["minRequiredPaletteGroups"])
        min_second = float(self.config["minSecondPaletteGroupShare"])
        min_body_strong = float(descriptor.min_body_cluster_strong)
        min_coverage = float(descriptor.min_required_palette_coverage)
        if (
            min_groups <= 0
            and min_second <= 0.0
            and min_coverage <= 0.0
            and min_body_strong <= 0.0
        ):
            return True
        bx, by, bw, bh = comp_bbox
        fh, fw = frame_bgr.shape[:2]
        x0 = max(0, int(bx))
        y0 = max(0, int(by))
        x1 = min(fw, x0 + max(0, int(bw)))
        y1 = min(fh, y0 + max(0, int(bh)))
        if x1 <= x0 or y1 <= y0:
            return False
        heat_crop = frame_bgr[y0:y1, x0:x1]
        present, second_share, match_coverage, body_strong = required_groups_structure(
            heat_crop,
            descriptor,
            float(descriptor.max_sprite_palette_distance),
            downscale=1,
        )
        if min_groups > 0 and present < min_groups:
            return False
        if min_second > 0.0 and second_share < min_second:
            return False
        if min_coverage > 0.0 and match_coverage < min_coverage:
            return False

        if min_body_strong > 0.0:
            if self._is_small_heat_cc(comp_bbox, descriptor):
                # Tiny heat CC: body density on heat crop is inflated — use
                # descriptor-sized full-res crop (same scale as build floor).
                desc_w = max(1, int(round(descriptor.avg_width)))
                desc_h = max(1, int(round(descriptor.avg_height)))
                cc_cx = bx + bw // 2
                cc_cy = by + bh // 2
                bx0 = max(0, cc_cx - desc_w // 2)
                by0 = max(0, cc_cy - desc_h // 2)
                bx1 = min(fw, bx0 + desc_w)
                by1 = min(fh, by0 + desc_h)
                bx0 = max(0, bx1 - desc_w)
                by0 = max(0, by1 - desc_h)
                if bx1 <= bx0 or by1 <= by0:
                    return False
                body_crop = frame_bgr[by0:by1, bx0:bx1]
                _p, _s, _c, body_strong = required_groups_structure(
                    body_crop,
                    descriptor,
                    float(descriptor.max_sprite_palette_distance),
                    downscale=1,
                )
            if body_strong < min_body_strong:
                return False

        return True

    def _passes_extract_body_gate(
        self,
        frame_bgr: np.ndarray,
        descriptor: MobDescriptor,
        extract_bbox: tuple[int, int, int, int] | None,
    ) -> bool:
        """Post-silhouette body confirm on the final extract crop.

        Heat-CC color can pass on a small/tinted blob that later silhouette-matches
        a wrong shape. The extract is descriptor-scaled and bridged — re-check
        mass body density there at a soft fraction of the build-time floor so
        patchy true mobs (Creamy) still clear while wrong-fill impostors fail.

        Skips when the descriptor has no body floor or no extract.
        """
        min_body_strong = float(descriptor.min_body_cluster_strong)
        if min_body_strong <= 0.0:
            return True
        if extract_bbox is None:
            return False
        x, y, w, h = extract_bbox
        fh, fw = frame_bgr.shape[:2]
        x0 = max(0, int(x))
        y0 = max(0, int(y))
        x1 = min(fw, x0 + max(0, int(w)))
        y1 = min(fh, y0 + max(0, int(h)))
        if x1 <= x0 or y1 <= y0:
            return False
        crop = frame_bgr[y0:y1, x0:x1]
        if crop.size == 0:
            return False
        # Body only — groups/coverage already cleared on the heat CC; extract can
        # be tighter and drop intermittent second-group share.
        _p, _s, _c, body_strong = required_groups_structure(
            crop,
            descriptor,
            float(descriptor.max_sprite_palette_distance),
            downscale=1,
        )
        floor = min_body_strong * _EXTRACT_BODY_STRONG_FLOOR_FRAC
        return body_strong >= floor

    def _descriptor_min_area_ratio(self, descriptor: MobDescriptor) -> float:
        """Mean stable silhouette occupancy across all facings, cached per descriptor.


        Cached on the descriptor object to avoid recomputing stable_bits per
        blob.  ``_GEOMETRY_AREA_SIL_FRAC_DIVISOR = 5.0`` already provides an
        80 % margin below the representative sprite footprint, so per-mask
        minimums are unnecessary leniency.
        """
        cached = getattr(descriptor, "_min_area_ratio", None)
        if cached is not None:
            return float(cached)
        stable_bits: list[bool] = []
        for mask in descriptor.silhouette_masks:
            if mask.stable_mask:
                stable_bits.extend(mask.stable_mask)
        if not stable_bits:
            descriptor._min_area_ratio = 1.0
            return 1.0
        sil_frac = float(np.mean(np.asarray(stable_bits, dtype=np.float32)))
        result = sil_frac / _GEOMETRY_AREA_SIL_FRAC_DIVISOR
        descriptor._min_area_ratio = result
        return result

    def _effective_aspect_band(self, descriptor: MobDescriptor) -> tuple[float, float]:
        """Aspect band for the active mode (GRF widens the build-time band).

        Modified sprites are static and palette-distinctive, but their palette CC
        extract is frequently clipped (a head/feet shade outside the match radius
        shortens one axis), which pushes the extract aspect past the build-time
        tight band — Anubis' clipped extract measured 1.17 vs a 1.03 max. GRF mode
        therefore scales the band outward; the red palette keeps wrong-size blobs
        from passing anyway.
        """
        if self.use_sprite_grf:
            scale = max(1.0, self.grf_aspect_band_scale)
            return (
                descriptor.min_aspect_ratio / scale,
                descriptor.max_aspect_ratio * scale,
            )
        return descriptor.min_aspect_ratio, descriptor.max_aspect_ratio

    def _passes_size_aspect_vs_descriptor(
        self,
        width: int,
        height: int,
        descriptor: MobDescriptor,
        *,
        require_min_area: bool,
        enforce_max_area: bool = True,
        grf_wide_aspect: bool = False,
    ) -> bool:
        """Descriptor-relative area + aspect band shared by heat and extract.

        Aspect is normalized by the mean sprite aspect (``(w/h) / (desc_w/desc_h)``)
        then compared to ``min_aspect_ratio`` / ``max_aspect_ratio``. Those bounds
        are measured from sprite tight-bboxes at build time (with margin) and
        floored by ``MIN_ASPECT_FLOOR`` so the band is expressed in the same
        normalized units the gate uses.

        ``grf_wide_aspect`` opts the *extract* check into GRF mode's widened band
        (``_effective_aspect_band``) — a clipped palette CC of a static red sprite
        often measures past the build-time band. The heat geometry pre-gate keeps
        the tight band so terrain mega-blobs still fail early.
        """
        if width < 1 or height < 1:
            return False
        desc_w = float(descriptor.avg_width)
        desc_h = float(descriptor.avg_height)
        if desc_w <= 0.0 or desc_h <= 0.0:
            return False
        desc_area = desc_w * desc_h
        desc_aspect = desc_w / desc_h
        area_ratio = (float(width) * float(height)) / desc_area
        aspect_ratio = (float(width) / float(height)) / desc_aspect
        if grf_wide_aspect and self.use_sprite_grf:
            min_aspect, max_aspect = self._effective_aspect_band(descriptor)
        else:
            min_aspect = descriptor.min_aspect_ratio
            max_aspect = descriptor.max_aspect_ratio
        if require_min_area and area_ratio < self._descriptor_min_area_ratio(descriptor):
            return False
        if enforce_max_area and area_ratio > _GEOMETRY_AREA_MAX_RATIO:
            return False
        if aspect_ratio < min_aspect or aspect_ratio > max_aspect:
            return False
        return True



    def _noisy_extraction_signal(
        self,
        extract_bbox: tuple[int, int, int, int] | None,
        descriptor: MobDescriptor,
        candidate: np.ndarray | None,
        *,
        extract_area_ratio: float | None = None,
    ) -> tuple[bool, bool, bool, float, float]:
        """Detect bloated crops and/or noisy silhouette *content*.

        ``extract_bloated``: bridged palette-CC area >= ``_EXTRACT_BLOAT_AREA_RATIO``
        × descriptor sprite area (terrain merge). Prefer the pre-shrink ratio from
        the silhouette gate so BLOAT still flags after a successful descriptor
        re-crop. Large but clean crops (e.g. some Noxious) can be bloated without
        being content-noisy.

        ``content_noisy``: soft occupancy mass >= ``_CONTENT_NOISE_SOFT_HARD_RATIO``
        × hard mass on the final candidate grid. A compact sprite has
        soft ≈ O(perimeter) ≈ O(sqrt(hard)), so soft/hard ≪ 1. soft/hard >= 2
        means the soft field dominates the hard body (terrain bleed / confetti),
        independent of bbox size.

        ``noisy_extract`` = bloated OR content_noisy. Cleanup hook only — no reject.
        Returns
        ``(noisy_extract, extract_bloated, content_noisy, extract_area_ratio, soft_hard_ratio)``.
        """
        if extract_area_ratio is None:
            extract_area_ratio = _bbox_area_ratio(extract_bbox, descriptor)
        extract_bloated = extract_area_ratio >= _EXTRACT_BLOAT_AREA_RATIO

        soft_hard_ratio = _occupancy_soft_hard_ratio(candidate)
        # Soft mass at least 2× hard: cannot be a compact 1-cell halo.
        content_noisy = soft_hard_ratio >= _CONTENT_NOISE_SOFT_HARD_RATIO

        noisy_extract = extract_bloated or content_noisy
        return (
            noisy_extract,
            extract_bloated,
            content_noisy,
            float(extract_area_ratio),
            float(soft_hard_ratio),
        )

    def _descriptor_silhouette_references(
        self,
        masks: list,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        if not masks:
            return []
        cache_key = tuple(id(mask) for mask in masks)
        cached = self._silhouette_ref_cache.get(cache_key)
        if cached is not None:
            return cached
        refs: list[tuple[np.ndarray, np.ndarray]] = []
        seen: list[tuple[np.ndarray, np.ndarray]] = []
        for mask in masks:
            if not mask.stable_mask or not any(mask.stable_mask):
                continue
            avg = np.array(mask.avg_mask, dtype=np.float32).reshape(mask.height, mask.width)
            stable = np.array(mask.stable_mask, dtype=bool).reshape(mask.height, mask.width)
            # Static modified sprites carry the same canonical frame under every
            # facing; dedup identical refs so the gate literally scores against
            # the one frame (matching how the sprite renders) and skips the
            # duplicated comparison. Rounding absorbs float noise across rebuilds.
            duplicate = False
            for prev_avg, prev_stable in seen:
                if (
                    avg.shape == prev_avg.shape
                    and np.array_equal(np.round(avg, 3), np.round(prev_avg, 3))
                    and np.array_equal(stable, prev_stable)
                ):
                    duplicate = True
                    break
            if duplicate:
                continue
            seen.append((avg, stable))
            refs.append((avg, stable))
        self._silhouette_ref_cache[cache_key] = refs
        return refs

    def silhouette_gate_thresholds(self) -> tuple[float, float]:
        """Return (min_recall, min_precision) for the active rendering mode.

        Modified sprite.grf assets are a single deterministic static frame with
        a distinctive red palette — far easier to recognize than the animated
        originals. GRF mode therefore relaxes the silhouette gate so a partially
        occluded or heavily deformed extract still passes, reducing discovery
        misses (and the teleport-away risk) for static modified sprites.
        """
        if self.use_sprite_grf:
            return (
                self.grf_min_silhouette_recall,
                self.grf_min_silhouette_precision,
            )
        return (
            float(self.config["minSilhouetteRecall"]),
            float(self.config["minSilhouettePrecision"]),
        )

    def descriptor_is_static(self, descriptor: MobDescriptor) -> bool:
        """True when every silhouette ref is the same frame (modified static sprite).

        Modified sprites are generated as one canonical frame, so the descriptor
        carries a single unique pose across all facings. A static descriptor has
        a deterministic appearance: the animation-diversity gate adds nothing and
        local tracking can follow it without the native-resolution verify.
        """
        cached = getattr(descriptor, "_static_descriptor", None)
        if cached is not None:
            return bool(cached)
        masks = descriptor.silhouette_masks
        if not masks:
            descriptor._static_descriptor = False
            return False
        first = masks[0]
        first_shape = (first.width, first.height)
        first_avg = tuple(round(float(v), 3) for v in first.avg_mask)
        first_stable = tuple(bool(v) for v in first.stable_mask)
        for mask in masks[1:]:
            if (
                (mask.width, mask.height) != first_shape
                or tuple(round(float(v), 3) for v in mask.avg_mask) != first_avg
                or tuple(bool(v) for v in mask.stable_mask) != first_stable
            ):
                descriptor._static_descriptor = False
                return False
        descriptor._static_descriptor = True
        return True



    def _evaluate_silhouette_gate(
        self,
        frame_bgr: np.ndarray,
        descriptor: MobDescriptor,
        bbox: tuple[int, int, int, int],
        *,
        comp_bbox: tuple[int, int, int, int] | None = None,
        masks: list | None = None,
        palette_heatmap_full: np.ndarray | None = None,
    ) -> tuple[
        bool,
        float,
        np.ndarray | None,
        int,
        list[float],
        tuple[int, int, int, int] | None,
        float,
        float,
        float,
    ]:
        """Silhouette gate matching a35ef47 crop style.

        Search around the heat CC (not sprite-inflated). Take the overlapping
        palette CC, bridge nearby same-row palette fragments horizontally
        (descriptor-scaled), crop tightly, then resize to descriptor size.
        Returns (passed, jaccard, candidate, matched_idx, scores, extract_bbox,
        precision, recall, bridged_extract_area_ratio).

        *masks* defaults to ``descriptor.silhouette_masks``.
        """
        fail = (False, 0.0, None, 0, [], None, 0.0, 0.0, 0.0)
        gate_masks = (
            list(masks)
            if masks is not None
            else list(descriptor.silhouette_masks)
        )
        refs = self._descriptor_silhouette_references(gate_masks)
        if not refs or not descriptor.match_palette_bgr:
            return fail
        gate_mask = gate_masks[0]
        desc_w, desc_h = _descriptor_sprite_size_px(descriptor)

        search = self._silhouette_search_window(frame_bgr, bbox, comp_bbox, desc_w, desc_h)
        if search is None:
            return fail
        search_region, search_x, search_y, ref_w, ref_h, local_bbox_left, local_bbox_top = search

        if palette_heatmap_full is not None:
            region_h, region_w = search_region.shape[:2]
            palette_heat = palette_heatmap_full[
                search_y : search_y + region_h,
                search_x : search_x + region_w,
            ]
            if palette_heat.shape != (region_h, region_w):
                palette_heat = sprite_palette_heatmap(
                    search_region,
                    descriptor.match_palette_bgr,
                    float(descriptor.max_sprite_palette_distance),
                )
        else:
            palette_heat = sprite_palette_heatmap(
                search_region, descriptor.match_palette_bgr,
                float(descriptor.max_sprite_palette_distance),
            )
        binary_raw = (palette_heat >= float(self.config["minSpritePaletteMatch"])).astype(np.uint8)
        if not np.any(binary_raw):
            return fail

        binary = cv2.dilate(binary_raw, _MORPH_ELLIPSE_3, iterations=1)
        best_mask = self._best_overlapping_palette_component(
            binary, local_bbox_left, local_bbox_top, ref_w, ref_h,
        )
        if best_mask is None:
            return fail

        occupancy = self._horizontal_bridge_occupancy(
            binary_raw, best_mask, desc_w, gate_mask.width,
        )
        ys, xs = np.where(occupancy)
        if len(xs) == 0:
            return fail
        comp_left = int(xs.min())
        comp_right = int(xs.max()) + 1
        comp_top = int(ys.min())
        comp_bottom = int(ys.max()) + 1
        comp_w = comp_right - comp_left
        comp_h = comp_bottom - comp_top
        if comp_w < _MIN_EXTRACT_COMPONENT_PX or comp_h < _MIN_EXTRACT_COMPONENT_PX:
            return fail

        desc_area = float(desc_w) * float(desc_h)
        extract_area_ratio = (float(comp_w) * float(comp_h)) / desc_area

        # True palette extract (before any desc-sized re-frame): same aspect band
        # as heat geometry. Min area applies; max is not enforced here so a
        # terrain-merged CC can still shrink for rasterization after aspect OK.
        if not self._passes_size_aspect_vs_descriptor(
            comp_w,
            comp_h,
            descriptor,
            require_min_area=True,
            enforce_max_area=False,
            grf_wide_aspect=True,
        ):
            extract_bbox = (
                search_x + comp_left,
                search_y + comp_top,
                comp_w,
                comp_h,
            )
            return False, 0.0, None, 0, [], extract_bbox, 0.0, 0.0, extract_area_ratio

        extract_bloated = extract_area_ratio >= _EXTRACT_BLOAT_AREA_RATIO
        if extract_bloated:
            # Bloated crop (terrain-merged CC): re-frame to descriptor-sized
            # window on the body centroid so silhouette sees a sprite-scale extract.
            mob_region, comp_mask, extract_bbox = self._shrink_bloated_extract_to_descriptor(
                search_region,
                binary_raw,
                best_mask,
                desc_w,
                desc_h,
                search_x,
                search_y,
            )
        else:
            extract_bbox = (
                search_x + comp_left,
                search_y + comp_top,
                comp_w,
                comp_h,
            )
            comp_mask = occupancy[comp_top:comp_bottom, comp_left:comp_right]
            mob_region = search_region[comp_top:comp_bottom, comp_left:comp_right]

        if mob_region.size == 0 or not np.any(comp_mask):
            return False, 0.0, None, 0, [], extract_bbox, 0.0, 0.0, extract_area_ratio

        if mob_region.shape[1] != desc_w or mob_region.shape[0] != desc_h:
            mob_region = cv2.resize(mob_region, (desc_w, desc_h), interpolation=cv2.INTER_LINEAR)
            comp_mask = cv2.resize(
                comp_mask.astype(np.uint8),
                (desc_w, desc_h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

        silhouette_distance = (
            float(descriptor.max_silhouette_palette_distance)
            * float(self.config["silhouettePaletteDistanceScale"])
        )
        palette = np.asarray(descriptor.match_palette_bgr, dtype=np.float32)
        candidate = candidate_silhouette(
            mob_region,
            palette,
            silhouette_distance,
            gate_mask.width, gate_mask.height,
            occupancy_mask=comp_mask,
        )
        candidate = self._maybe_deform_noisy_candidate(
            candidate, refs, mob_region, descriptor, palette, silhouette_distance, gate_mask,
        )

        similarity, matched_idx, scores, precision, recall = best_silhouette_match(
            candidate, refs,
        )
        hard_n = int((candidate >= HARD_OCCUPANCY).sum()) if candidate is not None else 0
        grid_n = int(gate_mask.width) * int(gate_mask.height)
        solid_fill = (
            grid_n > 0 and (float(hard_n) / float(grid_n)) >= _SOLID_FILL_HARD_FRACTION
        )
        min_recall, min_precision = self.silhouette_gate_thresholds()
        dual_ok = (
            recall >= min_recall
            and precision >= min_precision
        )
        # Content veto: solid palette fill of the gate grid (color smear in a
        # desc-sized window). Bloated CCs may still shrink after pre-shrink
        # aspect passes; soft/hard noise still uses deform for patchy mobs and
        # remains on SilhouetteCheck via _noisy_extraction_signal.
        passed = bool(dual_ok and not solid_fill)
        return (
            passed,
            float(similarity),
            candidate,
            matched_idx,
            scores,
            extract_bbox,
            float(precision),
            float(recall),
            float(extract_area_ratio),
        )

    def _silhouette_search_window(
        self,
        frame_bgr: np.ndarray,
        bbox: tuple[int, int, int, int],
        comp_bbox: tuple[int, int, int, int] | None,
        desc_w: int,
        desc_h: int,
    ) -> tuple[np.ndarray, int, int, int, int, int, int] | None:
        """Search around heat CC (or bbox): at least CC size and descriptor size."""
        x, y, w, h = bbox
        fh, fw = frame_bgr.shape[:2]
        if comp_bbox is not None:
            hx, hy, hw, hh = comp_bbox
            ref_cx = hx + hw // 2
            ref_cy = hy + hh // 2
            ref_w = max(hw, desc_w)
            ref_h = max(hh, desc_h)
        else:
            ref_cx = x + w // 2
            ref_cy = y + h // 2
            ref_w = max(w, desc_w)
            ref_h = max(h, desc_h)

        search_x = max(0, ref_cx - ref_w)
        search_y = max(0, ref_cy - ref_h)
        search_w = min(fw - search_x, ref_w * 2)
        search_h = min(fh - search_y, ref_h * 2)
        search_region = frame_bgr[search_y : search_y + search_h, search_x : search_x + search_w]
        if search_region.size == 0:
            return None
        local_bbox_left = ref_cx - ref_w // 2 - search_x
        local_bbox_top = ref_cy - ref_h // 2 - search_y
        return (
            search_region, search_x, search_y, ref_w, ref_h,
            local_bbox_left, local_bbox_top,
        )

    def _best_overlapping_palette_component(
        self,
        binary: np.ndarray,
        local_bbox_left: int,
        local_bbox_top: int,
        ref_w: int,
        ref_h: int,
    ) -> np.ndarray | None:
        """Dilated palette CC with largest overlap against the heat reference box."""
        _nl, labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if _nl <= 1:
            return None

        best_overlap = 0
        best_label = 0
        for lbl in range(1, _nl):
            cl = int(stats[lbl, cv2.CC_STAT_LEFT])
            ct = int(stats[lbl, cv2.CC_STAT_TOP])
            cr = cl + int(stats[lbl, cv2.CC_STAT_WIDTH])
            cb = ct + int(stats[lbl, cv2.CC_STAT_HEIGHT])
            ol = max(cl, local_bbox_left)
            ot = max(ct, local_bbox_top)
            o_r = min(cr, local_bbox_left + ref_w)
            o_b = min(cb, local_bbox_top + ref_h)
            if ol < o_r and ot < o_b:
                oa = (o_r - ol) * (o_b - ot)
                if oa > best_overlap:
                    best_overlap = oa
                    best_label = lbl
        if best_label == 0:
            return None
        best_mask = labels == best_label
        if not np.any(best_mask):
            return None
        return best_mask

    def _horizontal_bridge_occupancy(
        self,
        binary_raw: np.ndarray,
        best_mask: np.ndarray,
        desc_w: int,
        gate_width: int,
    ) -> np.ndarray:
        """Bridge same-row palette fragments the dilate-CC missed (patchy wings).

        Gap budget is N silhouette-grid cells mapped into sprite pixels:
        bridge_px ≈ cells * desc_w / grid_w. Geodesic grow stays inside the
        closed band so vertical terrain is not pulled in.
        """
        ys, xs = np.where(best_mask)
        comp_top = int(ys.min())
        comp_bottom = int(ys.max()) + 1
        bridge_cells = max(1, int(self.config["silhouetteHorizontalBridgeCells"]))
        bridge_px = max(
            _MIN_HORIZONTAL_BRIDGE_PX,
            int(round(bridge_cells * desc_w / float(gate_width))),
        )
        if bridge_px % 2 == 0:
            bridge_px += 1
        band = np.zeros_like(binary_raw)
        band[comp_top:comp_bottom, :] = binary_raw[comp_top:comp_bottom, :]
        closed = cv2.morphologyEx(
            band,
            cv2.MORPH_CLOSE,
            _horizontal_bridge_kernel(bridge_px),
        )
        grown = best_mask.astype(np.uint8)
        for _ in range(max(1, bridge_px // 2)):
            grown = cv2.bitwise_and(
                cv2.dilate(grown, _MORPH_RECT_3, iterations=1), closed,
            )
        return grown.astype(bool) | best_mask

    def _maybe_deform_noisy_candidate(
        self,
        candidate: np.ndarray,
        refs: list[tuple[np.ndarray, np.ndarray]],
        mob_region: np.ndarray,
        descriptor: MobDescriptor,
        palette: np.ndarray,
        silhouette_distance: float,
        gate_mask,
    ) -> np.ndarray:
        """If soft/hard is noisy but recall is already ok, deform best ref into heat."""
        soft_hard_ratio = _occupancy_soft_hard_ratio(candidate)
        if soft_hard_ratio < _CONTENT_NOISE_SOFT_HARD_RATIO:
            return candidate
        _sim0, facing_idx, _scores0, _prec0, rec0 = best_silhouette_match(
            candidate, refs,
        )
        min_recall, _min_precision = self.silhouette_gate_thresholds()
        if rec0 < min_recall:
            return candidate
        ref_avg, ref_stable = refs[facing_idx]
        deformed_mask = self._deform_silhouette_occupancy(
            mob_region, descriptor, ref_avg, ref_stable,
        )
        return candidate_silhouette(
            mob_region,
            palette,
            silhouette_distance,
            gate_mask.width, gate_mask.height,
            occupancy_mask=deformed_mask,
        )

    def _deform_silhouette_occupancy(
        self,
        region_bgr: np.ndarray,
        descriptor: MobDescriptor,
        ref_avg: np.ndarray,
        ref_stable: np.ndarray,
    ) -> np.ndarray:
        """Deform a gate silhouette ref into palette heat at descriptor resolution.

        Base is the hard stable ref upsampled to the crop. Expansion is allowed
        only inside a band of radius
        ``_DEFORM_RADIUS_SILHOUETTE_CELLS × max(1, round(desc_w / gate_w))``
        silhouette grid cells in sprite pixels and only where
        ``heat >= minSpritePaletteMatch``. The base shape is always kept.
        """
        h, w = region_bgr.shape[:2]
        empty = np.zeros((h, w), dtype=bool)
        ref = np.asarray(ref_avg, dtype=np.float32)
        stable = np.asarray(ref_stable, dtype=bool).reshape(ref.shape)
        base_small = ((ref >= HARD_OCCUPANCY) & stable).astype(np.uint8)
        if not np.any(base_small):
            return empty

        # Work in the same descriptor-scale image that the candidate is
        # resized to immediately afterward. This preserves the silhouette
        # geometry while making deformation cost independent of a terrain-
        # merged source crop's raw pixel dimensions.
        desc_w, desc_h = _descriptor_sprite_size_px(descriptor)
        work_w = min(w, max(desc_w, int(ref.shape[1])))
        work_h = min(h, max(desc_h, int(ref.shape[0])))
        if work_w != w or work_h != h:
            work_region = cv2.resize(
                region_bgr, (work_w, work_h), interpolation=cv2.INTER_AREA,
            )
        else:
            work_region = region_bgr
        base = cv2.resize(
            base_small, (work_w, work_h), interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        heat = sprite_palette_heatmap(
            work_region,
            descriptor.match_palette_bgr,
            float(descriptor.max_sprite_palette_distance),
        )
        match_thr = float(self.config["minSpritePaletteMatch"])
        signal = heat >= match_thr

        gate_w = int(ref.shape[1])
        # The work image is descriptor-scale, so this is the intended number
        # of source pixels per silhouette cell—not a value derived from a
        # frame-sized noisy extract.
        cell_px = max(1, int(round(work_w / float(gate_w))))
        radius_px = _DEFORM_RADIUS_SILHOUETTE_CELLS * cell_px
        ksize = 2 * radius_px + 1
        band = cv2.dilate(
            base.astype(np.uint8), _ellipse_kernel(ksize), iterations=1,
        ).astype(bool)
        allowed = (band & signal) | base

        grown = base.astype(np.uint8)
        allowed_u8 = allowed.astype(np.uint8)
        # Each pass expands at most one source-pixel. The explicit bound above
        # keeps this loop descriptor-scale and prevents frame-sized work from
        # being repeated for noisy/merged extracts.
        for _ in range(radius_px):
            nxt = cv2.bitwise_and(
                cv2.dilate(grown, _MORPH_ELLIPSE_3, iterations=1), allowed_u8,
            )
            if np.array_equal(nxt, grown):
                break
            grown = nxt
        result = grown.astype(bool)
        if result.shape != (h, w):
            result = cv2.resize(
                result.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        return result

    def _shrink_bloated_extract_to_descriptor(
        self,
        search_region: np.ndarray,
        binary_raw: np.ndarray,
        best_mask: np.ndarray,
        desc_w: int,
        desc_h: int,
        search_x: int,
        search_y: int,
    ) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
        """Re-crop a terrain-bloated CC to a descriptor-sized window.

        Centers on the body-CC centroid, keeps palette match in-window that
        belongs to the connected component containing that centroid.
        """
        ys, xs = np.where(best_mask)
        cy = int(round(float(ys.mean())))
        cx = int(round(float(xs.mean())))
        sh, sw = search_region.shape[:2]
        left = max(0, cx - desc_w // 2)
        top = max(0, cy - desc_h // 2)
        right = min(sw, left + desc_w)
        bottom = min(sh, top + desc_h)
        left = max(0, right - desc_w)
        top = max(0, bottom - desc_h)

        mob_region = search_region[top:bottom, left:right]
        window = best_mask[top:bottom, left:right] | binary_raw[top:bottom, left:right].astype(bool)
        nlab, labels, _stats, _centroids = cv2.connectedComponentsWithStats(
            window.astype(np.uint8), connectivity=8,
        )
        local_y = cy - top
        local_x = cx - left
        if (
            nlab > 1
            and 0 <= local_y < labels.shape[0]
            and 0 <= local_x < labels.shape[1]
            and int(labels[local_y, local_x]) > 0
        ):
            comp_mask = labels == int(labels[local_y, local_x])
        else:
            comp_mask = best_mask[top:bottom, left:right]

        extract_bbox = (
            search_x + left,
            search_y + top,
            right - left,
            bottom - top,
        )
        return mob_region, comp_mask, extract_bbox

    # ------------------------------------------------------------------
    #  Per-point scoring  (kept for local_tracker — silhouette-based)
    # ------------------------------------------------------------------

    def score_at(
        self,
        frame_bgr: np.ndarray,
        descriptor: MobDescriptor,
        cx: int,
        cy: int,
        scale: float = 1.0,
    ) -> tuple[bool, tuple[int, int, int, int] | None, float]:
        """Score a point via the living silhouette gate (discovery / tracker).

        Returns (accepted, bbox, similarity).
        """
        return self._score_at_with_masks(
            frame_bgr,
            descriptor,
            cx,
            cy,
            scale,
            masks=descriptor.silhouette_masks,
        )

    def _score_at_with_masks(
        self,
        frame_bgr: np.ndarray,
        descriptor: MobDescriptor,
        cx: int,
        cy: int,
        scale: float,
        *,
        masks: list,
    ) -> tuple[bool, tuple[int, int, int, int] | None, float]:
        w = max(_MIN_DESCRIPTOR_PX, int(round(descriptor.avg_width * scale)))
        h = max(_MIN_DESCRIPTOR_PX, int(round(descriptor.avg_height * scale)))
        x = int(round(cx - w / 2))
        y = int(round(cy - h / 2))
        fh, fw = frame_bgr.shape[:2]
        if x < 0 or y < 0 or x + w > fw or y + h > fh:
            return False, None, 0.0

        bbox = (x, y, w, h)
        passed, sim, _cand, _idx, _scores, extract_bbox, _prec, _rec, _area = (
            self._evaluate_silhouette_gate(
                frame_bgr, descriptor, bbox, comp_bbox=bbox, masks=masks,
            )
        )
        return passed, extract_bbox if extract_bbox is not None else bbox, float(sim)

    # ------------------------------------------------------------------
    #  Tracking — delegates to local_tracker
    # ------------------------------------------------------------------

    def track_local(self, frame_bgr, mob_name, track, *, offset_x=0, offset_y=0,
                    search_radius_px=None, suppress_positions=None):
        from pybot.recognition.detector.tracking.local_tracker import track_local as run_track_local
        return run_track_local(
            self, frame_bgr, mob_name, track,
            offset_x=offset_x, offset_y=offset_y,
            search_radius_px=search_radius_px,
            suppress_positions=suppress_positions,
        )

    # ------------------------------------------------------------------
    #  Accept
    # ------------------------------------------------------------------

    def _finalize_accepted(self, candidates: list[DetectionCandidate]) -> list[DetectionCandidate]:
        accepted = [c for c in candidates if c.accepted]
        accepted.sort(key=lambda c: c.final_score, reverse=True)
        return accepted
