"""Sprite heatmap + silhouette-gate mob detector.

Pipeline: sprite heatmap → blobs → geometry pre-gate → color-structure
pre-gate → silhouette gate → accept by heat score.
No RegionScorer, no structural pixels, no center refinement, no scales.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
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
    "discoveryClusterRadiusPx",
    "trackDedupRadiusPx",
    "debugOutputDir",
    # death-detection keys (local_tracker / opacity_probe / hunt)
    "deathOpacityBaselineSamples",
    "deathOpacityMinBaseline",
    "deathOpacityDropRatio",
    "deathOpacityConfirmMs",
    "trackJointAbsentConfirmMs",
    "deathRediscoveryCooldownMs",
    "deathOpacityMoveThresholdPx",
    "deathOpacityStopThresholdPx",
    "deathOpacityMinTrackAgeMs",
    "defaultAverageAttacksTillDeath",
    "attacksTillDeathHistoryWindow",
}

# Geometry pre-gate: heat-CC area must sit in [min_area_ratio, max_area_ratio]
# vs sprite area; aspect vs descriptor in
# [_GEOMETRY_ASPECT_MIN_RATIO, _GEOMETRY_ASPECT_MAX_RATIO].
_GEOMETRY_AREA_SIL_FRAC_DIVISOR = 4.0
_GEOMETRY_AREA_MAX_RATIO = 2.0
# Extract pre-shrink band floor still needs a universal guard; the runtime
# gate uses the descriptor's per-mob min_aspect_ratio / max_aspect_ratio
# (measured from sprite frames with a 45 % margin at build time).

# Extract / content-noise thresholds shared by silhouette gate control flow
# and the post-gate noisy_extract cleanup hook.
_EXTRACT_BLOAT_AREA_RATIO = 2.0
_CONTENT_NOISE_SOFT_HARD_RATIO = 2.0
# Full 16x16 hard fill = palette smear in a desc-sized window, not a sprite body.
_SOLID_FILL_HARD_FRACTION = 0.95

# Silhouette crop / morph / deform sizing.
_MIN_DESCRIPTOR_PX = 8
_MIN_EXTRACT_COMPONENT_PX = 4
_MIN_HORIZONTAL_BRIDGE_PX = 3
_DEFORM_RADIUS_SILHOUETTE_CELLS = 2
_MORPH_NEIGHBORHOOD_PX = 3


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



def load_detector_config(path: Optional[Path] = None) -> dict:
    config_path = path or (Path(__file__).resolve().parent / "detector_config.json")
    import json

    config = json.loads(config_path.read_text(encoding="utf-8"))
    missing = sorted(REQUIRED_CONFIG_KEYS - set(config))
    if missing:
        raise ValueError(f"missing detector config keys: {', '.join(missing)}")
    return config


class MobDetector:
    def __init__(
        self,
        project_root: Path,
        config: Optional[dict] = None,
    ):
        self.project_root = project_root
        self.config = load_detector_config() if config is None else config
        self.heatmap_detector = HeatmapDetector(self.config)
        self._descriptor_cache: dict[str, MobDescriptor] = {}
        self.discovery_heatmap_downscale = int(self.config["discoveryHeatmapDownscale"])
        self.discovery_heatmap_downscale_min_side = int(self.config["discoveryHeatmapDownscaleMinSide"])
        self.local_track_search_radius_px = int(self.config["localTrackSearchRadiusPx"])
        self.local_track_moving_search_radius_px = int(
            self.config["localTrackMovingSearchRadiusPx"]
        )

    def apply_runtime_config(self, config: dict) -> None:
        self.config = dict(config)
        self.heatmap_detector = HeatmapDetector(self.config)
        self.discovery_heatmap_downscale = int(self.config["discoveryHeatmapDownscale"])
        self.discovery_heatmap_downscale_min_side = int(self.config["discoveryHeatmapDownscaleMinSide"])
        self.local_track_search_radius_px = int(self.config["localTrackSearchRadiusPx"])
        self.local_track_moving_search_radius_px = int(
            self.config["localTrackMovingSearchRadiusPx"]
        )

    def descriptor_path(self, mob_name: str) -> Path:
        stem = mob_name.lower()
        return (
            self.project_root
            / "assets"
            / "generated_descriptors"
            / stem
            / "descriptor.json"
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

    def detect(
        self,
        frame_bgr: np.ndarray,
        mob_name: str,
        *,
        known_tracks: list[tuple[int, int, int, float]] | None = None,
    ) -> DetectionResult:
        """Heatmap discovery with known-track silhouette check.

        Order: heatmap → blobs → (new peaks: geometry + color structure) →
        silhouette. Known-track blobs skip geometry/color and score against
        living silhouettes only — death detection is owned by the tracker's
        opacity probe.
        """
        start = time.perf_counter()
        descriptor = self.ensure_descriptor(mob_name)

        # --- heatmap --------------------------------------------------
        heatmap_start = time.perf_counter()
        fh, fw = frame_bgr.shape[:2]
        downscale = 1
        if self.discovery_heatmap_downscale > 1 and min(fw, fh) >= self.discovery_heatmap_downscale_min_side:
            downscale = self.discovery_heatmap_downscale

        sprite_heatmap = self.heatmap_detector.build_sprite_heatmap(
            frame_bgr,
            descriptor,
            downscale=downscale,
        )
        heatmap_end = time.perf_counter()

        # --- blobs ----------------------------------------------------
        blobs = self.heatmap_detector.top_centers(sprite_heatmap, descriptor)
        blobs_end = time.perf_counter()

        dedup_radius = int(self.config["trackDedupRadiusPx"])
        known = list(known_tracks or ())
        blob_to_known = self._mark_known_blobs(blobs, known, dedup_radius)

        # --- gates → silhouette (known tracks skip pre-gates) ----------
        candidates: list[DetectionCandidate] = []
        silhouette_checks: list[SilhouetteCheck] = []

        for blob_index, (cx, cy, heat_score, comp_bbox) in enumerate(blobs):
            bx, by, bw, bh = comp_bbox
            bbox = (bx, by, bw, bh)
            known_hit = blob_to_known.get(blob_index)

            # New peaks must clear geometry + color structure. Known tracks were
            # already silhouette-confirmed when created — skip those pre-gates so
            # fading corpses can still reach silhouette scoring.
            if known_hit is None:
                if not self._passes_discovery_geometry_gate(comp_bbox, descriptor):
                    silhouette_checks.append(SilhouetteCheck(
                        center_x=cx,
                        center_y=cy,
                        heat_score=heat_score,
                        passed=False,
                        similarity=0.0,
                    ))
                    continue

                if not self._passes_color_structure_gate(
                    frame_bgr, descriptor, comp_bbox,
                ):
                    silhouette_checks.append(SilhouetteCheck(
                        center_x=cx,
                        center_y=cy,
                        heat_score=heat_score,
                        passed=False,
                        similarity=0.0,
                    ))
                    continue

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
            )
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

            if known_hit is None:
                # Newly detected peak: living silhouettes only.
                if passed:
                    candidates.append(DetectionCandidate(
                        mob_name=descriptor.mob_name,
                        center_x=cx, center_y=cy,
                        bbox=bbox,
                        final_score=heat_score,
                        heatmap_score=heat_score,
                        accepted=True,
                        rejection_reason="",
                    ))
                continue

            # Known track: living silhouette only (death is tracker-owned).
            if passed:
                candidates.append(DetectionCandidate(
                    mob_name=descriptor.mob_name,
                    center_x=cx, center_y=cy,
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
            "silhouetteGate": gate_end - blobs_end,
            "total": elapsed,
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

    @staticmethod
    def _mark_known_blobs(
        blobs: list[tuple[int, int, float, tuple[int, int, int, int]]],
        known_tracks: list[tuple[int, int, int, float]],
        dedup_radius: int,
    ) -> dict[int, tuple[int, int, int, float]]:
        """Map blob index → nearest known track within dedup radius (1:1)."""
        if not blobs or not known_tracks:
            return {}
        radius_sq = dedup_radius * dedup_radius
        claimed_tracks: set[int] = set()
        marked: dict[int, tuple[int, int, int, float]] = {}
        # Nearest pairs first so two peaks don't steal the same track poorly.
        pairs: list[tuple[int, int, int]] = []
        for blob_index, (cx, cy, _heat, _bbox) in enumerate(blobs):
            for track_id, kx, ky, _scale in known_tracks:
                dist = (int(cx) - int(kx)) ** 2 + (int(cy) - int(ky)) ** 2
                if dist <= radius_sq:
                    pairs.append((dist, blob_index, int(track_id)))
        pairs.sort(key=lambda item: item[0])
        track_by_id = {
            int(track_id): (int(track_id), int(kx), int(ky), float(scale))
            for track_id, kx, ky, scale in known_tracks
        }
        for _dist, blob_index, track_id in pairs:
            if blob_index in marked or track_id in claimed_tracks:
                continue
            known = track_by_id.get(track_id)
            if known is None:
                continue
            marked[blob_index] = known
            claimed_tracks.add(track_id)
        return marked

    # ------------------------------------------------------------------
    #  Geometry pre-gate + silhouette gate
    # ------------------------------------------------------------------

    def _passes_discovery_geometry_gate(
        self,
        comp_bbox: tuple[int, int, int, int],
        descriptor: MobDescriptor,
    ) -> bool:
        """Reject heat CCs whose size/aspect cannot plausibly match the mob.

        ``min_area_ratio = sil_frac / _GEOMETRY_AREA_SIL_FRAC_DIVISOR`` uses the
        descriptor's stable silhouette occupancy as a lower bound on heat-CC area
        vs sprite area. ``_GEOMETRY_AREA_MAX_RATIO`` caps terrain mega-blobs.
        Aspect vs descriptor sprite aspect uses the per-mob band
        ``descriptor.min_aspect_ratio`` / ``descriptor.max_aspect_ratio``
        (measured from sprite frames at build time with a 45 % margin).
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
        crop = frame_bgr[y0:y1, x0:x1]
        # Only use cached body map when it belongs to the same descriptor
        # to avoid cross-mob poisoning when detect() is called for
        # different mobs on the same HeatmapDetector instance.
        body_map = None
        body_ds = 0
        if self.heatmap_detector._last_body_descriptor_id == id(descriptor):
            body_map = self.heatmap_detector._last_body_best
            body_ds = self.heatmap_detector._last_body_downscale
        present, second_share, match_coverage, body_strong = required_groups_structure(
            crop,
            descriptor,
            float(descriptor.max_sprite_palette_distance),
            downscale=1,
            body_best_full=body_map,
            body_best_downscale=body_ds,
            crop_x=x0,
            crop_y=y0,
        )
        if min_groups > 0 and present < min_groups:
            return False
        if min_second > 0.0 and second_share < min_second:
            return False
        if min_coverage > 0.0 and match_coverage < min_coverage:
            return False
        if min_body_strong > 0.0 and body_strong < min_body_strong:
            return False

        return True

    def _descriptor_min_area_ratio(self, descriptor: MobDescriptor) -> float:
        """Mean stable silhouette occupancy across all facings, cached per descriptor.

        Cached on the descriptor object to avoid recomputing stable_bits per
        blob.  ``_GEOMETRY_AREA_SIL_FRAC_DIVISOR = 4.0`` already provides a
        75 % margin below the representative sprite footprint, so per-mask
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

    def _passes_size_aspect_vs_descriptor(
        self,
        width: int,
        height: int,
        descriptor: MobDescriptor,
        *,
        require_min_area: bool,
        enforce_max_area: bool = True,
    ) -> bool:
        """Descriptor-relative area + aspect band shared by heat and extract."""
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
        if require_min_area and area_ratio < self._descriptor_min_area_ratio(descriptor):
            return False
        if enforce_max_area and area_ratio > _GEOMETRY_AREA_MAX_RATIO:
            return False
        if aspect_ratio < descriptor.min_aspect_ratio or aspect_ratio > descriptor.max_aspect_ratio:
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
        refs: list[tuple[np.ndarray, np.ndarray]] = []
        for mask in masks:
            if not mask.stable_mask or not any(mask.stable_mask):
                continue
            refs.append((
                np.array(mask.avg_mask, dtype=np.float32).reshape(mask.height, mask.width),
                np.array(mask.stable_mask, dtype=bool).reshape(mask.height, mask.width),
            ))
        return refs



    def _evaluate_silhouette_gate(
        self,
        frame_bgr: np.ndarray,
        descriptor: MobDescriptor,
        bbox: tuple[int, int, int, int],
        *,
        comp_bbox: tuple[int, int, int, int] | None = None,
        masks: list | None = None,
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

        *masks* defaults to living ``descriptor.silhouette_masks``; pass death
        masks for corpse validation.
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

        palette_heat = sprite_palette_heatmap(
            search_region, descriptor.match_palette_bgr,
            float(descriptor.max_sprite_palette_distance),
        )
        binary_raw = (palette_heat >= float(self.config["minSpritePaletteMatch"])).astype(np.uint8)
        if not np.any(binary_raw):
            return fail

        k = _MORPH_NEIGHBORHOOD_PX
        binary = cv2.dilate(
            binary_raw,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)),
            iterations=1,
        )
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
        dual_ok = (
            recall >= float(self.config["minSilhouetteRecall"])
            and precision >= float(self.config["minSilhouettePrecision"])
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
            cv2.getStructuringElement(cv2.MORPH_RECT, (bridge_px, 1)),
        )
        grown = best_mask.astype(np.uint8)
        k = _MORPH_NEIGHBORHOOD_PX
        grow_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        for _ in range(max(1, bridge_px // 2)):
            grown = cv2.bitwise_and(cv2.dilate(grown, grow_kernel, iterations=1), closed)
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
        if rec0 < float(self.config["minSilhouetteRecall"]):
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

        base = cv2.resize(
            base_small, (w, h), interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        heat = sprite_palette_heatmap(
            region_bgr,
            descriptor.match_palette_bgr,
            float(descriptor.max_sprite_palette_distance),
        )
        match_thr = float(self.config["minSpritePaletteMatch"])
        signal = heat >= match_thr

        gate_w = int(ref.shape[1])
        cell_px = max(1, int(round(w / float(gate_w))))
        radius_px = _DEFORM_RADIUS_SILHOUETTE_CELLS * cell_px
        ksize = 2 * radius_px + 1
        band_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        band = cv2.dilate(base.astype(np.uint8), band_kernel, iterations=1).astype(bool)
        allowed = (band & signal) | base

        grown = base.astype(np.uint8)
        k = _MORPH_NEIGHBORHOOD_PX
        grow_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        allowed_u8 = allowed.astype(np.uint8)
        for _ in range(radius_px):
            nxt = cv2.bitwise_and(cv2.dilate(grown, grow_kernel, iterations=1), allowed_u8)
            if np.array_equal(nxt, grown):
                break
            grown = nxt
        return grown.astype(bool)

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

    def score_death_at(
        self,
        frame_bgr: np.ndarray,
        descriptor: MobDescriptor,
        cx: int,
        cy: int,
        scale: float = 1.0,
    ) -> tuple[bool, tuple[int, int, int, int] | None, float]:
        """Score a point against death/corpse silhouette refs.

        Returns (accepted, bbox, similarity). Empty death masks never accept.
        """
        if not descriptor.death_silhouette_masks:
            return False, None, 0.0
        return self._score_at_with_masks(
            frame_bgr,
            descriptor,
            cx,
            cy,
            scale,
            masks=descriptor.death_silhouette_masks,
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
                    search_radius_px=None, skip_opacity=False):
        from pybot.recognition.detector.tracking.local_tracker import track_local as run_track_local
        return run_track_local(
            self, frame_bgr, mob_name, track,
            offset_x=offset_x, offset_y=offset_y,
            search_radius_px=search_radius_px,
            skip_opacity=skip_opacity,
        )

    # ------------------------------------------------------------------
    #  Accept
    # ------------------------------------------------------------------

    def _finalize_accepted(self, candidates: list[DetectionCandidate]) -> list[DetectionCandidate]:
        accepted = [c for c in candidates if c.accepted]
        accepted.sort(key=lambda c: c.final_score, reverse=True)
        return accepted
