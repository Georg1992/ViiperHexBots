"""Sprite heatmap + silhouette-gate mob detector.

Pipeline: sprite palette heatmap → blobs → silhouette gate → NMS.
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
    best_silhouette_similarity,
    candidate_silhouette,
)
from pybot.recognition.detector.scoring.heatmap_detector import (
    HeatmapDetector,
    palette_heatmap,
    sprite_palette_heatmap,
)


REQUIRED_CONFIG_KEYS = {
    "minDiscoveryHeatmapScore",
    "discoveryHeatmapDownscale",
    "discoveryHeatmapDownscaleMinSide",
    "maxSpritePaletteDistance",
    "maxSilhouettePaletteDistance",
    "minSpritePaletteMatch",
    "minSilhouetteSimilarity",
    "topCandidateCenters",
    "minCenterHeat",
    "peakRelativeThreshold",
    "nmsDistancePx",
    "maxCandidates",
    "smallScaleMinFrameWidth",
    "smallScaleCutoff",
    "centerScales",
    "localTrackSearchRadiusPx",
    "minBlobFullScaleDimRatio",
    "minBlobAreaRatio",
    "minBlobSliverDimRatio",
    "minBlobHeatmapAreaRatio",
    "minBlobHeatmapAreaFloor",
    "minBlobModerateHeat",
    "minBlobAccentFootprint",
    "minBlobAccentPixelScore",
    "minBlobDimRatio",
    "maxBlobDimRatio",
    # death-detection keys (still used by local_tracker / opacity_probe)
    "deathOpacityBaselineSamples",
    "deathOpacityMinBaseline",
    "deathOpacityDecayRatio",
    "deathOpacityConfirmTicks",
    "deathRediscoveryCooldownMs",
    "deathOpacityMoveThresholdPx",
    "deathOpacityStopThresholdPx",
    "deathOpacityMinTrackAgeMs",
    "defaultAverageAttacksTillDeath",
    "attacksTillDeathHistoryWindow",
}


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
    bbox: tuple[int, int, int, int]
    comp_bbox: tuple[int, int, int, int]
    passed: bool
    similarity: float
    candidate_mask: list[float] | None = None
    matched_mask_index: int = 0
    mask_similarities: list[float] | None = None


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
        *,
        use_modified_descriptor: bool = False,
    ):
        self.project_root = project_root
        self.use_modified_descriptor = use_modified_descriptor
        self.config = load_detector_config() if config is None else config
        self.heatmap_detector = HeatmapDetector(self.config)
        self._descriptor_cache: dict[str, MobDescriptor] = {}
        self.min_discovery_heatmap_score = float(self.config["minDiscoveryHeatmapScore"])
        self.discovery_heatmap_downscale = int(self.config["discoveryHeatmapDownscale"])
        self.discovery_heatmap_downscale_min_side = int(self.config["discoveryHeatmapDownscaleMinSide"])
        self.local_track_search_radius_px = int(self.config.get("localTrackSearchRadiusPx", 120))

    def apply_runtime_config(self, config: dict) -> None:
        self.config = dict(config)
        self.heatmap_detector = HeatmapDetector(self.config)
        self.min_discovery_heatmap_score = float(self.config["minDiscoveryHeatmapScore"])
        self.discovery_heatmap_downscale = int(self.config["discoveryHeatmapDownscale"])
        self.discovery_heatmap_downscale_min_side = int(self.config["discoveryHeatmapDownscaleMinSide"])
        self.local_track_search_radius_px = int(self.config.get("localTrackSearchRadiusPx", 120))

    def descriptor_path(self, mob_name: str) -> Path:
        base = self.project_root / "assets" / "generated_descriptors"
        stem = mob_name.lower()
        if self.use_modified_descriptor:
            return base / "modified" / stem / "descriptor.json"
        return base / stem / "descriptor.json"

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
    #  Discovery pipeline: heatmap → blobs → silhouette gate → NMS
    # ------------------------------------------------------------------

    def detect(
        self,
        frame_bgr: np.ndarray,
        mob_name: str,
    ) -> DetectionResult:
        start = time.perf_counter()
        descriptor = self.ensure_descriptor(mob_name)

        hsv_start = time.perf_counter()
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        # --- heatmap --------------------------------------------------
        heatmap_start = time.perf_counter()
        fh, fw = frame_bgr.shape[:2]
        downscale = 1
        if self.discovery_heatmap_downscale > 1 and min(fw, fh) >= self.discovery_heatmap_downscale_min_side:
            downscale = self.discovery_heatmap_downscale

        sprite_heatmap = self.heatmap_detector.build_sprite_heatmap(
            frame_bgr,
            hsv,
            descriptor,
            downscale=downscale,
        )
        sprite_end = time.perf_counter()
        accent_heatmap = palette_heatmap(hsv, descriptor.accent_colors)
        accent_end = time.perf_counter()

        # --- blobs ----------------------------------------------------
        blobs = self.heatmap_detector.top_centers(sprite_heatmap)
        blobs_end = time.perf_counter()

        # --- validate each blob via silhouette gate -------------------
        candidates: list[DetectionCandidate] = []
        silhouette_checks: list[SilhouetteCheck] = []
        no_centers_found = len(blobs) == 0

        # Compute per-frame reference values among qualifying blobs (for relative filters)
        qualifying = [
            (cx, cy, h, bb)
            for cx, cy, h, bb in blobs
            if self._passes_blob_heat_filter(h, bb, descriptor, accent_heatmap)
        ]
        filter_end = time.perf_counter()
        multi_blob_frame = len(qualifying) > 1

        for cx, cy, heat_score, comp_bbox in qualifying:
            bx, by, bw, bh = comp_bbox

            if not self._passes_blob_size_filter(
                descriptor, comp_bbox, multi_blob_frame=multi_blob_frame,
            ):
                continue

            bbox = (bx, by, bw, bh)

            passed, similarity, candidate, matched_idx, scores = self._evaluate_silhouette_gate(
                frame_bgr,
                descriptor,
                bbox,
                comp_bbox=comp_bbox,
            )
            candidate_mask = (
                candidate.reshape(-1).tolist() if candidate is not None else None
            )
            silhouette_checks.append(SilhouetteCheck(
                center_x=cx,
                center_y=cy,
                heat_score=heat_score,
                bbox=bbox,
                comp_bbox=comp_bbox,
                passed=passed,
                similarity=similarity,
                candidate_mask=candidate_mask,
                matched_mask_index=matched_idx,
                mask_similarities=scores,
            ))
            if not passed:
                continue

            candidates.append(DetectionCandidate(
                mob_name=descriptor.mob_name,
                center_x=cx, center_y=cy,
                bbox=bbox,
                final_score=heat_score,
                heatmap_score=heat_score,
                accepted=True,
                rejection_reason="",
            ))

        # --- NMS ------------------------------------------------------
        nms_start = time.perf_counter()
        accepted = self._finalize_accepted(candidates)
        accepted_ids = {id(c) for c in accepted}

        for c in candidates:
            if c.accepted and id(c) not in accepted_ids:
                c.accepted = False
                c.rejection_reason = "nms_suppressed"
            if not c.accepted:
                if no_centers_found:
                    c.rejection_reason = f"discovery_fail:{c.rejection_reason}"
                elif not c.rejection_reason.startswith("discovery_fail:"):
                    c.rejection_reason = f"validation_fail:{c.rejection_reason}"

        elapsed = time.perf_counter() - start
        nms_end = time.perf_counter()
        timing = {
            "descriptor": hsv_start - start,
            "hsv": heatmap_start - hsv_start,
            "spriteHeatmap": sprite_end - heatmap_start,
            "accentHeatmap": accent_end - sprite_end,
            "blobCenters": blobs_end - accent_end,
            "blobFilters": filter_end - blobs_end,
            "silhouetteGate": nms_start - filter_end,
            "nms": nms_end - nms_start,
            "total": elapsed,
        }

        return DetectionResult(
            mob_name=mob_name.lower(),
            descriptor=descriptor,
            candidates=candidates[: int(self.config["maxCandidates"])],
            accepted=accepted,
            elapsed_s=elapsed,
            timing=timing,
            sprite_heatmap=sprite_heatmap,
            silhouette_checks=silhouette_checks,
        )

    # ------------------------------------------------------------------
    #  Blob pre-filters (descriptor-absolute, before silhouette gate)
    # ------------------------------------------------------------------

    def _min_blob_heatmap_area(self, descriptor: MobDescriptor) -> int:
        desc_min_w = descriptor.size.min_width
        desc_min_h = descriptor.size.min_height
        floor = int(self.config["minBlobHeatmapAreaFloor"])
        if desc_min_w is None or desc_min_h is None:
            return floor
        ratio_area = int(
            desc_min_w * desc_min_h * float(self.config["minBlobHeatmapAreaRatio"]),
        )
        return max(floor, ratio_area)

    @staticmethod
    def _blob_accent_footprint(
        accent_heatmap: np.ndarray,
        comp_bbox: tuple[int, int, int, int],
        accent_pixel_score: float,
    ) -> float:
        bx, by, bw, bh = comp_bbox
        accent_crop = accent_heatmap[by : by + bh, bx : bx + bw]
        if accent_crop.size == 0:
            return 0.0
        return float(np.mean(accent_crop >= accent_pixel_score))

    def _passes_blob_heat_filter(
        self,
        heat_score: float,
        comp_bbox: tuple[int, int, int, int],
        descriptor: MobDescriptor,
        accent_heatmap: np.ndarray,
    ) -> bool:
        """Reject heatmap blobs whose peak score or pixel footprint is too weak."""
        _bx, _by, bw, bh = comp_bbox
        if heat_score < self.min_discovery_heatmap_score:
            return False
        if bw * bh < self._min_blob_heatmap_area(descriptor):
            return False
        if heat_score >= float(self.config["minBlobModerateHeat"]):
            accent_footprint = self._blob_accent_footprint(
                accent_heatmap,
                comp_bbox,
                float(self.config["minBlobAccentPixelScore"]),
            )
            if accent_footprint < float(self.config["minBlobAccentFootprint"]):
                return False
        return True

    def _passes_blob_size_filter(
        self,
        descriptor: MobDescriptor,
        comp_bbox: tuple[int, int, int, int],
        *,
        multi_blob_frame: bool,
    ) -> bool:
        """Reject heatmap blobs whose bbox cannot plausibly be this mob.

        Size rules (descriptor-absolute, stable across hunt search range):
        1. Sliver — thinnest axis too small vs descriptor minimum.
        2. Area — when both axes are near full scale, bbox area must match.
        3. Bounds — single-blob minimum and oversized maximum vs descriptor.
        """
        _bx, _by, bw, bh = comp_bbox
        blob_area = bw * bh
        min_r = float(self.config["minBlobDimRatio"])
        max_r = float(self.config["maxBlobDimRatio"])
        desc_min_w = descriptor.size.min_width
        desc_min_h = descriptor.size.min_height
        desc_max_w = descriptor.size.max_width
        desc_max_h = descriptor.size.max_height

        if desc_min_w is not None and desc_min_h is not None:
            dim_min = min(bw / desc_min_w, bh / desc_min_h)
            if dim_min < float(self.config["minBlobSliverDimRatio"]):
                return False
            area_ratio = blob_area / (desc_min_w * desc_min_h)
            full_scale_dim = float(self.config["minBlobFullScaleDimRatio"])
            min_area = float(self.config["minBlobAreaRatio"])
            if dim_min >= full_scale_dim and area_ratio < min_area:
                return False

        if not multi_blob_frame and desc_min_w is not None and desc_min_h is not None:
            if bw < desc_min_w * min_r and bh < desc_min_h * min_r:
                return False
        if desc_max_w is not None and desc_max_h is not None:
            if bw > desc_max_w * max_r and bh > desc_max_h * max_r:
                return False
        return True

    # ------------------------------------------------------------------
    #  Silhouette gate  (component search + resize)
    # ------------------------------------------------------------------

    def _descriptor_silhouette_references(
        self,
        descriptor: MobDescriptor,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        if not descriptor.silhouette_masks:
            return []
        refs: list[tuple[np.ndarray, np.ndarray]] = []
        for mask in descriptor.silhouette_masks:
            if not mask.stable_mask or not any(mask.stable_mask):
                continue
            refs.append((
                np.array(mask.avg_mask, dtype=np.float32).reshape(mask.height, mask.width),
                np.array(mask.stable_mask, dtype=bool).reshape(mask.height, mask.width),
            ))
        return refs

    def _passes_silhouette_gate(
        self,
        frame_bgr: np.ndarray,
        descriptor: MobDescriptor,
        bbox: tuple[int, int, int, int],
        *,
        comp_bbox: tuple[int, int, int, int] | None = None,
    ) -> bool:
        passed, _, _, _, _ = self._evaluate_silhouette_gate(
            frame_bgr, descriptor, bbox, comp_bbox=comp_bbox,
        )
        return passed

    def _evaluate_silhouette_gate(
        self,
        frame_bgr: np.ndarray,
        descriptor: MobDescriptor,
        bbox: tuple[int, int, int, int],
        *,
        comp_bbox: tuple[int, int, int, int] | None = None,
    ) -> tuple[bool, float, np.ndarray | None, int, list[float]]:
        refs = self._descriptor_silhouette_references(descriptor)
        if not refs or not descriptor.match_palette_bgr:
            return True, 1.0, None, 0, []
        gate_mask = descriptor.silhouette_masks[0]
        x, y, w, h = bbox

        if comp_bbox is not None and comp_bbox[2] * comp_bbox[3] >= 500:
            hx, hy, hw, hh = comp_bbox
            ref_cx = hx + hw // 2
            ref_cy = hy + hh // 2
            ref_w = hw
            ref_h = hh
            search_x = max(0, ref_cx - hw)
            search_y = max(0, ref_cy - hh)
            search_w = min(frame_bgr.shape[1] - search_x, hw * 2)
            search_h = min(frame_bgr.shape[0] - search_y, hh * 2)
        elif w * h >= 500:
            ref_cx = x + w // 2
            ref_cy = y + h // 2
            ref_w = w
            ref_h = h
            search_x = max(0, ref_cx - w)
            search_y = max(0, ref_cy - h)
            search_w = min(frame_bgr.shape[1] - search_x, w * 2)
            search_h = min(frame_bgr.shape[0] - search_y, h * 2)
        else:
            desc_w = int(round(descriptor.avg_width))
            desc_h = int(round(descriptor.avg_height))
            ref_cx = x + w // 2
            ref_cy = y + h // 2
            ref_w = desc_w
            ref_h = desc_h
            search_x = max(0, ref_cx - desc_w)
            search_y = max(0, ref_cy - desc_h)
            search_w = min(frame_bgr.shape[1] - search_x, desc_w * 2)
            search_h = min(frame_bgr.shape[0] - search_y, desc_h * 2)

        search_region = frame_bgr[search_y : search_y + search_h, search_x : search_x + search_w]
        if search_region.size == 0:
            return False, 0.0, None, 0, []

        local_bbox_left = ref_cx - ref_w // 2 - search_x
        local_bbox_top = ref_cy - ref_h // 2 - search_y

        palette_heat = sprite_palette_heatmap(
            search_region, descriptor.match_palette_bgr,
            float(self.config["maxSpritePaletteDistance"]),
        )
        binary = (palette_heat >= float(self.config["minSpritePaletteMatch"])).astype(np.uint8)
        if not np.any(binary):
            return False, 0.0, None, 0, []

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.dilate(binary, kernel, iterations=1)

        _nl, labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if _nl <= 1:
            return False, 0.0, None, 0, []

        best_overlap = 0
        best_label = 0
        for lbl in range(1, _nl):
            cl = stats[lbl, cv2.CC_STAT_LEFT]
            ct = stats[lbl, cv2.CC_STAT_TOP]
            cr = cl + stats[lbl, cv2.CC_STAT_WIDTH]
            cb = ct + stats[lbl, cv2.CC_STAT_HEIGHT]
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
            return False, 0.0, None, 0, []

        comp_left = stats[best_label, cv2.CC_STAT_LEFT]
        comp_top = stats[best_label, cv2.CC_STAT_TOP]
        comp_w = stats[best_label, cv2.CC_STAT_WIDTH]
        comp_h = stats[best_label, cv2.CC_STAT_HEIGHT]

        if comp_w < 4 or comp_h < 4:
            return False, 0.0, None, 0, []

        comp_mask = labels[comp_top : comp_top + comp_h, comp_left : comp_left + comp_w] == best_label
        mob_region = search_region[comp_top : comp_top + comp_h, comp_left : comp_left + comp_w]
        if mob_region.size == 0:
            return False, 0.0, None, 0, []

        target_w = max(8, int(round(descriptor.avg_width)))
        target_h = max(8, int(round(descriptor.avg_height)))
        if mob_region.shape[1] != target_w or mob_region.shape[0] != target_h:
            mob_region = cv2.resize(mob_region, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            comp_mask = cv2.resize(
                comp_mask.astype(np.uint8),
                (target_w, target_h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

        silhouette_distance = float(self.config["maxSilhouettePaletteDistance"])
        candidate = candidate_silhouette(
            mob_region,
            np.asarray(descriptor.match_palette_bgr, dtype=np.float32),
            silhouette_distance,
            gate_mask.width, gate_mask.height,
            occupancy_mask=comp_mask,
        )
        similarity, matched_idx, scores = best_silhouette_similarity(candidate, refs)

        if similarity < float(self.config["minSilhouetteSimilarity"]):
            if comp_bbox is not None and comp_bbox[2] * comp_bbox[3] >= 500:
                return self._evaluate_silhouette_gate(
                    frame_bgr, descriptor, bbox, comp_bbox=None,
                )
            return False, float(similarity), candidate, matched_idx, scores
        return True, float(similarity), candidate, matched_idx, scores

    # ------------------------------------------------------------------
    #  Per-point scoring  (kept for local_tracker — silhouette-based)
    # ------------------------------------------------------------------

    def score_at(
        self,
        frame_bgr: np.ndarray,
        hsv: np.ndarray,
        descriptor: MobDescriptor,
        cx: int,
        cy: int,
        scale: float = 1.0,
    ) -> tuple[bool, tuple[int, int, int, int] | None, float]:
        """Score a point in the frame using silhouette check.

        Returns (accepted, bbox, similarity).  Used by local_tracker.
        """
        w = max(8, int(round(descriptor.avg_width * scale)))
        h = max(8, int(round(descriptor.avg_height * scale)))
        x = int(round(cx - w / 2))
        y = int(round(cy - h / 2))
        fh, fw = frame_bgr.shape[:2]
        if x < 0 or y < 0 or x + w > fw or y + h > fh:
            return False, None, 0.0

        bbox = (x, y, w, h)

        refs = self._descriptor_silhouette_references(descriptor)
        if not refs or not descriptor.match_palette_bgr:
            return True, bbox, 1.0

        gate_mask = descriptor.silhouette_masks[0]
        region = frame_bgr[y: y + h, x: x + w]
        if region.size == 0:
            return False, None, 0.0

        pal = np.asarray(descriptor.match_palette_bgr, dtype=np.float32)
        cand = candidate_silhouette(
            region, pal,
            float(self.config["maxSilhouettePaletteDistance"]),
            gate_mask.width, gate_mask.height,
        )
        sim, _, _ = best_silhouette_similarity(cand, refs)
        accepted = sim >= float(self.config["minSilhouetteSimilarity"])
        return accepted, bbox, float(sim)

    # ------------------------------------------------------------------
    #  Tracking — delegates to local_tracker
    # ------------------------------------------------------------------

    def track_local(self, frame_bgr, mob_name, track, *, offset_x=0, offset_y=0,
                    search_radius_px=None, death_detection_enabled=False):
        from pybot.recognition.detector.tracking.local_tracker import track_local as run_track_local
        return run_track_local(
            self, frame_bgr, mob_name, track,
            offset_x=offset_x, offset_y=offset_y,
            search_radius_px=search_radius_px,
            death_detection_enabled=death_detection_enabled,
        )

    # ------------------------------------------------------------------
    #  NMS
    # ------------------------------------------------------------------

    def _finalize_accepted(self, candidates: list[DetectionCandidate]) -> list[DetectionCandidate]:
        accepted = [c for c in candidates if c.accepted]
        accepted.sort(key=lambda c: c.final_score, reverse=True)
        return self._nms(accepted)

    def _nms(self, candidates: list[DetectionCandidate]) -> list[DetectionCandidate]:
        kept: list[DetectionCandidate] = []
        min_dist = int(self.config["nmsDistancePx"])
        min_dist_sq = min_dist * min_dist
        for c in sorted(candidates, key=lambda c: c.final_score, reverse=True):
            if all(
                (c.center_x - kc.center_x) ** 2 + (c.center_y - kc.center_y) ** 2 >= min_dist_sq
                for kc in kept
            ):
                kept.append(c)
        return kept
