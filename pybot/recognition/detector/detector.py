"""Sprite heatmap + silhouette-gate mob detector.

Pipeline: sprite palette heatmap → blobs → silhouette gate.
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
    sprite_palette_heatmap,
)


REQUIRED_CONFIG_KEYS = {
    "discoveryHeatmapDownscale",
    "discoveryHeatmapDownscaleMinSide",
    "maxSpritePaletteDistance",
    "silhouettePaletteDistanceScale",
    "silhouetteHorizontalBridgeCells",
    "minSpritePaletteMatch",
    "minSilhouetteRecall",
    "minSilhouettePrecision",
    "usePaletteDiversity",
    "topCandidateCenters",
    "minCenterHeat",
    "peakRelativeThreshold",
    "maxCandidates",
    "smallScaleMinFrameWidth",
    "smallScaleCutoff",
    "centerScales",
    "localTrackSearchRadiusPx",
    "discoveryClusterRadiusPx",
    "trackDedupRadiusPx",
    "trackLostMissLimit",
    "debugOutputDir",
    # death-detection keys (local_tracker / opacity_probe / hunt)
    "deathOpacityBaselineSamples",
    "deathOpacityMinBaseline",
    "deathOpacityDecayRatio",
    "deathOpacityConfirmTicks",
    "deathRediscoveryDataMs",
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
        *,
        use_modified_descriptor: bool = False,
    ):
        self.project_root = project_root
        self.use_modified_descriptor = use_modified_descriptor
        self.config = load_detector_config() if config is None else config
        self.heatmap_detector = HeatmapDetector(self.config)
        self._descriptor_cache: dict[str, MobDescriptor] = {}
        self.discovery_heatmap_downscale = int(self.config["discoveryHeatmapDownscale"])
        self.discovery_heatmap_downscale_min_side = int(self.config["discoveryHeatmapDownscaleMinSide"])
        self.local_track_search_radius_px = int(self.config["localTrackSearchRadiusPx"])

    def apply_runtime_config(self, config: dict) -> None:
        self.config = dict(config)
        self.heatmap_detector = HeatmapDetector(self.config)
        self.discovery_heatmap_downscale = int(self.config["discoveryHeatmapDownscale"])
        self.discovery_heatmap_downscale_min_side = int(self.config["discoveryHeatmapDownscaleMinSide"])
        self.local_track_search_radius_px = int(self.config["localTrackSearchRadiusPx"])

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
    #  Discovery pipeline: heatmap → blobs → silhouette gate
    # ------------------------------------------------------------------

    def detect(
        self,
        frame_bgr: np.ndarray,
        mob_name: str,
    ) -> DetectionResult:
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

        # --- geometry pre-gate, then silhouette gate -------------------
        candidates: list[DetectionCandidate] = []
        silhouette_checks: list[SilhouetteCheck] = []

        for cx, cy, heat_score, comp_bbox in blobs:
            bx, by, bw, bh = comp_bbox
            bbox = (bx, by, bw, bh)

            if not self._passes_discovery_geometry_gate(comp_bbox, descriptor):
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
                extract_bbox, descriptor, candidate,
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

        gate_end = time.perf_counter()
        accepted = self._finalize_accepted(candidates)

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
            candidates=candidates[: int(self.config["maxCandidates"])],
            accepted=accepted,
            elapsed_s=elapsed,
            timing=timing,
            sprite_heatmap=sprite_heatmap,
            silhouette_checks=silhouette_checks,
        )

    # ------------------------------------------------------------------
    #  Geometry pre-gate + silhouette gate
    # ------------------------------------------------------------------

    def _passes_discovery_geometry_gate(
        self,
        comp_bbox: tuple[int, int, int, int],
        descriptor: MobDescriptor,
    ) -> bool:
        """Reject heat CCs whose size/aspect cannot plausibly match the mob.

        ``min_area_ratio = sil_frac / 4`` uses the descriptor's stable silhouette
        occupancy as a lower bound on heat-CC area vs sprite area.
        ``aspect_band = 3/2`` allows ±50% aspect slack vs descriptor sprite aspect.
        """
        _x, _y, hw, hh = comp_bbox
        if hw < 1 or hh < 1:
            return False

        desc_w = float(descriptor.avg_width)
        desc_h = float(descriptor.avg_height)
        desc_area = desc_w * desc_h
        desc_aspect = desc_w / desc_h

        stable_bits: list[bool] = []
        for mask in descriptor.silhouette_masks:
            if mask.stable_mask:
                stable_bits.extend(mask.stable_mask)
        if not stable_bits:
            return False
        sil_frac = float(np.mean(np.asarray(stable_bits, dtype=np.float32)))
        min_area_ratio = sil_frac / 4.0
        aspect_band = 3.0 / 2.0

        area_ratio = (float(hw) * float(hh)) / desc_area
        aspect_ratio = (float(hw) / float(hh)) / desc_aspect
        if area_ratio < min_area_ratio:
            return False
        if aspect_ratio < (1.0 / aspect_band) or aspect_ratio > aspect_band:
            return False
        return True

    def _noisy_extraction_signal(
        self,
        extract_bbox: tuple[int, int, int, int] | None,
        descriptor: MobDescriptor,
        candidate: np.ndarray | None,
    ) -> tuple[bool, bool, bool, float, float]:
        """Detect bloated crops and/or noisy silhouette *content*.

        ``extract_bloated``: extract area >= 2× descriptor sprite area (search-window
        fill from terrain merge). Large but clean crops (e.g. some Noxious) can be
        bloated without being content-noisy.

        ``content_noisy``: soft occupancy mass >= 2× hard mass on the 16×16 candidate.
        A compact sprite has soft ≈ O(perimeter) ≈ O(sqrt(hard)), so soft/hard ≪ 1.
        soft/hard >= 2 means the soft field dominates the hard body (terrain bleed /
        confetti), independent of bbox size.

        ``noisy_extract`` = bloated OR content_noisy. Cleanup hook only — no reject.
        Returns
        ``(noisy_extract, extract_bloated, content_noisy, extract_area_ratio, soft_hard_ratio)``.
        """
        extract_area_ratio = 0.0
        extract_bloated = False
        if extract_bbox is not None:
            _x, _y, ew, eh = extract_bbox
            if ew >= 1 and eh >= 1:
                desc_area = float(descriptor.avg_width) * float(descriptor.avg_height)
                extract_area_ratio = (float(ew) * float(eh)) / desc_area
                extract_bloated = extract_area_ratio >= 2.0

        soft_hard_ratio = 0.0
        content_noisy = False
        if candidate is not None and candidate.size > 0:
            hard = candidate >= HARD_OCCUPANCY
            soft = (candidate > 0) & ~hard
            hard_n = int(hard.sum())
            soft_n = int(soft.sum())
            if hard_n > 0:
                soft_hard_ratio = float(soft_n) / float(hard_n)
                # Soft mass at least 2× hard: cannot be a compact 1-cell halo.
                content_noisy = soft_hard_ratio >= 2.0

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

    def _evaluate_silhouette_gate(
        self,
        frame_bgr: np.ndarray,
        descriptor: MobDescriptor,
        bbox: tuple[int, int, int, int],
        *,
        comp_bbox: tuple[int, int, int, int] | None = None,
    ) -> tuple[
        bool,
        float,
        np.ndarray | None,
        int,
        list[float],
        tuple[int, int, int, int] | None,
        float,
        float,
    ]:
        """Silhouette gate matching a35ef47 crop style.

        Search around the heat CC (not sprite-inflated). Take the overlapping
        palette CC, bridge nearby same-row palette fragments horizontally
        (descriptor-scaled), crop tightly, then resize to descriptor size.
        Returns (passed, jaccard, candidate, matched_idx, scores, extract_bbox,
        precision, recall).
        """
        refs = self._descriptor_silhouette_references(descriptor)
        if not refs or not descriptor.match_palette_bgr:
            return False, 0.0, None, 0, [], None, 0.0, 0.0
        gate_mask = descriptor.silhouette_masks[0]
        x, y, w, h = bbox
        desc_w = max(8, int(round(descriptor.avg_width)))
        desc_h = max(8, int(round(descriptor.avg_height)))
        fh, fw = frame_bgr.shape[:2]

        # Search window: at least heat-CC size and at least descriptor size.
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
            return False, 0.0, None, 0, [], None, 0.0, 0.0

        local_bbox_left = ref_cx - ref_w // 2 - search_x
        local_bbox_top = ref_cy - ref_h // 2 - search_y

        palette_heat = sprite_palette_heatmap(
            search_region, descriptor.match_palette_bgr,
            float(descriptor.max_sprite_palette_distance),
        )
        binary_raw = (palette_heat >= float(self.config["minSpritePaletteMatch"])).astype(np.uint8)
        if not np.any(binary_raw):
            return False, 0.0, None, 0, [], None, 0.0, 0.0

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.dilate(binary_raw, kernel, iterations=1)

        _nl, labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if _nl <= 1:
            return False, 0.0, None, 0, [], None, 0.0, 0.0

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
            return False, 0.0, None, 0, [], None, 0.0, 0.0

        best_mask = labels == best_label
        ys, xs = np.where(best_mask)
        if len(xs) == 0:
            return False, 0.0, None, 0, [], None, 0.0, 0.0
        comp_left = int(xs.min())
        comp_right = int(xs.max()) + 1
        comp_top = int(ys.min())
        comp_bottom = int(ys.max()) + 1

        # Horizontally bridge body-height palette fragments that the dilate-CC
        # missed (patchy wings). Gap budget is N silhouette-grid cells mapped
        # into sprite pixels: bridge_px ≈ cells * desc_w / grid_w.
        # Geodesic grow stays inside the closed band so vertical terrain is not
        # pulled in.
        bridge_cells = max(1, int(self.config["silhouetteHorizontalBridgeCells"]))
        bridge_px = max(
            3,
            int(round(bridge_cells * desc_w / float(gate_mask.width))),
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
        grow_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        for _ in range(max(1, bridge_px // 2)):
            grown = cv2.bitwise_and(cv2.dilate(grown, grow_kernel, iterations=1), closed)
        occupancy = grown.astype(bool) | best_mask

        ys, xs = np.where(occupancy)
        if len(xs) == 0:
            return False, 0.0, None, 0, [], None, 0.0, 0.0
        comp_left = int(xs.min())
        comp_right = int(xs.max()) + 1
        comp_top = int(ys.min())
        comp_bottom = int(ys.max()) + 1
        comp_w = comp_right - comp_left
        comp_h = comp_bottom - comp_top
        if comp_w < 4 or comp_h < 4:
            return False, 0.0, None, 0, [], None, 0.0, 0.0

        desc_area = float(desc_w) * float(desc_h)
        extract_area_ratio = (float(comp_w) * float(comp_h)) / desc_area
        if extract_area_ratio >= 2.0:
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
            return False, 0.0, None, 0, [], extract_bbox, 0.0, 0.0

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
        candidate = candidate_silhouette(
            mob_region,
            np.asarray(descriptor.match_palette_bgr, dtype=np.float32),
            silhouette_distance,
            gate_mask.width, gate_mask.height,
            occupancy_mask=comp_mask,
        )
        similarity, matched_idx, scores, precision, recall = best_silhouette_match(
            candidate, refs,
        )
        passed = (
            recall >= float(self.config["minSilhouetteRecall"])
            and precision >= float(self.config["minSilhouettePrecision"])
        )
        return (
            passed,
            float(similarity),
            candidate,
            matched_idx,
            scores,
            extract_bbox,
            float(precision),
            float(recall),
        )

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
        """Score a point via the same silhouette gate as discovery.

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
        passed, sim, _cand, _idx, _scores, extract_bbox, _prec, _rec = (
            self._evaluate_silhouette_gate(
                frame_bgr, descriptor, bbox, comp_bbox=bbox,
            )
        )
        return passed, extract_bbox if extract_bbox is not None else bbox, float(sim)

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
    #  Accept
    # ------------------------------------------------------------------

    def _finalize_accepted(self, candidates: list[DetectionCandidate]) -> list[DetectionCandidate]:
        accepted = [c for c in candidates if c.accepted]
        accepted.sort(key=lambda c: c.final_score, reverse=True)
        return accepted
