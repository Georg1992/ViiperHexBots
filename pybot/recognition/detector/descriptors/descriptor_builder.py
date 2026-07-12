"""Build runtime descriptors from ACT-composed SPR frames."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from pybot.recognition.act_reader import ActReader
from pybot.recognition.frame_renderer import render_act_frame
from pybot.recognition.spr_reader import SprReader

from pybot.recognition.detector.descriptors.descriptor import (
    ColorCluster,
    LayoutGrid,
    MobDescriptor,
    SilhouetteMask,
    SizeDescriptor,
)
from pybot.recognition.detector.descriptors.layout_utils import (
    frame_alpha_occupancy_grid,
    frame_cluster_grid,
    frame_palette_coverage_grid,
    frame_silhouette,
)

DESCRIPTOR_VERSION = 13
# RO act layout: actions 0-7 stand/walk (4 facings), 8-15 attack/jump (4 facings).
# Pairs: (0,1) (2,3) (4,5) (6,7) | (8,9) (10,11) (12,13) (14,15).
# Actions 16+ (wide leap / special) are excluded by size auto-detect in
# _living_action_pairs() — not by a hardcoded action index cap.
STAND_WALK_ACTION_COUNT = 8
JUMP_ACTION_COUNT = 8
LIVING_FACING_ACTION_LIMIT = STAND_WALK_ACTION_COUNT + JUMP_ACTION_COUNT
LIVING_ACTION_SIZE_TOLERANCE = 0.40  # ±40% area vs stand/walk baseline
MATCH_PALETTE_MAX_COLORS = 20
MATCH_PALETTE_COVERAGE_DISTANCE = 20.0  # matches config maxSpritePaletteDistance
MIN_DISTINCTIVE_SATURATION = 40.0
MIN_DISTINCTIVE_VALUE = 30.0
LAYOUT_GRID_SIZE = 5
SILHOUETTE_WIDTH = 16
SILHOUETTE_HEIGHT = 16
GATE_SILHOUETTE_MASK_COUNT = 4
STABLE_CELL_OCCUPANCY = 0.08
STABLE_SILHOUETTE_VALUE = 0.20
MATCH_DISTANCE = 20.0


class DescriptorBuilder:
    def __init__(self, project_root: Path):
        self.project_root = project_root

    def asset_dir(self, mob_name: str) -> Path:
        mob_name_lower = mob_name.lower()
        base = self.project_root / "assets" / "mobs"
        # Fast path: direct lowercase folder name (e.g. horn/horn.spr)
        direct = base / mob_name_lower
        if (direct / f"{mob_name_lower}.spr").is_file():
            return direct
        # Fallback: scan folders for one containing the matching SPR file
        # This handles folder/file name mismatches (e.g. DesertWolf/desert_wolf.spr)
        if base.is_dir():
            for folder in sorted(base.iterdir()):
                if folder.is_dir() and (folder / f"{mob_name_lower}.spr").is_file():
                    return folder
        return direct  # will raise FileNotFoundError in build() with clear message

    def output_dir(self, mob_name: str) -> Path:
        return self.project_root / "assets" / "generated_descriptors" / mob_name.lower()

    def modified_output_dir(self, spr_stem: str) -> Path:
        return (
            self.project_root
            / "assets"
            / "generated_descriptors"
            / "modified"
            / spr_stem.lower()
        )

    def modified_asset_dir(self, asset_name: str, spr_stem: str) -> Path:
        return self.project_root / "assets" / "modified_mobs" / asset_name

    def build(self, mob_name: str, force: bool = False) -> MobDescriptor:
        mob_name = mob_name.lower()
        return self._build_from_asset_dir(
            mob_name,
            self.asset_dir(mob_name),
            spr_stem=mob_name,
            output_dir=self.output_dir(mob_name),
            force=force,
        )

    def build_modified(
        self,
        asset_name: str,
        spr_stem: str,
        force: bool = False,
    ) -> MobDescriptor:
        spr_stem = spr_stem.lower()
        asset_dir = self.project_root / "assets" / "modified_mobs" / asset_name
        return self._build_from_asset_dir(
            spr_stem,
            asset_dir,
            spr_stem=spr_stem,
            output_dir=self.modified_output_dir(spr_stem),
            force=force,
        )

    def _build_from_asset_dir(
        self,
        mob_name: str,
        asset_dir: Path,
        *,
        spr_stem: str,
        output_dir: Path,
        force: bool,
    ) -> MobDescriptor:
        output_dir.mkdir(parents=True, exist_ok=True)
        descriptor_path = output_dir / "descriptor.json"
        if descriptor_path.exists() and not force:
            return MobDescriptor.load(descriptor_path)

        spr_path = asset_dir / f"{spr_stem}.spr"
        act_path = asset_dir / f"{spr_stem}.act"
        if not spr_path.exists() or not act_path.exists():
            raise FileNotFoundError(f"missing SPR/ACT for mob '{mob_name}' in {asset_dir}")

        spr_file = SprReader(spr_path).load()
        act_file = ActReader(act_path).load()
        facing_pairs = self._living_action_pairs(act_file, spr_file)
        if not facing_pairs:
            raise RuntimeError(f"no stand/walk actions found for {mob_name}")
        all_facing_frames = self._collect_facing_frames(spr_file, act_file, facing_pairs)
        if not all_facing_frames:
            raise RuntimeError(f"no stand/walk frames could be rendered for {mob_name}")

        # Color clusters come from all facing directions so both front
        # and back views of the mob share the same body/accent palette.
        profile = self._build_frame_profile(all_facing_frames, dominant_tolerance=(12, 35, 55))
        profile["size"] = self._size_descriptor(all_facing_frames)

        profile_body_colors = self._distinctive_clusters(profile["body_colors"])
        if not profile_body_colors:
            raise RuntimeError(f"no distinctive body colors for {mob_name}")
        profile_dominant = profile_body_colors[0]
        profile_supporting = profile_body_colors[1:]
        profile_accent_colors = self._distinctive_clusters(profile["accent_colors"])
        if not profile_accent_colors:
            raise RuntimeError(f"no distinctive accent colors for {mob_name}")
        distinctive_rare = self._distinctive_clusters(profile["rare_colors"])
        match_palette_bgr = self._match_palette_union(spr_file, act_file, facing_pairs)
        # Enrich with accent + rare BGRs so the discovery heatmap includes
        # distinctive colours (e.g. purple shirt on Alligator) that may not
        # make the greedy coverage cut.
        seen_colors: set[tuple[int, int, int]] = set(match_palette_bgr)
        for cluster in profile_accent_colors:
            bgr = (int(cluster.bgr[0]), int(cluster.bgr[1]), int(cluster.bgr[2]))
            if bgr not in seen_colors:
                seen_colors.add(bgr)
                match_palette_bgr.append(bgr)
        for cluster in distinctive_rare:
            bgr = (int(cluster.bgr[0]), int(cluster.bgr[1]), int(cluster.bgr[2]))
            if bgr not in seen_colors:
                seen_colors.add(bgr)
                match_palette_bgr.append(bgr)
        # Cap at 20
        match_palette_bgr = match_palette_bgr[:MATCH_PALETTE_MAX_COLORS]
        dominant_pixels_bgr, accent_pixels_bgr = self._collect_structural_pixels(
            spr_file,
            act_file,
            facing_pairs,
        )
        dominant_pixel_bgr = dominant_pixels_bgr[0] if dominant_pixels_bgr else None
        accent_pixel_bgr = accent_pixels_bgr[0] if accent_pixels_bgr else None
        all_clusters = [profile_dominant, *profile_supporting, *profile_accent_colors, *distinctive_rare]
        layout_grid = self._build_layout_grid(all_facing_frames, all_clusters, match_palette_bgr)
        facing_silhouette_masks = self._build_facing_silhouette_masks(
            spr_file, act_file, facing_pairs,
        )
        if not facing_silhouette_masks:
            raise RuntimeError(f"no silhouette masks could be built for {mob_name}")
        silhouette_masks = self._build_gate_silhouette_masks(facing_silhouette_masks)

        descriptor = MobDescriptor(
            mob_name=mob_name,
            version=DESCRIPTOR_VERSION,
            size=profile["size"],
            dominant_color=profile_dominant,
            supporting_colors=profile_supporting,
            accent_colors=profile_accent_colors,
            rare_colors=distinctive_rare,
            match_palette_bgr=match_palette_bgr,
            hsv_histogram=profile["hsv_histogram"],
            dominant_pixel_bgr=dominant_pixel_bgr,
            accent_pixel_bgr=accent_pixel_bgr,
            dominant_pixels_bgr=dominant_pixels_bgr,
            accent_pixels_bgr=accent_pixels_bgr,
            layout_grid=layout_grid,
            facing_silhouette_masks=facing_silhouette_masks,
            silhouette_masks=silhouette_masks,
        )
        descriptor.save(descriptor_path)
        return descriptor

    @staticmethod
    def _living_facing_pairs(action_count: int) -> tuple[tuple[int, int], ...]:
        """Stand/walk and jump action pairs up to LIVING_FACING_ACTION_LIMIT."""
        limit = min(action_count, LIVING_FACING_ACTION_LIMIT)
        if limit <= 0:
            return ()
        if limit == 1:
            return ((0, 0),)
        pairs: list[tuple[int, int]] = []
        for start in range(0, limit - 1, 2):
            pairs.append((start, start + 1))
        return tuple(pairs)

    def _living_action_pairs(
        self,
        act_file,
        spr_file,
    ) -> tuple[tuple[int, int], ...]:
        """Auto-detect action pairs whose frame size matches stand/walk baseline.

        Considers stand/walk (0-7) and both jump rows (8-15). Each pair is
        included only when its first-frame bbox area is within
        LIVING_ACTION_SIZE_TOLERANCE of the stand/walk baseline. Stops at the
        first out-of-tolerance pair so wide leap poses (typically action 16+)
        never enter the descriptor.
        """
        raw_pairs = self._living_facing_pairs(len(act_file.actions))
        if not raw_pairs:
            return ()

        # Render first frame of the baseline pair (0,1)
        baseline_areas: list[float] = []
        for pair in raw_pairs:
            frames = self._collect_frames(spr_file, act_file, pair, frame_start=0)
            if frames:
                area = float(frames[0].shape[1] * frames[0].shape[0])
                baseline_areas.append(area)

        if not baseline_areas:
            return raw_pairs[:1]  # fallback: at least one pair

        baseline = float(np.median(baseline_areas[:2]))  # median of first 2 pairs
        if baseline <= 0:
            return raw_pairs[:1]

        max_area = baseline * (1.0 + LIVING_ACTION_SIZE_TOLERANCE)
        min_area = baseline * (1.0 - LIVING_ACTION_SIZE_TOLERANCE)

        kept: list[tuple[int, int]] = []
        for idx, pair in enumerate(raw_pairs):
            if idx < len(baseline_areas):
                area = baseline_areas[idx]
            else:
                # Compute on demand for pairs beyond the baseline
                frames = self._collect_frames(spr_file, act_file, pair, frame_start=0)
                if not frames:
                    continue
                area = float(frames[0].shape[1] * frames[0].shape[0])

            if min_area <= area <= max_area:
                kept.append(pair)
            else:
                break  # Stop at the first out-of-tolerance pair

        if not kept:
            return raw_pairs[:1]
        return tuple(kept)

    def _collect_facing_frames(
        self,
        spr_file,
        act_file,
        facing_pairs: tuple[tuple[int, int], ...],
    ) -> list[np.ndarray]:
        frames: list[np.ndarray] = []
        for pair in facing_pairs:
            frames.extend(self._collect_frames(spr_file, act_file, pair, frame_start=0))
        return frames

    def _match_palette_union(
        self,
        spr_file,
        act_file,
        facing_pairs: tuple[tuple[int, int], ...],
    ) -> list[tuple[int, int, int]]:
        """Greedy coverage palette union across facing pairs.

        Runs greedy selection per facing pair, then unions the results.
        This yields more total colors than running greedy on all frames
        at once because similar-but-distinct colors from different facings
        (e.g. front vs back) are kept separately.
        """
        seen: set[tuple[int, int, int]] = set()
        palette: list[tuple[int, int, int]] = []
        for pair in facing_pairs:
            frames = self._collect_frames(spr_file, act_file, pair, frame_start=0)
            if not frames:
                continue
            for color in self._match_palette_greedy(frames):
                if color in seen:
                    continue
                seen.add(color)
                palette.append(color)
                if len(palette) >= MATCH_PALETTE_MAX_COLORS:
                    return palette
        return palette

    def _collect_structural_pixels(
        self,
        spr_file,
        act_file,
        facing_pairs: tuple[tuple[int, int], ...],
    ) -> tuple[list[list[int]], list[list[int]]]:
        dominants: list[list[int]] = []
        accents: list[list[int]] = []
        for pair in facing_pairs:
            frames = self._collect_frames(spr_file, act_file, pair, frame_start=0)
            if not frames:
                continue
            dominant = self._find_dominant_pixel(frames)
            accent = self._find_accent_pixel(frames)
            if accent is None:
                continue
            if any(
                self._pixel_distance(dominant, existing) < 14.0
                for existing in dominants
            ):
                continue
            dominants.append(dominant)
            accents.append(accent)
        return dominants, accents

    @staticmethod
    def _pixel_distance(left: list[int], right: list[int]) -> float:
        delta = np.asarray(left, dtype=np.float32) - np.asarray(right, dtype=np.float32)
        return float(np.linalg.norm(delta))

    @staticmethod
    def _size_descriptor(frames: list[np.ndarray]) -> SizeDescriptor:
        widths: list[int] = []
        heights: list[int] = []
        for bgra in frames:
            widths.append(int(bgra.shape[1]))
            heights.append(int(bgra.shape[0]))
        return SizeDescriptor(
            avg_width=float(np.mean(widths)),
            avg_height=float(np.mean(heights)),
            min_width=float(min(widths)),
            max_width=float(max(widths)),
            min_height=float(min(heights)),
            max_height=float(max(heights)),
        )

    def _collect_frames(
        self,
        spr_file,
        act_file,
        action_indices: tuple[int, ...],
        *,
        frame_start: int,
    ) -> list[np.ndarray]:
        frames: list[np.ndarray] = []
        for action_index in action_indices:
            if action_index >= len(act_file.actions):
                continue
            action = act_file.actions[action_index]
            # Clip frame_start to available frames so we always take at least the last frame
            total = len(action.frames)
            start = min(frame_start, max(0, total - 1))
            for frame_index, frame_ref in enumerate(action.frames):
                if frame_index < start:
                    continue
                bgra = render_act_frame(spr_file, frame_ref)
                cropped = self._tight_crop(bgra)
                if cropped.shape[0] <= 1 or cropped.shape[1] <= 1:
                    continue
                frames.append(cropped)
        return frames

    def _build_frame_profile(self, frames: list[np.ndarray], *, dominant_tolerance: tuple[float, float, float] | None = None) -> dict:
        widths: list[int] = []
        heights: list[int] = []
        opaque_bgr_parts: list[np.ndarray] = []
        opaque_hsv_parts: list[np.ndarray] = []
        accent_hsv_parts: list[np.ndarray] = []

        for bgra in frames:
            bgr = bgra[:, :, :3]
            mask = (bgra[:, :, 3] >= 128).astype(np.uint8) * 255
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            widths.append(int(bgra.shape[1]))
            heights.append(int(bgra.shape[0]))
            opaque_bgr_parts.append(bgr[mask > 0])
            opaque_hsv_parts.append(hsv[mask > 0])

            accent_mask = self._accent_mask(bgr, hsv, mask)
            if np.any(accent_mask > 0):
                accent_hsv_parts.append(hsv[accent_mask > 0])

        opaque_bgr = np.concatenate(opaque_bgr_parts, axis=0)
        opaque_hsv = np.concatenate(opaque_hsv_parts, axis=0)
        if not accent_hsv_parts:
            raise ValueError("no accent pixels found while building frame profile")
        accent_hsv = np.concatenate(accent_hsv_parts, axis=0)

        body_colors = self._clusters("body", opaque_bgr, opaque_hsv, count=6, tolerance=(18, 55, 55))
        # Apply tighter tolerance to the dominant cluster only
        if dominant_tolerance is not None and body_colors:
            dc = body_colors[0]
            body_colors[0] = ColorCluster(
                label=dc.label,
                bgr=dc.bgr,
                hsv=dc.hsv,
                fraction=dc.fraction,
                tolerance=dominant_tolerance,
            )
        accent_colors = self._clusters("accent", None, accent_hsv, count=4, tolerance=(16, 60, 65))
        rare_colors = self._rare_clusters(opaque_bgr, opaque_hsv)
        hsv_hist = self._hsv_histogram(opaque_hsv)

        size = SizeDescriptor(
            avg_width=float(np.mean(widths)),
            avg_height=float(np.mean(heights)),
        )
        return {
            "size": size,
            "body_colors": body_colors,
            "accent_colors": accent_colors,
            "rare_colors": rare_colors,
            "hsv_histogram": hsv_hist,
        }

    @staticmethod
    def _find_dominant_pixel(frames: list[np.ndarray]) -> list[int]:
        """Find the single most common opaque pixel BGR value across all frames."""
        all_pixels: list[np.ndarray] = []
        for bgra in frames:
            mask = bgra[:, :, 3] >= 128
            if np.any(mask):
                all_pixels.append(bgra[:, :, :3][mask])
        if not all_pixels:
            return [0, 0, 0]
        stacked = np.concatenate(all_pixels, axis=0)
        unique, counts = np.unique(stacked, axis=0, return_counts=True)
        best_idx = int(np.argmax(counts))
        return [int(v) for v in unique[best_idx]]

    def _find_accent_pixel(self, frames: list[np.ndarray]) -> list[int] | None:
        accent_pixels: list[np.ndarray] = []
        for bgra in frames:
            bgr = bgra[:, :, :3]
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            mask = (bgra[:, :, 3] >= 128).astype(np.uint8) * 255
            accent_mask = self._accent_mask(bgr, hsv, mask)
            if np.any(accent_mask > 0):
                accent_pixels.append(bgr[accent_mask > 0])
        if not accent_pixels:
            return None
        stacked = np.concatenate(accent_pixels, axis=0)
        unique, counts = np.unique(stacked, axis=0, return_counts=True)
        best_idx = int(np.argmax(counts))
        return [int(v) for v in unique[best_idx]]

    @staticmethod
    def _tight_crop(bgra: np.ndarray) -> np.ndarray:
        alpha = bgra[:, :, 3]
        ys, xs = np.where(alpha >= 128)
        if len(xs) == 0:
            raise ValueError("cannot build descriptor from an empty sprite frame")
        return bgra[int(ys.min()) : int(ys.max()) + 1, int(xs.min()) : int(xs.max()) + 1].copy()

    @staticmethod
    def _accent_mask(bgr: np.ndarray, hsv: np.ndarray, mask: np.ndarray) -> np.ndarray:
        opaque = mask > 0
        if not np.any(opaque):
            return np.zeros(mask.shape, dtype=np.uint8)
        v = hsv[:, :, 2].astype(np.float32)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        contrast = gray - cv2.GaussianBlur(gray, (5, 5), 0)
        v_threshold = float(np.percentile(v[opaque], 72))
        c_threshold = max(5.0, float(np.percentile(contrast[opaque], 60)))
        accent = opaque & (v >= v_threshold) & (contrast >= c_threshold)
        saturation = hsv[:, :, 1].astype(np.float32)
        accent &= saturation >= MIN_DISTINCTIVE_SATURATION
        return accent.astype(np.uint8) * 255

    @staticmethod
    def _is_distinctive_bgr(bgr: tuple[int, int, int]) -> bool:
        pixel = np.uint8([[list(bgr)]])
        _hue, saturation, value = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0, 0]
        if float(saturation) < MIN_DISTINCTIVE_SATURATION:
            return False
        if float(value) < MIN_DISTINCTIVE_VALUE:
            return False
        if float(value) > 245.0 and float(saturation) < 30.0:
            return False
        return True
    @staticmethod
    def _is_distinctive_cluster(cluster: ColorCluster) -> bool:
        return DescriptorBuilder._is_distinctive_bgr(
            tuple(int(v) for v in cluster.bgr),
        )

    @classmethod
    def _distinctive_clusters(cls, clusters: list[ColorCluster]) -> list[ColorCluster]:
        return [cluster for cluster in clusters if cls._is_distinctive_cluster(cluster)]

    @staticmethod
    def _hsv_distance(a: ColorCluster, b: ColorCluster) -> float:
        h_diff = min(abs(a.hsv[0] - b.hsv[0]), 180.0 - abs(a.hsv[0] - b.hsv[0]))
        s_diff = a.hsv[1] - b.hsv[1]
        v_diff = a.hsv[2] - b.hsv[2]
        return float(np.sqrt(h_diff * h_diff + s_diff * s_diff + v_diff * v_diff))

    @classmethod
    def _union_clusters(
        cls,
        clusters: list[ColorCluster],
        max_dist: float,
    ) -> list[ColorCluster]:
        """Deduplicate clusters: keep highest-fraction cluster within HSV distance."""
        kept: list[ColorCluster] = []
        for candidate in sorted(clusters, key=lambda c: c.fraction, reverse=True):
            if all(cls._hsv_distance(candidate, existing) > max_dist for existing in kept):
                kept.append(candidate)
        return kept

    def _match_palette_greedy(self, frames: list[np.ndarray]) -> list[tuple[int, int, int]]:
        """Pick up to MATCH_PALETTE_MAX_COLORS colors that maximize coverage of sprite pixels.

        Uses greedy set cover: each iteration picks the color that covers the most
        currently uncovered pixels (within MATCH_PALETTE_COVERAGE_DISTANCE in BGR space).
        Called per-facing-pair through _match_palette_union so that similar colors
        from different facings (front vs back) are kept separately.
        """
        # Collect all opaque pixels
        all_pixels_list: list[np.ndarray] = []
        for bgra in frames:
            mask = bgra[:, :, 3] >= 128
            pixels = bgra[:, :, :3][mask]
            if len(pixels) > 0:
                all_pixels_list.append(pixels)

        if not all_pixels_list:
            raise ValueError("no frames with opaque pixels to build match palette")

        all_pixels = np.concatenate(all_pixels_list, axis=0)

        # Unique colors weighted by pixel count
        unique_colors, counts = np.unique(all_pixels, axis=0, return_counts=True)

        # Build candidate list: deduplicated colors sorted by frequency.
        # Only distinctive colors (saturation >= 40, value >= 30, not white)
        # are included — shadows, outlines, and gray/dull/muddy colors that
        # match the game background are excluded from the descriptor entirely.
        sorted_order = np.argsort(counts)[::-1]
        sorted_unique = unique_colors[sorted_order]
        dedup_local = self._deduplicate_palette_indices(sorted_unique)
        dedup_original_indices = [int(sorted_order[i]) for i in dedup_local]
        candidates: list[tuple[int, tuple[int, int, int]]] = [
            (i, tuple(int(v) for v in unique_colors[i]))
            for i in dedup_original_indices
            if DescriptorBuilder._is_distinctive_bgr(tuple(int(v) for v in unique_colors[i]))
        ]

        # Greedy coverage selection
        max_dist_sq = MATCH_PALETTE_COVERAGE_DISTANCE * MATCH_PALETTE_COVERAGE_DISTANCE
        uncovered = np.ones(len(unique_colors), dtype=bool)
        selected: list[tuple[int, int, int]] = []

        for _ in range(MATCH_PALETTE_MAX_COLORS):
            best_cand_idx = -1
            best_mass = 0

            for cand_idx, bgr_tuple in candidates:
                if not uncovered[cand_idx]:
                    continue
                # Distance from this candidate to all currently uncovered unique colors
                diff = unique_colors[uncovered].astype(np.float32) - unique_colors[cand_idx].astype(np.float32)
                dist_sq = np.sum(diff * diff, axis=1)
                # Pixel mass of uncovered colors within max_dist
                covered_mass = int(np.sum(counts[uncovered][dist_sq <= max_dist_sq]))
                if covered_mass > best_mass:
                    best_mass = covered_mass
                    best_cand_idx = cand_idx

            if best_cand_idx < 0 or best_mass == 0:
                break

            selected.append(tuple(int(v) for v in unique_colors[best_cand_idx]))

            # Mark all unique colors within max_dist of the chosen color as covered
            diff = unique_colors[uncovered].astype(np.float32) - unique_colors[best_cand_idx].astype(np.float32)
            dist_sq = np.sum(diff * diff, axis=1)
            uncovered_indices = np.where(uncovered)[0]
            covered_these = uncovered_indices[dist_sq <= max_dist_sq]
            uncovered[covered_these] = False

        if not selected:
            # Fallback: most common from candidates
            sorted_by_count = sorted(candidates, key=lambda x: int(counts[x[0]]), reverse=True)
            return [bgr for _, bgr in sorted_by_count[:MATCH_PALETTE_MAX_COLORS]]

        # Fill remaining slots with the highest-frequency candidates not yet selected.
        # This ensures we always use the full MATCH_PALETTE_MAX_COLORS budget even
        # when the greedy cover converges early (fewer than 48 distinct color groups).
        selected_set: set[tuple[int, int, int]] = set(selected)
        for cand_idx, bgr_tuple in candidates:
            if len(selected) >= MATCH_PALETTE_MAX_COLORS:
                break
            if bgr_tuple not in selected_set:
                selected_set.add(bgr_tuple)
                selected.append(bgr_tuple)

        return selected

    @staticmethod
    def _deduplicate_palette_indices(
        unique_colors: np.ndarray,
        h_thresh: float = 12.0,
        sv_thresh: float = 25.0,
    ) -> list[int]:
        """Return indices of unique_colors array that pass HSV dedup.

        For each pair of colors, if both hue and sat+val are within threshold,
        only the first occurrence (lower index) is kept.
        """
        if len(unique_colors) == 0:
            return []
        bgr_arr = unique_colors.reshape(-1, 1, 3).astype(np.uint8)
        hsv_arr = cv2.cvtColor(bgr_arr, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
        kept: list[int] = []
        for i in range(len(hsv_arr)):
            is_unique = True
            for j in kept:
                h_diff = min(abs(hsv_arr[i, 0] - hsv_arr[j, 0]), 180.0 - abs(hsv_arr[i, 0] - hsv_arr[j, 0]))
                s_diff = abs(hsv_arr[i, 1] - hsv_arr[j, 1])
                v_diff = abs(hsv_arr[i, 2] - hsv_arr[j, 2])
                if h_diff < h_thresh and s_diff < sv_thresh and v_diff < sv_thresh:
                    is_unique = False
                    break
            if is_unique:
                kept.append(i)
        return kept

    @staticmethod
    def _clusters(
        label: str,
        bgr_pixels: np.ndarray | None,
        hsv_pixels: np.ndarray,
        *,
        count: int,
        tolerance: tuple[float, float, float],
    ) -> list[ColorCluster]:
        if hsv_pixels.size == 0:
            return []
        samples = hsv_pixels.astype(np.float32)
        k = min(count, len(samples))
        _, labels, centers = cv2.kmeans(
            samples,
            k,
            None,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.4),
            3,
            cv2.KMEANS_PP_CENTERS,
        )
        clusters: list[ColorCluster] = []
        for idx, center in enumerate(centers):
            fraction = float((labels.ravel() == idx).mean())
            if fraction < 0.015:
                continue
            if bgr_pixels is None:
                center_hsv = np.uint8([[center]])
                center_bgr = cv2.cvtColor(center_hsv, cv2.COLOR_HSV2BGR)[0, 0].astype(np.float32)
            else:
                center_bgr = bgr_pixels[labels.ravel() == idx].mean(axis=0)
            clusters.append(
                ColorCluster(
                    label=f"{label}_{idx}",
                    bgr=tuple(float(v) for v in center_bgr),
                    hsv=tuple(float(v) for v in center),
                    fraction=fraction,
                    tolerance=tolerance,
                )
            )
        clusters.sort(key=lambda c: c.fraction, reverse=True)
        return clusters

    def _rare_clusters(self, bgr_pixels: np.ndarray, hsv_pixels: np.ndarray) -> list[ColorCluster]:
        clusters = self._clusters("rare", bgr_pixels, hsv_pixels, count=10, tolerance=(14, 45, 45))
        return [cluster for cluster in clusters if cluster.fraction <= 0.12][:4]

    @staticmethod
    def _hsv_histogram(hsv_pixels: np.ndarray) -> list[float]:
        strip = hsv_pixels.astype(np.uint8).reshape(1, -1, 3)
        hist = cv2.calcHist([strip], [0, 1], None, [24, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        return hist.reshape(-1).astype(float).tolist()

    def _build_layout_grid(
        self,
        frames: list[np.ndarray],
        clusters: list[ColorCluster],
        match_palette_bgr: list[tuple[int, int, int]],
    ) -> LayoutGrid:
        grid_size = LAYOUT_GRID_SIZE
        occupancy_frames: list[np.ndarray] = []
        coverage_frames: list[np.ndarray] = []
        cluster_frames: list[np.ndarray] = []
        for bgra in frames:
            alpha = bgra[:, :, 3]
            bgr = bgra[:, :, :3]
            occupancy_frames.append(frame_alpha_occupancy_grid(alpha, grid_size))
            coverage_frames.append(
                frame_palette_coverage_grid(bgr, alpha, match_palette_bgr, grid_size, MATCH_DISTANCE)
            )
            cluster_frames.append(frame_cluster_grid(bgr, alpha, clusters, grid_size))
        avg_occupancy = np.mean(np.stack(occupancy_frames, axis=0), axis=0).reshape(-1).tolist()
        palette_coverage = np.mean(np.stack(coverage_frames, axis=0), axis=0).reshape(-1).tolist()
        stable_occupied = [
            float(value) >= STABLE_CELL_OCCUPANCY for value in avg_occupancy
        ]
        dominant_cluster_ids: list[int] = []
        stacked_clusters = np.stack(cluster_frames, axis=0)
        for cell in range(grid_size * grid_size):
            gy, gx = divmod(cell, grid_size)
            values = stacked_clusters[:, gy, gx]
            valid = values[values >= 0]
            if valid.size == 0:
                dominant_cluster_ids.append(-1)
                continue
            counts = np.bincount(valid.astype(np.int32))
            dominant_cluster_ids.append(int(np.argmax(counts)))
        return LayoutGrid(
            grid_size=grid_size,
            avg_occupancy=avg_occupancy,
            stable_occupied=stable_occupied,
            dominant_cluster_ids=dominant_cluster_ids,
            palette_coverage=palette_coverage,
        )

    def _build_facing_silhouette_masks(
        self,
        spr_file,
        act_file,
        facing_pairs: tuple[tuple[int, int], ...],
    ) -> list[SilhouetteMask]:
        """One silhouette ref per facing pair (stand/walk/jump row).

        Frames within a pair share orientation; averaging across pairs would
        smear directional sprites into a single blob.
        """
        masks: list[SilhouetteMask] = []
        for pair in facing_pairs:
            frames = self._collect_frames(spr_file, act_file, pair, frame_start=0)
            if not frames:
                continue
            masks.append(self._build_silhouette_mask(frames))
        return masks

    @staticmethod
    def _silhouette_mask_binary(mask: SilhouetteMask) -> np.ndarray:
        avg = np.array(mask.avg_mask, dtype=np.float32).reshape(mask.height, mask.width)
        return (avg >= 0.5).astype(np.float32)

    @staticmethod
    def _binary_mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
        overlap = float(np.sum(np.minimum(mask_a, mask_b)))
        union = float(np.sum(np.maximum(mask_a, mask_b)))
        if union <= 0.0:
            return 1.0
        return overlap / union

    def _build_gate_silhouette_masks(
        self,
        facing_masks: list[SilhouetteMask],
    ) -> list[SilhouetteMask]:
        """Up to four gate refs: one per facing when ≤4, else recursive medoid clusters."""
        if len(facing_masks) <= GATE_SILHOUETTE_MASK_COUNT:
            return facing_masks
        return self._cluster_facing_masks_into_k(
            facing_masks, GATE_SILHOUETTE_MASK_COUNT,
        )

    def _split_facing_masks(
        self,
        masks: list[SilhouetteMask],
    ) -> tuple[list[SilhouetteMask], list[SilhouetteMask]]:
        binaries = [self._silhouette_mask_binary(mask) for mask in masks]
        best_i, best_j = 0, 1
        min_iou = 1.0
        for i in range(len(binaries)):
            for j in range(i + 1, len(binaries)):
                iou = self._binary_mask_iou(binaries[i], binaries[j])
                if iou < min_iou:
                    min_iou = iou
                    best_i, best_j = i, j

        clusters: list[list[SilhouetteMask]] = [[], []]
        seed_bins = [binaries[best_i], binaries[best_j]]
        for mask, binary in zip(masks, binaries):
            iou_a = self._binary_mask_iou(binary, seed_bins[0])
            iou_b = self._binary_mask_iou(binary, seed_bins[1])
            clusters[0 if iou_a >= iou_b else 1].append(mask)

        if not clusters[0] or not clusters[1]:
            raise RuntimeError("failed to split silhouette facing masks")
        return clusters[0], clusters[1]

    def _cluster_facing_masks_into_k(
        self,
        masks: list[SilhouetteMask],
        k: int,
    ) -> list[SilhouetteMask]:
        if k <= 0:
            raise ValueError("k must be positive")
        if len(masks) < k:
            raise RuntimeError(f"need at least {k} facing masks to select {k} gate refs")
        if len(masks) == k:
            return masks
        if k == 1:
            return [self._medoid_silhouette_mask(masks)]

        left, right = self._split_facing_masks(masks)
        left_k = k // 2
        right_k = k - left_k
        return (
            self._cluster_facing_masks_into_k(left, left_k)
            + self._cluster_facing_masks_into_k(right, right_k)
        )

    def _medoid_silhouette_mask(self, masks: list[SilhouetteMask]) -> SilhouetteMask:
        """Return the facing mask closest to all others in the cluster."""
        if len(masks) == 1:
            return masks[0]
        binaries = [self._silhouette_mask_binary(mask) for mask in masks]
        best_idx = 0
        best_score = -1.0
        for i in range(len(binaries)):
            score = sum(
                self._binary_mask_iou(binaries[i], binaries[j])
                for j in range(len(binaries))
                if j != i
            )
            if score > best_score:
                best_score = score
                best_idx = i
        return masks[best_idx]

    @staticmethod
    def _merge_facing_silhouette_masks(masks: list[SilhouetteMask]) -> SilhouetteMask:
        """Union per-facing refs into one mask that covers every orientation.

        Per-cell max (not mean) keeps each facing's shape without smearing
        opposing directions into a blob.
        """
        if len(masks) == 1:
            return masks[0]
        height = masks[0].height
        width = masks[0].width
        avg_stack = np.stack([
            np.array(mask.avg_mask, dtype=np.float32).reshape(height, width)
            for mask in masks
        ])
        merged_avg = np.max(avg_stack, axis=0)
        stable_stack = np.stack([
            np.array(mask.stable_mask, dtype=bool).reshape(height, width)
            for mask in masks
        ])
        merged_stable = np.any(stable_stack, axis=0)
        return SilhouetteMask(
            width=width,
            height=height,
            avg_mask=merged_avg.reshape(-1).tolist(),
            stable_mask=merged_stable.reshape(-1).tolist(),
        )

    def _build_silhouette_mask(self, frames: list[np.ndarray]) -> SilhouetteMask:
        masks = [
            frame_silhouette(bgra[:, :, 3], SILHOUETTE_WIDTH, SILHOUETTE_HEIGHT)
            for bgra in frames
        ]
        avg_mask = np.mean(np.stack(masks, axis=0), axis=0).reshape(-1).tolist()
        stable_mask = [float(value) >= STABLE_SILHOUETTE_VALUE for value in avg_mask]
        return SilhouetteMask(
            width=SILHOUETTE_WIDTH,
            height=SILHOUETTE_HEIGHT,
            avg_mask=avg_mask,
            stable_mask=stable_mask,
        )
