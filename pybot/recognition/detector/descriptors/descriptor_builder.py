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
    MobDescriptor,
    SilhouetteMask,
    SizeDescriptor,
)
from pybot.recognition.detector.descriptors.layout_utils import (
    frame_silhouette,
)
from pybot.recognition.detector.descriptors.palette_groups import (
    cluster_match_palette_groups,
)

DESCRIPTOR_VERSION = 18
# RO act layout: actions 0-7 stand/walk (4 facings), 8-15 attack/jump (4 facings).
# Pairs: (0,1) (2,3) (4,5) (6,7) | (8,9) (10,11) (12,13) (14,15).
# Actions 16+ (wide leap / special) are excluded by size auto-detect in
# _living_action_pairs() — not by a hardcoded action index cap.
STAND_WALK_ACTION_COUNT = 8
JUMP_ACTION_COUNT = 8
LIVING_FACING_ACTION_LIMIT = STAND_WALK_ACTION_COUNT + JUMP_ACTION_COUNT
LIVING_ACTION_SIZE_TOLERANCE = 0.40  # ±40% area vs stand/walk baseline
MATCH_PALETTE_MAX_COLORS = 20
MATCH_PALETTE_MAX_ACCENT_COLORS = 4
PALETTE_DEDUP_H_THRESH = 12.0
PALETTE_DEDUP_SV_THRESH = 25.0
PALETTE_NEAR_BLACK_MAX_VALUE = 20.0
PALETTE_NEAR_BLACK_MAX_SATURATION = 25.0
PALETTE_NEAR_WHITE_MIN_VALUE = 250.0
PALETTE_NEAR_WHITE_MAX_SATURATION = 15.0
MIN_DISTINCTIVE_SATURATION = 40.0
MIN_DISTINCTIVE_VALUE = 30.0
BODY_CLUSTER_MAX_DISTANCE = 28.0
DOMINANT_CLUSTER_MAX_DISTANCE = 20.0
ACCENT_CLUSTER_MAX_DISTANCE = 30.0
SILHOUETTE_WIDTH = 16
SILHOUETTE_HEIGHT = 16
GATE_SILHOUETTE_MASK_COUNT = 4
STABLE_SILHOUETTE_VALUE = 0.20


def _bgr_value_saturation(bgr: tuple[int, int, int] | np.ndarray) -> tuple[float, float] | tuple[np.ndarray, np.ndarray]:
    """Value/saturation from BGR channels (OpenCV HSV V/S formulas, no cvtColor)."""
    if isinstance(bgr, tuple):
        value = float(max(bgr))
        chroma = float(max(bgr) - min(bgr))
        saturation = (chroma / value * 255.0) if value > 0.0 else 0.0
        return value, saturation
    bgr_f = bgr.astype(np.float32)
    value = bgr_f.max(axis=2)
    chroma = value - bgr_f.min(axis=2)
    saturation = np.zeros_like(value)
    np.divide(chroma, value, out=saturation, where=value > 0.0)
    saturation *= np.float32(255.0)
    return value, saturation


def _bgr_hue_sat_val(colors: np.ndarray) -> np.ndarray:
    """OpenCV-compatible HSV for Nx3 BGR colors, without cvtColor (H in 0..180)."""
    b = colors[:, 0].astype(np.float32)
    g = colors[:, 1].astype(np.float32)
    r = colors[:, 2].astype(np.float32)
    value = np.maximum(np.maximum(b, g), r)
    min_c = np.minimum(np.minimum(b, g), r)
    chroma = value - min_c
    saturation = np.zeros_like(value)
    np.divide(chroma, value, out=saturation, where=value > 0.0)
    saturation *= np.float32(255.0)

    hue = np.zeros_like(value)
    # Avoid div-by-zero on gray pixels; hue stays 0 there.
    safe = chroma > 0.0
    rb = safe & (value == b)
    rg = safe & (value == g) & ~rb
    rr = safe & (value == r) & ~rb & ~rg
    hue[rb] = 60.0 * ((r[rb] - g[rb]) / chroma[rb]) + 240.0
    hue[rg] = 60.0 * ((b[rg] - r[rg]) / chroma[rg]) + 120.0
    hue[rr] = 60.0 * ((g[rr] - b[rr]) / chroma[rr])
    hue = np.mod(hue, 360.0) * np.float32(0.5)  # OpenCV 0..180
    return np.stack([hue, saturation, value], axis=1)


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
        profile = self._build_frame_profile(all_facing_frames)
        profile["size"] = self._size_descriptor(all_facing_frames)

        profile_body_colors = self._distinctive_clusters(profile["body_colors"])
        if not profile_body_colors:
            raise RuntimeError(f"no distinctive body colors for {mob_name}")
        dc0 = profile_body_colors[0]
        profile_dominant = ColorCluster(
            label=dc0.label,
            bgr=dc0.bgr,
            fraction=dc0.fraction,
            max_distance=DOMINANT_CLUSTER_MAX_DISTANCE,
        )
        profile_supporting = profile_body_colors[1:]
        profile_accent_colors = self._distinctive_clusters(profile["accent_colors"])
        if not profile_accent_colors:
            raise RuntimeError(f"no distinctive accent colors for {mob_name}")
        match_palette_bgr, match_palette_weights = self._match_palette(all_facing_frames)
        match_palette_groups = cluster_match_palette_groups(match_palette_bgr)
        dominant_pixels_bgr, accent_pixels_bgr = self._collect_structural_pixels(
            spr_file,
            act_file,
            facing_pairs,
        )
        if not dominant_pixels_bgr or not accent_pixels_bgr:
            raise RuntimeError(f"no structural pixels for {mob_name}")
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
            match_palette_bgr=match_palette_bgr,
            match_palette_weights=match_palette_weights,
            match_palette_groups=match_palette_groups,
            dominant_pixels_bgr=dominant_pixels_bgr,
            accent_pixels_bgr=accent_pixels_bgr,
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

    @staticmethod
    def _is_scene_matching_speck(bgr: tuple[int, int, int]) -> bool:
        """Drop colors that match generic scene shadows/highlights instead of mob fill."""
        value, saturation = _bgr_value_saturation(bgr)
        if value <= PALETTE_NEAR_BLACK_MAX_VALUE and saturation <= PALETTE_NEAR_BLACK_MAX_SATURATION:
            return True
        if value >= PALETTE_NEAR_WHITE_MIN_VALUE and saturation <= PALETTE_NEAR_WHITE_MAX_SATURATION:
            return True
        return False

    @classmethod
    def _rank_accent_colors(
        cls,
        accent_counts: dict[tuple[int, int, int], int],
    ) -> list[tuple[int, int, int]]:
        if not accent_counts:
            return []
        ordered = sorted(accent_counts.items(), key=lambda item: item[1], reverse=True)
        unique_colors = np.asarray([list(bgr) for bgr, _count in ordered], dtype=np.uint8)
        dedup_local = cls._deduplicate_palette_indices(unique_colors)
        return [
            tuple(int(v) for v in unique_colors[idx])
            for idx in dedup_local
        ]

    @staticmethod
    def _collect_palette_pixel_data(
        frames: list[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, dict[tuple[int, int, int], int]]:
        """Return unique BGR colors sorted by mass, per-color counts, accent masses."""
        pixel_chunks: list[np.ndarray] = []
        accent_counts: dict[tuple[int, int, int], int] = {}

        for bgra in frames:
            opaque = bgra[:, :, 3] >= 128
            if not np.any(opaque):
                continue
            bgr = bgra[:, :, :3]
            pixel_chunks.append(bgr[opaque])
            mask = opaque.astype(np.uint8) * 255
            accent_mask = DescriptorBuilder._accent_mask(bgr, mask)
            accent_pixels = bgr[accent_mask > 0]
            if len(accent_pixels) == 0:
                continue
            unique_accent, accent_n = np.unique(accent_pixels, axis=0, return_counts=True)
            for color, count in zip(unique_accent, accent_n):
                key = tuple(int(v) for v in color)
                accent_counts[key] = accent_counts.get(key, 0) + int(count)

        if not pixel_chunks:
            raise ValueError("no frames with opaque pixels to build match palette")

        all_pixels = np.concatenate(pixel_chunks, axis=0)
        unique_colors, counts = np.unique(all_pixels, axis=0, return_counts=True)
        order = np.argsort(counts)[::-1]
        return unique_colors[order], counts[order], accent_counts

    @classmethod
    def _append_palette_colors(
        cls,
        palette: list[tuple[int, int, int]],
        candidates: list[tuple[int, int, int]],
        *,
        limit: int,
    ) -> None:
        palette_set = set(palette)
        for bgr in candidates:
            if len(palette) >= limit:
                return
            if cls._is_scene_matching_speck(bgr) or bgr in palette_set:
                continue
            palette.append(bgr)
            palette_set.add(bgr)

    def _match_palette(
        self,
        frames: list[np.ndarray],
    ) -> tuple[list[tuple[int, int, int]], list[float]]:
        """Build match palette from unique sprite fill colors plus accents and shades.

        Structural colors are ranked by opaque pixel mass with hue/sat/val shade dedup.
        Up to MATCH_PALETTE_MAX_ACCENT_COLORS accent-region colors are appended,
        then remaining slots are filled with the next-most-frequent shade variants.
        Each entry carries a weight proportional to opaque sprite pixel mass.
        """
        sorted_unique, counts, accent_counts = self._collect_palette_pixel_data(frames)
        dedup_local = self._deduplicate_palette_indices(sorted_unique)

        mass_by_color: dict[tuple[int, int, int], float] = {}
        for color, count in zip(sorted_unique, counts):
            key = tuple(int(v) for v in color)
            mass_by_color[key] = float(count)
        for color, count in accent_counts.items():
            mass_by_color[color] = mass_by_color.get(color, 0.0) + float(count)

        palette: list[tuple[int, int, int]] = []
        raw_weights: list[float] = []

        def append_weighted(candidates: list[tuple[int, int, int]], *, limit: int) -> None:
            palette_set = set(palette)
            for bgr in candidates:
                if len(palette) >= limit:
                    return
                if self._is_scene_matching_speck(bgr) or bgr in palette_set:
                    continue
                palette.append(bgr)
                raw_weights.append(mass_by_color.get(bgr, 1.0))
                palette_set.add(bgr)

        structural = [
            tuple(int(v) for v in sorted_unique[local_idx])
            for local_idx in dedup_local
        ]
        append_weighted(structural, limit=MATCH_PALETTE_MAX_COLORS)

        accent_budget = min(
            MATCH_PALETTE_MAX_ACCENT_COLORS,
            MATCH_PALETTE_MAX_COLORS - len(palette),
        )
        if accent_budget > 0:
            append_weighted(
                self._rank_accent_colors(accent_counts)[:accent_budget],
                limit=len(palette) + accent_budget,
            )

        shade_candidates = [
            tuple(int(v) for v in sorted_unique[idx])
            for idx in range(len(sorted_unique))
        ]
        append_weighted(shade_candidates, limit=MATCH_PALETTE_MAX_COLORS)

        if not palette:
            raise ValueError("no match palette colors after deduplication")

        peak = max(raw_weights)
        weights = [weight / peak for weight in raw_weights]
        return palette, weights

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

    def _build_frame_profile(
        self,
        frames: list[np.ndarray],
    ) -> dict:
        widths: list[int] = []
        heights: list[int] = []
        opaque_bgr_parts: list[np.ndarray] = []
        accent_bgr_parts: list[np.ndarray] = []

        for bgra in frames:
            bgr = bgra[:, :, :3]
            mask = (bgra[:, :, 3] >= 128).astype(np.uint8) * 255
            widths.append(int(bgra.shape[1]))
            heights.append(int(bgra.shape[0]))
            opaque_bgr_parts.append(bgr[mask > 0])

            accent_mask = self._accent_mask(bgr, mask)
            if np.any(accent_mask > 0):
                accent_bgr_parts.append(bgr[accent_mask > 0])

        opaque_bgr = np.concatenate(opaque_bgr_parts, axis=0)
        if not accent_bgr_parts:
            raise ValueError("no accent pixels found while building frame profile")
        accent_bgr = np.concatenate(accent_bgr_parts, axis=0)

        body_colors = self._clusters(
            "body",
            opaque_bgr,
            count=6,
            max_distance=BODY_CLUSTER_MAX_DISTANCE,
        )
        accent_colors = self._clusters(
            "accent",
            accent_bgr,
            count=4,
            max_distance=ACCENT_CLUSTER_MAX_DISTANCE,
        )

        size = SizeDescriptor(
            avg_width=float(np.mean(widths)),
            avg_height=float(np.mean(heights)),
        )
        return {
            "size": size,
            "body_colors": body_colors,
            "accent_colors": accent_colors,
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
            mask = (bgra[:, :, 3] >= 128).astype(np.uint8) * 255
            accent_mask = self._accent_mask(bgr, mask)
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
    def _accent_mask(bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        opaque = mask > 0
        if not np.any(opaque):
            return np.zeros(mask.shape, dtype=np.uint8)
        value, saturation = _bgr_value_saturation(bgr)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        contrast = gray - cv2.GaussianBlur(gray, (5, 5), 0)
        v_threshold = float(np.percentile(value[opaque], 72))
        c_threshold = max(5.0, float(np.percentile(contrast[opaque], 60)))
        accent = opaque & (value >= v_threshold) & (contrast >= c_threshold)
        accent &= saturation >= MIN_DISTINCTIVE_SATURATION
        return accent.astype(np.uint8) * 255

    @staticmethod
    def _is_distinctive_bgr(bgr: tuple[int, int, int]) -> bool:
        value, saturation = _bgr_value_saturation(bgr)
        if saturation < MIN_DISTINCTIVE_SATURATION:
            return False
        if value < MIN_DISTINCTIVE_VALUE:
            return False
        if value > 245.0 and saturation < 30.0:
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
    def _deduplicate_palette_indices(
        unique_colors: np.ndarray,
        h_thresh: float = PALETTE_DEDUP_H_THRESH,
        sv_thresh: float = PALETTE_DEDUP_SV_THRESH,
    ) -> list[int]:
        """Return indices of unique_colors that pass hue/sat/val shade dedup.

        Hue/sat/val are derived from BGR channel formulas (no cvtColor).
        """
        if len(unique_colors) == 0:
            return []
        hsv_arr = _bgr_hue_sat_val(unique_colors)
        kept: list[int] = []
        for i in range(len(hsv_arr)):
            is_unique = True
            for j in kept:
                h_diff = min(
                    abs(hsv_arr[i, 0] - hsv_arr[j, 0]),
                    180.0 - abs(hsv_arr[i, 0] - hsv_arr[j, 0]),
                )
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
        bgr_pixels: np.ndarray,
        *,
        count: int,
        max_distance: float,
    ) -> list[ColorCluster]:
        if bgr_pixels.size == 0:
            return []
        samples = bgr_pixels.astype(np.float32)
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
        for idx in range(len(centers)):
            fraction = float((labels.ravel() == idx).mean())
            if fraction < 0.015:
                continue
            center_bgr = samples[labels.ravel() == idx].mean(axis=0)
            clusters.append(
                ColorCluster(
                    label=f"{label}_{idx}",
                    bgr=tuple(float(v) for v in center_bgr),
                    fraction=fraction,
                    max_distance=max_distance,
                )
            )
        clusters.sort(key=lambda c: c.fraction, reverse=True)
        return clusters

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
