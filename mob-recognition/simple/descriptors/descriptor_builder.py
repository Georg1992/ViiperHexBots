"""Build simple runtime descriptors from ACT-composed SPR frames."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

MOB_REC_DIR = Path(__file__).resolve().parents[2]
if str(MOB_REC_DIR) not in sys.path:
    sys.path.insert(0, str(MOB_REC_DIR))

from act_reader import ActReader
from frame_renderer import render_act_frame
from spr_reader import SprReader

from descriptors.descriptor import ColorCluster, DeadStateProfile, SimpleMobDescriptor, SizeDescriptor

DESCRIPTOR_VERSION = 4
LIVING_ACTIONS = (0, 1)
DEATH_ACTIONS = tuple(range(32, 40))
DEATH_CORPSE_FRAME_START = 5


class SimpleDescriptorBuilder:
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
        return self.project_root / "generated_descriptors" / mob_name.lower() / "simple"

    def build(self, mob_name: str, force: bool = False) -> SimpleMobDescriptor:
        mob_name = mob_name.lower()
        output_dir = self.output_dir(mob_name)
        descriptor_path = output_dir / "descriptor.json"
        if descriptor_path.exists() and not force:
            return SimpleMobDescriptor.load(descriptor_path)

        spr_path = self.asset_dir(mob_name) / f"{mob_name}.spr"
        act_path = self.asset_dir(mob_name) / f"{mob_name}.act"
        if not spr_path.exists() or not act_path.exists():
            raise FileNotFoundError(f"missing SPR/ACT for mob '{mob_name}' in {self.asset_dir(mob_name)}")

        spr_file = SprReader(spr_path).load()
        act_file = ActReader(act_path).load()
        living_frames = self._collect_frames(
            spr_file,
            act_file,
            LIVING_ACTIONS,
            frame_start=0,
        )
        if not living_frames:
            raise RuntimeError(f"no stand/walk frames could be rendered for {mob_name}")

        dead_frames = self._collect_frames(
            spr_file,
            act_file,
            DEATH_ACTIONS,
            frame_start=DEATH_CORPSE_FRAME_START,
        )
        if not dead_frames:
            raise RuntimeError(f"no death corpse frames could be rendered for {mob_name}")

        # Build full profiles via k-means pipeline (accent_colors, body_colors, rare_colors, histograms)
        # Tighten hue+sat tolerance on the dominant cluster for more selective color matching
        living_profile = self._build_frame_profile(living_frames, dominant_tolerance=(12, 35, 55))
        dead_profile = self._build_frame_profile(dead_frames)

        # Living body colors: use k-means centroids (robust across animation frames)
        living_body_colors = living_profile["body_colors"]
        living_dominant = living_body_colors[0] if living_body_colors else living_profile["body_colors"][0]
        living_supporting = living_body_colors[1:] if len(living_body_colors) > 1 else []
        # Single most common pixel color across all sprite frames (not a cluster center)
        dominant_pixel_bgr = self._find_dominant_pixel(living_frames)

        # Death body colors: use per-frame intersection (stricter — corpse has fewer frame variants)
        dead_dominant_color, dead_supporting_colors = self._extract_dominant_colors(dead_frames)

        descriptor = SimpleMobDescriptor(
            mob_name=mob_name,
            version=DESCRIPTOR_VERSION,
            size=living_profile["size"],
            dominant_color=living_dominant,
            supporting_colors=living_supporting,
            accent_colors=living_profile["accent_colors"],
            rare_colors=living_profile["rare_colors"],
            sprite_palette_bgr=living_profile["sprite_palette_bgr"],
            hsv_histogram=living_profile["hsv_histogram"],
            dominant_pixel_bgr=dominant_pixel_bgr,
            dead=DeadStateProfile(
                size=dead_profile["size"],
                # Use per-frame intersection for corpse body colors
                body_colors=[dead_dominant_color] + dead_supporting_colors,
                accent_colors=dead_profile["accent_colors"],
                hsv_histogram=dead_profile["hsv_histogram"],
                sprite_palette_bgr=dead_profile["sprite_palette_bgr"],
            ),
        )
        descriptor.save(descriptor_path)
        return descriptor

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
            mask = (bgra[:, :, 3] > 0).astype(np.uint8) * 255
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
        sprite_palette_bgr = self._sprite_palette(opaque_bgr)
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
            "sprite_palette_bgr": sprite_palette_bgr,
            "hsv_histogram": hsv_hist,
        }

    def _extract_dominant_colors(
        self, frames: list[np.ndarray]
    ) -> tuple[ColorCluster, list[ColorCluster]]:
        """
        Find exact pixel colors that exist in all living frames, ranked by frequency.

        1. For each frame, collect the set of unique BGR colors.
        2. Intersect across all frames to find colors present in EVERY frame.
        3. Fall back to >= 90% threshold if no single color exists in all frames.
        4. Sort by total pixel count across all frames.
        5. Top = dominant_color, next up to 4 = supporting_colors.
        """
        per_frame_sets: list[set[tuple[int, int, int]]] = []
        per_frame_counts: list[dict[tuple[int, int, int], int]] = []

        for bgra in frames:
            bgr = bgra[:, :, :3]
            mask = bgra[:, :, 3] > 0
            pixels = bgr[mask]
            if len(pixels) == 0:
                continue
            unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
            color_set = {tuple(int(v) for v in c) for c in unique_colors}
            per_frame_sets.append(color_set)
            per_frame_counts.append(
                {
                    tuple(int(v) for v in c): int(counts[i])
                    for i, c in enumerate(unique_colors)
                }
            )

        if not per_frame_sets:
            raise ValueError("no frames with opaque pixels to extract dominant colors")

        # Intersection across all frames
        common_colors = per_frame_sets[0]
        for s in per_frame_sets[1:]:
            common_colors = common_colors & s
            if not common_colors:
                break

        # Fallback: present in >= 90% of frames
        if not common_colors:
            threshold = max(1, int(len(per_frame_sets) * 0.9))
            color_votes: dict[tuple[int, int, int], int] = {}
            for color_set in per_frame_sets:
                for color in color_set:
                    color_votes[color] = color_votes.get(color, 0) + 1
            common_colors = {c for c, v in color_votes.items() if v >= threshold}

        # Ultimate fallback: aggregate top color
        if not common_colors:
            all_pixels = np.concatenate(
                [bgra[:, :, :3][bgra[:, :, 3] > 0] for bgra in frames], axis=0
            )
            unique, counts = np.unique(all_pixels, axis=0, return_counts=True)
            top_idx = int(np.argmax(counts))
            top_color = tuple(int(v) for v in unique[top_idx])
            common_colors = {top_color}

        # Count total pixels for each common color across all frames
        total_counts: dict[tuple[int, int, int], int] = {}
        for frame_counts in per_frame_counts:
            for color, count in frame_counts.items():
                if color in common_colors:
                    total_counts[color] = total_counts.get(color, 0) + count

        sorted_colors = sorted(total_counts.items(), key=lambda x: x[1], reverse=True)
        total_pixels = sum(total_counts.values())

        def _bgr_to_hsv(bgr_tuple):
            pixel = np.uint8([[list(bgr_tuple)]])
            hsv = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0, 0]
            return tuple(float(v) for v in hsv)

        dominant_bgr = sorted_colors[0][0]
        dominant_color = ColorCluster(
            label="dominant",
            bgr=tuple(float(v) for v in dominant_bgr),
            hsv=_bgr_to_hsv(dominant_bgr),
            fraction=sorted_colors[0][1] / total_pixels if total_pixels > 0 else 0.0,
            tolerance=(14, 40, 40),
        )

        supporting: list[ColorCluster] = []
        for i, (bgr_color, count) in enumerate(sorted_colors[1:5], 1):
            supporting.append(
                ColorCluster(
                    label=f"supporting_{i}",
                    bgr=tuple(float(v) for v in bgr_color),
                    hsv=_bgr_to_hsv(bgr_color),
                    fraction=count / total_pixels if total_pixels > 0 else 0.0,
                    tolerance=(16, 45, 45),
                )
            )

        return dominant_color, supporting

    @staticmethod
    def _find_dominant_pixel(frames: list[np.ndarray]) -> list[int]:
        """Find the single most common opaque pixel BGR value across all frames."""
        all_pixels: list[np.ndarray] = []
        for bgra in frames:
            mask = bgra[:, :, 3] > 0
            if np.any(mask):
                all_pixels.append(bgra[:, :, :3][mask])
        if not all_pixels:
            return [0, 0, 0]
        stacked = np.concatenate(all_pixels, axis=0)
        unique, counts = np.unique(stacked, axis=0, return_counts=True)
        best_idx = int(np.argmax(counts))
        return [int(v) for v in unique[best_idx]]

    @staticmethod
    def _tight_crop(bgra: np.ndarray) -> np.ndarray:
        alpha = bgra[:, :, 3]
        ys, xs = np.where(alpha > 0)
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
        return accent.astype(np.uint8) * 255

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
    def _sprite_palette(bgr_pixels: np.ndarray) -> list[tuple[int, int, int]]:
        if bgr_pixels.size == 0:
            return []
        unique = np.unique(np.rint(bgr_pixels).astype(np.uint8), axis=0)
        unique = unique[np.lexsort((unique[:, 2], unique[:, 1], unique[:, 0]))]
        return [tuple(int(v) for v in color) for color in unique]

    @staticmethod
    def _hsv_histogram(hsv_pixels: np.ndarray) -> list[float]:
        strip = hsv_pixels.astype(np.uint8).reshape(1, -1, 3)
        hist = cv2.calcHist([strip], [0, 1], None, [24, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        return hist.reshape(-1).astype(float).tolist()
