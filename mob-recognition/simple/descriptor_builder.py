"""Build simple runtime descriptors from ACT-composed SPR frames."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

MOB_REC_DIR = Path(__file__).resolve().parent.parent
if str(MOB_REC_DIR) not in sys.path:
    sys.path.insert(0, str(MOB_REC_DIR))

from act_reader import ActReader
from frame_renderer import render_act_frame
from spr_reader import SprReader

from descriptor import ColorCluster, PatchSignature, SimpleMobDescriptor, SizeDescriptor

DESCRIPTOR_VERSION = 2
EXPORT_ACTIONS = (0, 1)


class SimpleDescriptorBuilder:
    def __init__(self, project_root: Path):
        self.project_root = project_root

    def asset_dir(self, mob_name: str) -> Path:
        return self.project_root / "assets" / "mobs" / mob_name.lower()

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
        frames = []
        action_indices = []
        for action_index in EXPORT_ACTIONS:
            if action_index >= len(act_file.actions):
                continue
            action = act_file.actions[action_index]
            action_indices.append(action_index)
            for frame_ref in action.frames:
                bgra = render_act_frame(spr_file, frame_ref)
                cropped = self._tight_crop(bgra)
                if cropped.shape[0] <= 1 or cropped.shape[1] <= 1:
                    continue
                frames.append((action_index, frame_ref.frame_index, cropped))
        if not frames:
            raise RuntimeError(f"no stand/walk frames could be rendered for {mob_name}")

        widths: list[int] = []
        heights: list[int] = []
        areas: list[int] = []
        opaque_bgr_parts: list[np.ndarray] = []
        opaque_hsv_parts: list[np.ndarray] = []
        accent_hsv_parts: list[np.ndarray] = []
        patch_signatures: list[PatchSignature] = []

        for _action_index, _frame_index, bgra in frames:
            bgr = bgra[:, :, :3]
            mask = (bgra[:, :, 3] > 0).astype(np.uint8) * 255
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            widths.append(int(bgra.shape[1]))
            heights.append(int(bgra.shape[0]))
            areas.append(int((mask > 0).sum()))
            opaque_bgr_parts.append(bgr[mask > 0])
            opaque_hsv_parts.append(hsv[mask > 0])

            accent_mask = self._accent_mask(bgr, hsv, mask)
            if np.any(accent_mask > 0):
                accent_hsv_parts.append(hsv[accent_mask > 0])
                patch_signatures.extend(self._patch_signatures(hsv, accent_mask, patch_size=5, limit=8))
                patch_signatures.extend(self._patch_signatures(hsv, accent_mask, patch_size=7, limit=6))

        opaque_bgr = np.concatenate(opaque_bgr_parts, axis=0)
        opaque_hsv = np.concatenate(opaque_hsv_parts, axis=0)
        if not accent_hsv_parts:
            raise ValueError(f"no accent pixels found while building descriptor for {mob_name}")
        accent_hsv = np.concatenate(accent_hsv_parts, axis=0)

        body_colors = self._clusters("body", opaque_bgr, opaque_hsv, count=6, tolerance=(18, 55, 55))
        accent_colors = self._clusters("accent", None, accent_hsv, count=4, tolerance=(16, 60, 65))
        rare_colors = self._rare_clusters(opaque_bgr, opaque_hsv)
        sprite_palette_bgr = self._sprite_palette(opaque_bgr)
        hsv_hist = self._hsv_histogram(opaque_hsv)
        rgb_hist = self._rgb_histogram(opaque_bgr)

        size = SizeDescriptor(
            avg_width=float(np.mean(widths)),
            avg_height=float(np.mean(heights)),
            min_width=int(min(widths)),
            max_width=int(max(widths)),
            min_height=int(min(heights)),
            max_height=int(max(heights)),
            aspect_ratio=float(np.mean(widths) / max(np.mean(heights), 1.0)),
            opaque_area=float(np.mean(areas)),
        )
        descriptor = SimpleMobDescriptor(
            mob_name=mob_name,
            version=DESCRIPTOR_VERSION,
            size=size,
            body_colors=body_colors,
            accent_colors=accent_colors,
            rare_colors=rare_colors,
            sprite_palette_bgr=sprite_palette_bgr,
            hsv_histogram=hsv_hist,
            rgb_histogram=rgb_hist,
            patch_signatures=patch_signatures[:120],
            template_count=len(frames),
            action_indices=action_indices,
        )
        descriptor.save(descriptor_path)
        return descriptor

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
    def _patch_signatures(hsv: np.ndarray, mask: np.ndarray, patch_size: int, limit: int) -> list[PatchSignature]:
        half = patch_size // 2
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return []
        stride = max(1, len(xs) // max(limit * 3, 1))
        signatures: list[PatchSignature] = []
        for cx, cy in zip(xs[::stride], ys[::stride]):
            x1, y1 = int(cx) - half, int(cy) - half
            x2, y2 = x1 + patch_size, y1 + patch_size
            if x1 < 0 or y1 < 0 or x2 > hsv.shape[1] or y2 > hsv.shape[0]:
                continue
            patch = hsv[y1:y2, x1:x2].astype(np.float32)
            signatures.append(
                PatchSignature(
                    patch_size=patch_size,
                    hsv_mean=tuple(float(v) for v in patch.reshape(-1, 3).mean(axis=0)),
                    hsv_std=tuple(float(v) for v in patch.reshape(-1, 3).std(axis=0)),
                    weight=1.0,
                )
            )
            if len(signatures) >= limit:
                break
        return signatures

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

    @staticmethod
    def _rgb_histogram(bgr_pixels: np.ndarray) -> list[float]:
        strip = bgr_pixels.astype(np.uint8).reshape(1, -1, 3)
        hist = cv2.calcHist([strip], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        return hist.reshape(-1).astype(float).tolist()

