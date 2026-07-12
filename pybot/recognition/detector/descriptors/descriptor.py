"""Serializable descriptor for the heatmap mob detector."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


@dataclass
class ColorCluster:
    label: str
    bgr: tuple[float, float, float]
    hsv: tuple[float, float, float]
    fraction: float
    tolerance: tuple[float, float, float]


@dataclass
class SizeDescriptor:
    avg_width: float
    avg_height: float
    min_width: float | None = None
    max_width: float | None = None
    min_height: float | None = None
    max_height: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SizeDescriptor":
        return cls(
            avg_width=float(data.get("avg_width", data.get("avgWidth", 0))),
            avg_height=float(data.get("avg_height", data.get("avgHeight", 0))),
            min_width=_optional_float(data.get("min_width", data.get("minWidth"))),
            max_width=_optional_float(data.get("max_width", data.get("maxWidth"))),
            min_height=_optional_float(data.get("min_height", data.get("minHeight"))),
            max_height=_optional_float(data.get("max_height", data.get("maxHeight"))),
        )


@dataclass
class LayoutGrid:
    grid_size: int
    avg_occupancy: list[float]
    stable_occupied: list[bool]
    dominant_cluster_ids: list[int]
    palette_coverage: list[float]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LayoutGrid":
        return cls(
            grid_size=int(data.get("gridSize", data.get("grid_size", 5))),
            avg_occupancy=[float(v) for v in data.get("avgOccupancy", data.get("avg_occupancy", []))],
            stable_occupied=[bool(v) for v in data.get("stableOccupied", data.get("stable_occupied", []))],
            dominant_cluster_ids=[
                int(v) for v in data.get("dominantClusterIds", data.get("dominant_cluster_ids", []))
            ],
            palette_coverage=[float(v) for v in data.get("paletteCoverage", data.get("palette_coverage", []))],
        )


@dataclass
class SilhouetteMask:
    width: int
    height: int
    avg_mask: list[float]
    stable_mask: list[bool]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SilhouetteMask":
        return cls(
            width=int(data["width"]),
            height=int(data["height"]),
            avg_mask=[float(v) for v in data.get("avgMask", data.get("avg_mask", []))],
            stable_mask=[bool(v) for v in data.get("stableMask", data.get("stable_mask", []))],
        )


@dataclass
class MobDescriptor:
    mob_name: str
    version: int
    size: SizeDescriptor
    dominant_color: ColorCluster
    supporting_colors: list[ColorCluster]
    accent_colors: list[ColorCluster]
    rare_colors: list[ColorCluster]
    match_palette_bgr: list[tuple[int, int, int]]
    hsv_histogram: list[float]
    dominant_pixel_bgr: list[int] | None = None
    accent_pixel_bgr: list[int] | None = None
    dominant_pixels_bgr: list[list[int]] | None = None
    accent_pixels_bgr: list[list[int]] | None = None
    layout_grid: LayoutGrid | None = None
    facing_silhouette_masks: list[SilhouetteMask] | None = None
    silhouette_masks: list[SilhouetteMask] | None = None

    def structural_pixel_pairs(self) -> list[tuple[list[int], list[int]]]:
        """Per-facing dominant/accent pixels used by structural discovery gates."""
        dominants = self.dominant_pixels_bgr or (
            [self.dominant_pixel_bgr] if self.dominant_pixel_bgr is not None else []
        )
        accents = self.accent_pixels_bgr or (
            [self.accent_pixel_bgr] if self.accent_pixel_bgr is not None else []
        )
        if not dominants:
            return []
        if not accents:
            return [(dominant, dominant) for dominant in dominants]
        pairs: list[tuple[list[int], list[int]]] = []
        for index, dominant in enumerate(dominants):
            accent = accents[index] if index < len(accents) else accents[-1]
            pairs.append((dominant, accent))
        return pairs

    def effective_size_stats(self) -> dict:
        """Derived size bounds for debug / reporting."""
        width = self.size.avg_width
        height = self.size.avg_height
        min_w = self.size.min_width if self.size.min_width is not None else width * 0.55
        max_w = self.size.max_width if self.size.max_width is not None else width * 1.45
        min_h = self.size.min_height if self.size.min_height is not None else height * 0.55
        max_h = self.size.max_height if self.size.max_height is not None else height * 1.45
        aspect = width / max(height, 1.0)
        return {
            "minWidth": float(min_w),
            "maxWidth": float(max_w),
            "avgWidth": float(width),
            "minHeight": float(min_h),
            "maxHeight": float(max_h),
            "avgHeight": float(height),
            "minAspect": float(aspect * 0.75),
            "maxAspect": float(aspect * 1.35),
            "avgAspect": float(aspect),
        }

    def effective_occupancy_stats(self) -> dict:
        """Derived occupancy bounds for debug / reporting."""
        area = max(1.0, self.size.avg_width * self.size.avg_height)
        avg_opaque = area * 0.45
        return {
            "minOpaquePixels": int(max(8, avg_opaque * 0.35)),
            "maxOpaquePixels": int(avg_opaque * 1.8),
            "avgOpaquePixels": float(avg_opaque),
            "minDensity": 0.12,
            "maxDensity": 0.95,
            "avgDensity": 0.45,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MobDescriptor":
        if "dominantColor" in data:
            dominant_color = ColorCluster(**data["dominantColor"])
            supporting_colors = [ColorCluster(**item) for item in data.get("supportingColors", [])]
        elif "bodyColors" in data and data["bodyColors"]:
            legacy = [ColorCluster(**item) for item in data["bodyColors"]]
            dominant_color = legacy[0]
            supporting_colors = legacy[1:]
        else:
            dominant_color = ColorCluster(
                label="dominant", bgr=(0, 0, 0), hsv=(0, 0, 0), fraction=0.0, tolerance=(14, 40, 40)
            )
            supporting_colors = []

        raw_match = data.get("matchPaletteBgr")
        if raw_match is not None:
            match_palette_bgr = [tuple(int(v) for v in item) for item in raw_match]
        else:
            raw_sprite = data.get("spritePaletteBgr")
            if raw_sprite is not None:
                match_palette_bgr = [tuple(int(v) for v in item) for item in raw_sprite]
            else:
                match_palette_bgr = []

        raw_dominant = data.get("dominantPixelBgr")
        dominant_pixel_bgr = [int(v) for v in raw_dominant] if raw_dominant is not None else None
        raw_accent = data.get("accentPixelBgr")
        accent_pixel_bgr = [int(v) for v in raw_accent] if raw_accent is not None else None
        raw_dominants = data.get("dominantPixelsBgr")
        if raw_dominants is not None:
            dominant_pixels_bgr = [[int(v) for v in item] for item in raw_dominants]
        elif dominant_pixel_bgr is not None:
            dominant_pixels_bgr = [dominant_pixel_bgr]
        else:
            dominant_pixels_bgr = None
        raw_accents = data.get("accentPixelsBgr")
        if raw_accents is not None:
            accent_pixels_bgr = [[int(v) for v in item] for item in raw_accents]
        elif accent_pixel_bgr is not None:
            accent_pixels_bgr = [accent_pixel_bgr]
        else:
            accent_pixels_bgr = None

        layout_grid = LayoutGrid.from_dict(data["layoutGrid"]) if "layoutGrid" in data else None
        facing_silhouette_masks = None
        if "facingSilhouetteMasks" in data:
            facing_silhouette_masks = [
                SilhouetteMask.from_dict(item) for item in data["facingSilhouetteMasks"]
            ]
        silhouette_masks = None
        if "silhouetteMasks" in data:
            silhouette_masks = [
                SilhouetteMask.from_dict(item) for item in data["silhouetteMasks"]
            ]

        return cls(
            mob_name=str(data["mobName"]),
            version=int(data["version"]),
            size=SizeDescriptor.from_dict(data["size"]),
            dominant_color=dominant_color,
            supporting_colors=supporting_colors,
            accent_colors=[ColorCluster(**item) for item in data["accentColors"]],
            rare_colors=[ColorCluster(**item) for item in data["rareColors"]],
            match_palette_bgr=match_palette_bgr,
            hsv_histogram=[float(v) for v in data["hsvHistogram"]],
            dominant_pixel_bgr=dominant_pixel_bgr,
            accent_pixel_bgr=accent_pixel_bgr,
            dominant_pixels_bgr=dominant_pixels_bgr,
            accent_pixels_bgr=accent_pixels_bgr,
            layout_grid=layout_grid,
            facing_silhouette_masks=facing_silhouette_masks,
            silhouette_masks=silhouette_masks,
        )

    @classmethod
    def load(cls, path: Path) -> "MobDescriptor":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        payload: dict[str, Any] = {
            "mobName": data["mob_name"],
            "version": data["version"],
            "size": data["size"],
            "dominantColor": data["dominant_color"],
            "supportingColors": data["supporting_colors"],
            "accentColors": data["accent_colors"],
            "rareColors": data["rare_colors"],
            "matchPaletteBgr": data["match_palette_bgr"],
            "hsvHistogram": data["hsv_histogram"],
            "dominantPixelBgr": data.get("dominant_pixel_bgr"),
            "accentPixelBgr": data.get("accent_pixel_bgr"),
            "dominantPixelsBgr": data.get("dominant_pixels_bgr"),
            "accentPixelsBgr": data.get("accent_pixels_bgr"),
        }
        if self.layout_grid is not None:
            payload["layoutGrid"] = {
                "gridSize": self.layout_grid.grid_size,
                "avgOccupancy": self.layout_grid.avg_occupancy,
                "stableOccupied": self.layout_grid.stable_occupied,
                "dominantClusterIds": self.layout_grid.dominant_cluster_ids,
                "paletteCoverage": self.layout_grid.palette_coverage,
            }
        if self.facing_silhouette_masks is not None:
            payload["facingSilhouetteMasks"] = [
                {
                    "width": mask.width,
                    "height": mask.height,
                    "avgMask": mask.avg_mask,
                    "stableMask": mask.stable_mask,
                }
                for mask in self.facing_silhouette_masks
            ]
        if self.silhouette_masks is not None:
            payload["silhouetteMasks"] = [
                {
                    "width": mask.width,
                    "height": mask.height,
                    "avgMask": mask.avg_mask,
                    "stableMask": mask.stable_mask,
                }
                for mask in self.silhouette_masks
            ]
        return payload

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @property
    def avg_width(self) -> int:
        return max(1, int(round(self.size.avg_width)))

    @property
    def avg_height(self) -> int:
        return max(1, int(round(self.size.avg_height)))

    @property
    def body_palette(self) -> list[ColorCluster]:
        """Convenience: dominant + supporting colors for heatmap/scoring."""
        return [self.dominant_color] + self.supporting_colors
