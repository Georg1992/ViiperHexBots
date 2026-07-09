"""Serializable descriptor for the heatmap mob detector."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
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
class SizeStats:
    min_width: float
    max_width: float
    avg_width: float
    std_width: float
    min_height: float
    max_height: float
    avg_height: float
    std_height: float
    min_aspect: float
    max_aspect: float
    avg_aspect: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SizeStats":
        return cls(
            min_width=float(data.get("minWidth", data.get("min_width", 0))),
            max_width=float(data.get("maxWidth", data.get("max_width", 0))),
            avg_width=float(data.get("avgWidth", data.get("avg_width", 0))),
            std_width=float(data.get("stdWidth", data.get("std_width", 0))),
            min_height=float(data.get("minHeight", data.get("min_height", 0))),
            max_height=float(data.get("maxHeight", data.get("max_height", 0))),
            avg_height=float(data.get("avgHeight", data.get("avg_height", 0))),
            std_height=float(data.get("stdHeight", data.get("std_height", 0))),
            min_aspect=float(data.get("minAspect", data.get("min_aspect", 0))),
            max_aspect=float(data.get("maxAspect", data.get("max_aspect", 0))),
            avg_aspect=float(data.get("avgAspect", data.get("avg_aspect", 0))),
        )


@dataclass
class OccupancyStats:
    min_opaque_pixels: int
    max_opaque_pixels: int
    avg_opaque_pixels: float
    min_density: float
    max_density: float
    avg_density: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OccupancyStats":
        return cls(
            min_opaque_pixels=int(data.get("minOpaquePixels", data.get("min_opaque_pixels", 0))),
            max_opaque_pixels=int(data.get("maxOpaquePixels", data.get("max_opaque_pixels", 0))),
            avg_opaque_pixels=float(data.get("avgOpaquePixels", data.get("avg_opaque_pixels", 0))),
            min_density=float(data.get("minDensity", data.get("min_density", 0))),
            max_density=float(data.get("maxDensity", data.get("max_density", 0))),
            avg_density=float(data.get("avgDensity", data.get("avg_density", 0))),
        )


@dataclass
class ColorStat:
    label: str
    bgr: tuple[float, float, float]
    hsv: tuple[float, float, float]
    frame_presence: float
    avg_fraction: float
    min_fraction: float
    max_fraction: float
    is_stable: bool
    is_distinctive: bool
    tolerance: tuple[float, float, float]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ColorStat":
        return cls(
            label=str(data["label"]),
            bgr=tuple(float(v) for v in data["bgr"]),
            hsv=tuple(float(v) for v in data["hsv"]),
            frame_presence=float(data.get("framePresence", data.get("frame_presence", 0))),
            avg_fraction=float(data.get("avgFraction", data.get("avg_fraction", 0))),
            min_fraction=float(data.get("minFraction", data.get("min_fraction", 0))),
            max_fraction=float(data.get("maxFraction", data.get("max_fraction", 0))),
            is_stable=bool(data.get("isStable", data.get("is_stable", False))),
            is_distinctive=bool(data.get("isDistinctive", data.get("is_distinctive", False))),
            tolerance=tuple(float(v) for v in data["tolerance"]),
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
    sprite_palette_bgr: list[tuple[int, int, int]]
    match_palette_bgr: list[tuple[int, int, int]]
    hsv_histogram: list[float]
    dominant_pixel_bgr: list[int] | None = None
    accent_pixel_bgr: list[int] | None = None
    dominant_pixels_bgr: list[list[int]] | None = None
    accent_pixels_bgr: list[list[int]] | None = None
    size_stats: SizeStats | None = None
    occupancy_stats: OccupancyStats | None = None
    color_stats: list[ColorStat] = field(default_factory=list)
    layout_grid: LayoutGrid | None = None
    silhouette_mask: SilhouetteMask | None = None

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

    def effective_size_stats(self) -> SizeStats:
        if self.size_stats is not None:
            return self.size_stats
        width = self.size.avg_width
        height = self.size.avg_height
        min_w = self.size.min_width if self.size.min_width is not None else width * 0.55
        max_w = self.size.max_width if self.size.max_width is not None else width * 1.45
        min_h = self.size.min_height if self.size.min_height is not None else height * 0.55
        max_h = self.size.max_height if self.size.max_height is not None else height * 1.45
        aspect = width / max(height, 1.0)
        return SizeStats(
            min_width=float(min_w),
            max_width=float(max_w),
            avg_width=float(width),
            std_width=0.0,
            min_height=float(min_h),
            max_height=float(max_h),
            avg_height=float(height),
            std_height=0.0,
            min_aspect=float(aspect * 0.75),
            max_aspect=float(aspect * 1.35),
            avg_aspect=float(aspect),
        )

    def effective_occupancy_stats(self) -> OccupancyStats:
        if self.occupancy_stats is not None:
            return self.occupancy_stats
        area = max(1.0, self.size.avg_width * self.size.avg_height)
        avg_opaque = area * 0.45
        return OccupancyStats(
            min_opaque_pixels=int(max(8, avg_opaque * 0.35)),
            max_opaque_pixels=int(avg_opaque * 1.8),
            avg_opaque_pixels=float(avg_opaque),
            min_density=0.12,
            max_density=0.95,
            avg_density=0.45,
        )

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
            match_palette_bgr = [tuple(int(v) for v in item) for item in data["spritePaletteBgr"]]

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

        size_stats = SizeStats.from_dict(data["sizeStats"]) if "sizeStats" in data else None
        occupancy_stats = (
            OccupancyStats.from_dict(data["occupancyStats"]) if "occupancyStats" in data else None
        )
        color_stats = [ColorStat.from_dict(item) for item in data.get("colorStats", [])]
        layout_grid = LayoutGrid.from_dict(data["layoutGrid"]) if "layoutGrid" in data else None
        silhouette_mask = (
            SilhouetteMask.from_dict(data["silhouetteMask"]) if "silhouetteMask" in data else None
        )

        return cls(
            mob_name=str(data["mobName"]),
            version=int(data["version"]),
            size=SizeDescriptor.from_dict(data["size"]),
            dominant_color=dominant_color,
            supporting_colors=supporting_colors,
            accent_colors=[ColorCluster(**item) for item in data["accentColors"]],
            rare_colors=[ColorCluster(**item) for item in data["rareColors"]],
            sprite_palette_bgr=[tuple(int(v) for v in item) for item in data["spritePaletteBgr"]],
            match_palette_bgr=match_palette_bgr,
            hsv_histogram=[float(v) for v in data["hsvHistogram"]],
            dominant_pixel_bgr=dominant_pixel_bgr,
            accent_pixel_bgr=accent_pixel_bgr,
            dominant_pixels_bgr=dominant_pixels_bgr,
            accent_pixels_bgr=accent_pixels_bgr,
            size_stats=size_stats,
            occupancy_stats=occupancy_stats,
            color_stats=color_stats,
            layout_grid=layout_grid,
            silhouette_mask=silhouette_mask,
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
            "spritePaletteBgr": data["sprite_palette_bgr"],
            "matchPaletteBgr": data["match_palette_bgr"],
            "hsvHistogram": data["hsv_histogram"],
            "dominantPixelBgr": data.get("dominant_pixel_bgr"),
            "accentPixelBgr": data.get("accent_pixel_bgr"),
            "dominantPixelsBgr": data.get("dominant_pixels_bgr"),
            "accentPixelsBgr": data.get("accent_pixels_bgr"),
        }
        if self.size_stats is not None:
            payload["sizeStats"] = {
                "minWidth": self.size_stats.min_width,
                "maxWidth": self.size_stats.max_width,
                "avgWidth": self.size_stats.avg_width,
                "stdWidth": self.size_stats.std_width,
                "minHeight": self.size_stats.min_height,
                "maxHeight": self.size_stats.max_height,
                "avgHeight": self.size_stats.avg_height,
                "stdHeight": self.size_stats.std_height,
                "minAspect": self.size_stats.min_aspect,
                "maxAspect": self.size_stats.max_aspect,
                "avgAspect": self.size_stats.avg_aspect,
            }
        if self.occupancy_stats is not None:
            payload["occupancyStats"] = {
                "minOpaquePixels": self.occupancy_stats.min_opaque_pixels,
                "maxOpaquePixels": self.occupancy_stats.max_opaque_pixels,
                "avgOpaquePixels": self.occupancy_stats.avg_opaque_pixels,
                "minDensity": self.occupancy_stats.min_density,
                "maxDensity": self.occupancy_stats.max_density,
                "avgDensity": self.occupancy_stats.avg_density,
            }
        if self.color_stats:
            payload["colorStats"] = [
                {
                    "label": stat.label,
                    "bgr": stat.bgr,
                    "hsv": stat.hsv,
                    "framePresence": stat.frame_presence,
                    "avgFraction": stat.avg_fraction,
                    "minFraction": stat.min_fraction,
                    "maxFraction": stat.max_fraction,
                    "isStable": stat.is_stable,
                    "isDistinctive": stat.is_distinctive,
                    "tolerance": stat.tolerance,
                }
                for stat in self.color_stats
            ]
        if self.layout_grid is not None:
            payload["layoutGrid"] = {
                "gridSize": self.layout_grid.grid_size,
                "avgOccupancy": self.layout_grid.avg_occupancy,
                "stableOccupied": self.layout_grid.stable_occupied,
                "dominantClusterIds": self.layout_grid.dominant_cluster_ids,
                "paletteCoverage": self.layout_grid.palette_coverage,
            }
        if self.silhouette_mask is not None:
            payload["silhouetteMask"] = {
                "width": self.silhouette_mask.width,
                "height": self.silhouette_mask.height,
                "avgMask": self.silhouette_mask.avg_mask,
                "stableMask": self.silhouette_mask.stable_mask,
            }
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

    def stable_match_palette(self) -> list[tuple[int, int, int]]:
        stable = [stat for stat in self.color_stats if stat.is_stable and stat.is_distinctive]
        if stable:
            return [tuple(int(v) for v in stat.bgr) for stat in stable]
        return self.match_palette_bgr
