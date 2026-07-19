"""Serializable descriptor for the heatmap mob detector."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class ColorCluster:
    label: str
    bgr: tuple[float, float, float]
    fraction: float
    max_distance: float


@dataclass
class SizeDescriptor:
    avg_width: float
    avg_height: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SizeDescriptor":
        return cls(
            avg_width=float(data.get("avg_width", data.get("avgWidth", 0))),
            avg_height=float(data.get("avg_height", data.get("avgHeight", 0))),
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
    match_palette_bgr: list[tuple[int, int, int]]
    match_palette_weights: list[float]
    match_palette_required: list[bool]
    match_palette_groups: list[list[int]]
    match_palette_required_groups: list[list[int]]
    match_palette_optional_groups: list[list[int]]
    max_sprite_palette_distance: float
    max_silhouette_palette_distance: float
    dominant_pixels_bgr: list[list[int]]
    accent_pixels_bgr: list[list[int]]
    silhouette_masks: list[SilhouetteMask]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MobDescriptor":
        if "dominantColor" not in data:
            raise ValueError("descriptor missing dominantColor")
        dominant_color = ColorCluster(**data["dominantColor"])
        supporting_colors = [ColorCluster(**item) for item in data.get("supportingColors", [])]

        if "matchPaletteBgr" not in data:
            raise ValueError("descriptor missing matchPaletteBgr")
        match_palette_bgr = [tuple(int(v) for v in item) for item in data["matchPaletteBgr"]]

        if "matchPaletteWeights" not in data:
            raise ValueError("descriptor missing matchPaletteWeights")
        match_palette_weights = [float(v) for v in data["matchPaletteWeights"]]
        if len(match_palette_weights) != len(match_palette_bgr):
            raise ValueError("matchPaletteWeights length must match matchPaletteBgr")

        if "matchPaletteRequired" not in data:
            raise ValueError("descriptor missing matchPaletteRequired")
        match_palette_required = [bool(v) for v in data["matchPaletteRequired"]]
        if len(match_palette_required) != len(match_palette_bgr):
            raise ValueError("matchPaletteRequired length must match matchPaletteBgr")

        if "dominantPixelsBgr" not in data:
            raise ValueError("descriptor missing dominantPixelsBgr")
        dominant_pixels_bgr = [[int(v) for v in item] for item in data["dominantPixelsBgr"]]

        if "accentPixelsBgr" not in data:
            raise ValueError("descriptor missing accentPixelsBgr")
        accent_pixels_bgr = [[int(v) for v in item] for item in data["accentPixelsBgr"]]

        if "silhouetteMasks" not in data:
            raise ValueError("descriptor missing silhouetteMasks")
        silhouette_masks = [
            SilhouetteMask.from_dict(item) for item in data["silhouetteMasks"]
        ]
        if not silhouette_masks:
            raise ValueError("descriptor silhouetteMasks must be non-empty")

        if "accentColors" not in data:
            raise ValueError("descriptor missing accentColors")

        if "matchPaletteGroups" not in data:
            raise ValueError("descriptor missing matchPaletteGroups")
        match_palette_groups = [
            [int(idx) for idx in group] for group in data["matchPaletteGroups"]
        ]

        if "matchPaletteRequiredGroups" not in data:
            raise ValueError("descriptor missing matchPaletteRequiredGroups")
        if "matchPaletteOptionalGroups" not in data:
            raise ValueError("descriptor missing matchPaletteOptionalGroups")
        match_palette_required_groups = [
            [int(idx) for idx in group] for group in data["matchPaletteRequiredGroups"]
        ]
        match_palette_optional_groups = [
            [int(idx) for idx in group] for group in data["matchPaletteOptionalGroups"]
        ]

        if "maxSpritePaletteDistance" not in data:
            raise ValueError("descriptor missing maxSpritePaletteDistance")
        if "maxSilhouettePaletteDistance" not in data:
            raise ValueError("descriptor missing maxSilhouettePaletteDistance")
        max_sprite_palette_distance = float(data["maxSpritePaletteDistance"])
        max_silhouette_palette_distance = float(data["maxSilhouettePaletteDistance"])
        if max_sprite_palette_distance <= 0.0 or max_silhouette_palette_distance <= 0.0:
            raise ValueError("palette distances must be positive")

        return cls(
            mob_name=str(data["mobName"]),
            version=int(data["version"]),
            size=SizeDescriptor.from_dict(data["size"]),
            dominant_color=dominant_color,
            supporting_colors=supporting_colors,
            accent_colors=[ColorCluster(**item) for item in data["accentColors"]],
            match_palette_bgr=match_palette_bgr,
            match_palette_weights=match_palette_weights,
            match_palette_required=match_palette_required,
            match_palette_groups=match_palette_groups,
            match_palette_required_groups=match_palette_required_groups,
            match_palette_optional_groups=match_palette_optional_groups,
            max_sprite_palette_distance=max_sprite_palette_distance,
            max_silhouette_palette_distance=max_silhouette_palette_distance,
            dominant_pixels_bgr=dominant_pixels_bgr,
            accent_pixels_bgr=accent_pixels_bgr,
            silhouette_masks=silhouette_masks,
        )

    @classmethod
    def load(cls, path: Path) -> "MobDescriptor":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return {
            "mobName": data["mob_name"],
            "version": data["version"],
            "size": data["size"],
            "dominantColor": data["dominant_color"],
            "supportingColors": data["supporting_colors"],
            "accentColors": data["accent_colors"],
            "matchPaletteBgr": data["match_palette_bgr"],
            "matchPaletteWeights": data["match_palette_weights"],
            "matchPaletteRequired": data["match_palette_required"],
            "matchPaletteGroups": data["match_palette_groups"],
            "matchPaletteRequiredGroups": data["match_palette_required_groups"],
            "matchPaletteOptionalGroups": data["match_palette_optional_groups"],
            "maxSpritePaletteDistance": data["max_sprite_palette_distance"],
            "maxSilhouettePaletteDistance": data["max_silhouette_palette_distance"],
            "dominantPixelsBgr": data["dominant_pixels_bgr"],
            "accentPixelsBgr": data["accent_pixels_bgr"],
            "silhouetteMasks": [
                {
                    "width": mask.width,
                    "height": mask.height,
                    "avgMask": mask.avg_mask,
                    "stableMask": mask.stable_mask,
                }
                for mask in self.silhouette_masks
            ],
        }

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
