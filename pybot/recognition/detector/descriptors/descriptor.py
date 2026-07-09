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
    hsv: tuple[float, float, float]
    fraction: float
    tolerance: tuple[float, float, float]


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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MobDescriptor":
        # Backward compat: if dominantColor is missing (legacy descriptor),
        # synthesize from the first bodyColors entry
        if "dominantColor" in data:
            dominant_color = ColorCluster(**data["dominantColor"])
            supporting_colors = [ColorCluster(**item) for item in data.get("supportingColors", [])]
        elif "bodyColors" in data and data["bodyColors"]:
            legacy = [ColorCluster(**item) for item in data["bodyColors"]]
            dominant_color = legacy[0]
            supporting_colors = legacy[1:]
        else:
            dominant_color = ColorCluster(
                label="dominant", bgr=(0,0,0), hsv=(0,0,0), fraction=0.0, tolerance=(14,40,40)
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
            "rareColors": data["rare_colors"],
            "spritePaletteBgr": data["sprite_palette_bgr"],
            "matchPaletteBgr": data["match_palette_bgr"],
            "hsvHistogram": data["hsv_histogram"],
            "dominantPixelBgr": data.get("dominant_pixel_bgr"),
            "accentPixelBgr": data.get("accent_pixel_bgr"),
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
