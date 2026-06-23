"""Serializable descriptor for the simple heatmap detector."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
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
    min_width: int
    max_width: int
    min_height: int
    max_height: int
    aspect_ratio: float
    opaque_area: float


@dataclass
class PatchSignature:
    patch_size: int
    hsv_mean: tuple[float, float, float]
    hsv_std: tuple[float, float, float]
    weight: float


@dataclass
class SimpleMobDescriptor:
    mob_name: str
    version: int
    size: SizeDescriptor
    body_colors: list[ColorCluster]
    accent_colors: list[ColorCluster]
    rare_colors: list[ColorCluster]
    sprite_palette_bgr: list[tuple[int, int, int]]
    hsv_histogram: list[float]
    rgb_histogram: list[float]
    patch_signatures: list[PatchSignature] = field(default_factory=list)
    template_count: int = 0
    action_indices: list[int] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SimpleMobDescriptor":
        return cls(
            mob_name=str(data["mobName"]),
            version=int(data.get("version", 1)),
            size=SizeDescriptor(**data["size"]),
            body_colors=[ColorCluster(**item) for item in data.get("bodyColors", [])],
            accent_colors=[ColorCluster(**item) for item in data.get("accentColors", [])],
            rare_colors=[ColorCluster(**item) for item in data.get("rareColors", [])],
            sprite_palette_bgr=[tuple(int(v) for v in item) for item in data.get("spritePaletteBgr", [])],
            hsv_histogram=[float(v) for v in data.get("hsvHistogram", [])],
            rgb_histogram=[float(v) for v in data.get("rgbHistogram", [])],
            patch_signatures=[PatchSignature(**item) for item in data.get("patchSignatures", [])],
            template_count=int(data.get("templateCount", 0)),
            action_indices=[int(v) for v in data.get("actionIndices", [])],
        )

    @classmethod
    def load(cls, path: Path) -> "SimpleMobDescriptor":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return {
            "mobName": data["mob_name"],
            "version": data["version"],
            "size": data["size"],
            "bodyColors": data["body_colors"],
            "accentColors": data["accent_colors"],
            "rareColors": data["rare_colors"],
            "spritePaletteBgr": data["sprite_palette_bgr"],
            "hsvHistogram": data["hsv_histogram"],
            "rgbHistogram": data["rgb_histogram"],
            "patchSignatures": data["patch_signatures"],
            "templateCount": data["template_count"],
            "actionIndices": data["action_indices"],
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
