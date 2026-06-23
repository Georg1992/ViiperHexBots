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
class DeadStateProfile:
    size: SizeDescriptor
    body_colors: list[ColorCluster]
    accent_colors: list[ColorCluster]
    hsv_histogram: list[float]
    rgb_histogram: list[float]
    patch_signatures: list[PatchSignature] = field(default_factory=list)
    action_indices: list[int] = field(default_factory=list)
    corpse_frame_start: int = 0


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
    dead: DeadStateProfile | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SimpleMobDescriptor":
        dead_data = data.get("dead")
        dead_profile = None
        if dead_data:
            dead_profile = DeadStateProfile(
                size=SizeDescriptor(**dead_data["size"]),
                body_colors=[ColorCluster(**item) for item in dead_data["bodyColors"]],
                accent_colors=[ColorCluster(**item) for item in dead_data["accentColors"]],
                hsv_histogram=[float(v) for v in dead_data["hsvHistogram"]],
                rgb_histogram=[float(v) for v in dead_data["rgbHistogram"]],
                patch_signatures=[PatchSignature(**item) for item in dead_data["patchSignatures"]],
                action_indices=[int(v) for v in dead_data["actionIndices"]],
                corpse_frame_start=int(dead_data.get("corpseFrameStart", 0)),
            )
        return cls(
            mob_name=str(data["mobName"]),
            version=int(data["version"]),
            size=SizeDescriptor(**data["size"]),
            body_colors=[ColorCluster(**item) for item in data["bodyColors"]],
            accent_colors=[ColorCluster(**item) for item in data["accentColors"]],
            rare_colors=[ColorCluster(**item) for item in data["rareColors"]],
            sprite_palette_bgr=[tuple(int(v) for v in item) for item in data["spritePaletteBgr"]],
            hsv_histogram=[float(v) for v in data["hsvHistogram"]],
            rgb_histogram=[float(v) for v in data["rgbHistogram"]],
            patch_signatures=[PatchSignature(**item) for item in data["patchSignatures"]],
            template_count=int(data["templateCount"]),
            action_indices=[int(v) for v in data["actionIndices"]],
            dead=dead_profile,
        )

    @classmethod
    def load(cls, path: Path) -> "SimpleMobDescriptor":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        payload = {
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
        if data["dead"] is not None:
            dead = data["dead"]
            payload["dead"] = {
                "size": dead["size"],
                "bodyColors": dead["body_colors"],
                "accentColors": dead["accent_colors"],
                "hsvHistogram": dead["hsv_histogram"],
                "rgbHistogram": dead["rgb_histogram"],
                "patchSignatures": dead["patch_signatures"],
                "actionIndices": dead["action_indices"],
                "corpseFrameStart": dead["corpse_frame_start"],
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

    def dead_scoring_view(self) -> "SimpleMobDescriptor":
        if self.dead is None:
            raise ValueError(f"descriptor for mob '{self.mob_name}' has no dead profile")
        return SimpleMobDescriptor(
            mob_name=self.mob_name,
            version=self.version,
            size=self.dead.size,
            body_colors=self.dead.body_colors,
            accent_colors=self.dead.accent_colors,
            rare_colors=self.rare_colors,
            sprite_palette_bgr=self.sprite_palette_bgr,
            hsv_histogram=self.dead.hsv_histogram,
            rgb_histogram=self.dead.rgb_histogram,
            patch_signatures=self.dead.patch_signatures,
            template_count=self.template_count,
            action_indices=self.dead.action_indices,
            dead=None,
        )
