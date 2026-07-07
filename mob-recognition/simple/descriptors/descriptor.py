"""Serializable descriptor for the simple heatmap detector."""

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
class DeadStateProfile:
    size: SizeDescriptor
    body_colors: list[ColorCluster]
    accent_colors: list[ColorCluster]
    hsv_histogram: list[float]
    sprite_palette_bgr: list[tuple[int, int, int]] | None = None


@dataclass
class SimpleMobDescriptor:
    mob_name: str
    version: int
    size: SizeDescriptor
    dominant_color: ColorCluster
    supporting_colors: list[ColorCluster]
    accent_colors: list[ColorCluster]
    rare_colors: list[ColorCluster]
    sprite_palette_bgr: list[tuple[int, int, int]]
    hsv_histogram: list[float]
    dead: DeadStateProfile | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SimpleMobDescriptor":
        dead_data = data.get("dead")
        dead_profile = None
        if dead_data:
            sprite_bgr = dead_data.get("spritePaletteBgr")
            dead_profile = DeadStateProfile(
                size=SizeDescriptor.from_dict(dead_data["size"]),
                body_colors=[ColorCluster(**item) for item in dead_data["bodyColors"]],
                accent_colors=[ColorCluster(**item) for item in dead_data["accentColors"]],
                hsv_histogram=[float(v) for v in dead_data["hsvHistogram"]],
                sprite_palette_bgr=[tuple(int(v) for v in item) for item in sprite_bgr] if sprite_bgr else None,
            )

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

        return cls(
            mob_name=str(data["mobName"]),
            version=int(data["version"]),
            size=SizeDescriptor.from_dict(data["size"]),
            dominant_color=dominant_color,
            supporting_colors=supporting_colors,
            accent_colors=[ColorCluster(**item) for item in data["accentColors"]],
            rare_colors=[ColorCluster(**item) for item in data["rareColors"]],
            sprite_palette_bgr=[tuple(int(v) for v in item) for item in data["spritePaletteBgr"]],
            hsv_histogram=[float(v) for v in data["hsvHistogram"]],
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
            "dominantColor": data["dominant_color"],
            "supportingColors": data["supporting_colors"],
            "accentColors": data["accent_colors"],
            "rareColors": data["rare_colors"],
            "spritePaletteBgr": data["sprite_palette_bgr"],
            "hsvHistogram": data["hsv_histogram"],
        }
        if data["dead"] is not None:
            dead = data["dead"]
            payload["dead"] = {
                "size": dead["size"],
                "bodyColors": dead["body_colors"],
                "accentColors": dead["accent_colors"],
                "hsvHistogram": dead["hsv_histogram"],
            }
            if dead.get("sprite_palette_bgr"):
                payload["dead"]["spritePaletteBgr"] = dead["sprite_palette_bgr"]
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

    def dead_scoring_view(self) -> "SimpleMobDescriptor":
        if self.dead is None:
            raise ValueError(f"descriptor for mob '{self.mob_name}' has no dead profile")
        dead = self.dead
        # Map dead body_colors into dominant + supporting layout
        dc = dead.body_colors[0] if dead.body_colors else self.dominant_color
        sc = dead.body_colors[1:] if len(dead.body_colors) > 1 else []
        return SimpleMobDescriptor(
            mob_name=self.mob_name,
            version=self.version,
            size=dead.size,
            dominant_color=dc,
            supporting_colors=sc,
            accent_colors=dead.accent_colors,
            rare_colors=self.rare_colors,
            sprite_palette_bgr=dead.sprite_palette_bgr if dead.sprite_palette_bgr else self.sprite_palette_bgr,
            hsv_histogram=dead.hsv_histogram,
            dead=None,
        )
