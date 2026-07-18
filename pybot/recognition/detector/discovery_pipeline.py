"""Canonical discovery pipeline structure.

Single source of truth for:
  - debug_vis pipeline.txt
  - drift checks against production detector source

When you change MobDetector / HeatmapDetector pipeline code, update the
matching stage items and source_markers here. Viz and tests will fail if
markers no longer appear in production source.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class SourceCheck:
    resolve: Callable[[], Callable]
    markers: tuple[str, ...]


@dataclass(frozen=True)
class PipelineStage:
    title: str
    items: tuple[str, ...]
    sources: tuple[SourceCheck, ...]


def _detect() -> Callable:
    from pybot.recognition.detector.detector import MobDetector
    return MobDetector.detect


def _build_sprite_heatmap() -> Callable:
    from pybot.recognition.detector.scoring.heatmap_detector import HeatmapDetector
    return HeatmapDetector.build_sprite_heatmap


def _finish_heatmap() -> Callable:
    from pybot.recognition.detector.scoring.heatmap_detector import HeatmapDetector
    return HeatmapDetector._finish_heatmap


def _top_centers() -> Callable:
    from pybot.recognition.detector.scoring.heatmap_detector import HeatmapDetector
    return HeatmapDetector.top_centers


def _silhouette_gate() -> Callable:
    from pybot.recognition.detector.detector import MobDetector
    return MobDetector._evaluate_silhouette_gate


def _finalize_accepted() -> Callable:
    from pybot.recognition.detector.detector import MobDetector
    return MobDetector._finalize_accepted


def _nms() -> Callable:
    from pybot.recognition.detector.detector import MobDetector
    return MobDetector._nms


DISCOVERY_PIPELINE: tuple[PipelineStage, ...] = (
    PipelineStage(
        title="Descriptor",
        items=(
            "ensure_descriptor(mob_name)",
        ),
        sources=(
            SourceCheck(
                _detect,
                (
                    "ensure_descriptor",
                    "build_sprite_heatmap",
                    "top_centers",
                    "_evaluate_silhouette_gate",
                    "_finalize_accepted",
                ),
            ),
        ),
    ),
    PipelineStage(
        title="Sprite heatmap",
        items=(
            "downscale frame when large enough",
            "weighted_sprite_palette_heatmap",
            "optional palette diversity",
            "edge-density boost",
            "GaussianBlur",
            "upscale + local peak boost",
        ),
        sources=(
            SourceCheck(
                _build_sprite_heatmap,
                (
                    "downscale",
                    "weighted_sprite_palette_heatmap",
                    "use_palette_diversity",
                    "_finish_heatmap",
                ),
            ),
            SourceCheck(
                _finish_heatmap,
                (
                    "edge_density",
                    "GaussianBlur",
                    "_local_peak_boost",
                    "_nearest_upscale",
                ),
            ),
        ),
    ),
    PipelineStage(
        title="Blob centers",
        items=(
            "threshold heatmap",
            "connectedComponentsWithStats",
            "peak-weighted centers",
            "component bbox per blob",
        ),
        sources=(
            SourceCheck(
                _top_centers,
                (
                    "peak_relative_threshold",
                    "min_center_heat",
                    "connectedComponentsWithStats",
                    "CC_STAT_LEFT",
                    "np.average",
                ),
            ),
        ),
    ),
    PipelineStage(
        title="Silhouette gate",
        items=(
            "search around blob / component bbox",
            "sprite_palette_heatmap on search region",
            "binary threshold + dilate -> CC component",
            "resize component to descriptor avg size",
            "candidate_silhouette vs descriptor masks",
            "pass / fail per blob",
        ),
        sources=(
            SourceCheck(
                _silhouette_gate,
                (
                    "search_region",
                    "sprite_palette_heatmap",
                    "minSpritePaletteMatch",
                    "dilate",
                    "connectedComponentsWithStats",
                    "candidate_silhouette",
                    "best_silhouette_similarity",
                    "minSilhouetteSimilarity",
                ),
            ),
        ),
    ),
    PipelineStage(
        title="NMS / accept",
        items=(
            "sort accepted by heat score",
            "suppress nearby passes via nmsDistancePx",
            "final accepted set",
        ),
        sources=(
            SourceCheck(_finalize_accepted, ("accepted.sort", "_nms")),
            SourceCheck(_nms, ("nmsDistancePx",)),
        ),
    ),
)


def format_discovery_pipeline_text() -> str:
    lines = ["Discovery pipeline", ""]
    for index, stage in enumerate(DISCOVERY_PIPELINE, start=1):
        lines.append(f"{index}. {stage.title}")
        for item_index, item in enumerate(stage.items):
            letter = chr(ord("a") + item_index)
            lines.append(f"   {letter}) {item}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def assert_discovery_pipeline_matches_source() -> None:
    """Fail if documented stages no longer match production function source."""
    for stage in DISCOVERY_PIPELINE:
        for check in stage.sources:
            fn = check.resolve()
            source = inspect.getsource(fn)
            missing = [marker for marker in check.markers if marker not in source]
            if missing:
                raise AssertionError(
                    f"discovery pipeline stage {stage.title!r} is out of date for "
                    f"{fn.__qualname__}: missing markers {missing}"
                )
