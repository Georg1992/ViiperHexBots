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


def _geometry_gate() -> Callable:
    from pybot.recognition.detector.detector import MobDetector
    return MobDetector._passes_discovery_geometry_gate


def _noisy_extract() -> Callable:
    from pybot.recognition.detector.detector import MobDetector
    return MobDetector._noisy_extraction_signal


def _silhouette_gate() -> Callable:
    from pybot.recognition.detector.detector import MobDetector
    return MobDetector._evaluate_silhouette_gate


def _silhouette_search() -> Callable:
    from pybot.recognition.detector.detector import MobDetector
    return MobDetector._silhouette_search_window


def _palette_cc() -> Callable:
    from pybot.recognition.detector.detector import MobDetector
    return MobDetector._best_overlapping_palette_component


def _horizontal_bridge() -> Callable:
    from pybot.recognition.detector.detector import MobDetector
    return MobDetector._horizontal_bridge_occupancy


def _maybe_deform() -> Callable:
    from pybot.recognition.detector.detector import MobDetector
    return MobDetector._maybe_deform_noisy_candidate


def _finalize_accepted() -> Callable:
    from pybot.recognition.detector.detector import MobDetector
    return MobDetector._finalize_accepted


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
                    "_passes_discovery_geometry_gate",
                    "_evaluate_silhouette_gate",
                    "_noisy_extraction_signal",
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
            "upscale",
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
            "dedup nearby peaks by sprite size",
        ),
        sources=(
            SourceCheck(
                _top_centers,
                (
                    "peak_relative_threshold",
                    "min_center_heat",
                    "connectedComponentsWithStats",
                    "_blob_from_mask",
                    "_dedup_blobs_by_sprite_size",
                    "avg_width",
                    "avg_height",
                ),
            ),
        ),
    ),
    PipelineStage(
        title="Geometry pre-gate",
        items=(
            "geometry pre-gate (area + aspect vs descriptor)",
        ),
        sources=(
            SourceCheck(
                _geometry_gate,
                (
                    "sil_frac",
                    "min_area_ratio",
                    "aspect_band",
                ),
            ),
        ),
    ),
    PipelineStage(
        title="Silhouette gate",
        items=(
            "search around heat CC bbox (not sprite-inflated)",
            "palette binary_raw + dilate(1) -> CC overlapping heat",
            "horizontal MORPH_CLOSE bridge (silhouetteHorizontalBridgeCells)",
            "if extract_area_ratio >= 2: shrink to descriptor window on body centroid",
            "if soft/hard >= 2 and cand0 recall >= minSilhouetteRecall: deform best ref into heat within 2 silhouette cells",
            "tight bridged crop resized to descriptor size",
            "candidate_silhouette vs descriptor masks",
            "noisy extract flag (bloated crop and/or soft/hard content noise)",
            "pass / fail per blob",
        ),
        sources=(
            SourceCheck(
                _silhouette_gate,
                (
                    "search_region",
                    "sprite_palette_heatmap",
                    "binary_raw",
                    "minSpritePaletteMatch",
                    "dilate",
                    "_shrink_bloated_extract_to_descriptor",
                    "extract_bbox",
                    "candidate_silhouette",
                    "best_silhouette_match",
                    "minSilhouetteRecall",
                    "minSilhouettePrecision",
                    "_maybe_deform_noisy_candidate",
                ),
            ),
            SourceCheck(
                _silhouette_search,
                ("search_region", "comp_bbox", "desc_w", "desc_h"),
            ),
            SourceCheck(
                _palette_cc,
                ("connectedComponentsWithStats", "best_overlap", "best_label"),
            ),
            SourceCheck(
                _horizontal_bridge,
                (
                    "silhouetteHorizontalBridgeCells",
                    "MORPH_CLOSE",
                    "bridge_px",
                ),
            ),
            SourceCheck(
                _maybe_deform,
                (
                    "_occupancy_soft_hard_ratio",
                    "_CONTENT_NOISE_SOFT_HARD_RATIO",
                    "minSilhouetteRecall",
                    "_deform_silhouette_occupancy",
                    "candidate_silhouette",
                ),
            ),
            SourceCheck(
                _noisy_extract,
                (
                    "extract_area_ratio",
                    "extract_bloated",
                    "content_noisy",
                    "soft_hard_ratio",
                    "noisy_extract",
                ),
            ),
        ),
    ),
    PipelineStage(
        title="Accept",
        items=(
            "sort accepted by heat score",
            "final accepted set",
        ),
        sources=(
            SourceCheck(_finalize_accepted, ("accepted.sort",)),
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
