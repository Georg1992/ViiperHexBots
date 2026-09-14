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


def _color_structure_gate() -> Callable:
    from pybot.recognition.detector.detector import MobDetector
    return MobDetector._passes_color_structure_gate


def _noisy_extract() -> Callable:
    from pybot.recognition.detector.detector import MobDetector
    return MobDetector._noisy_extraction_signal


def _silhouette_gate() -> Callable:
    from pybot.recognition.detector.detector import MobDetector
    return MobDetector._evaluate_silhouette_gate


def _extract_body_gate() -> Callable:
    from pybot.recognition.detector.detector import MobDetector
    return MobDetector._passes_extract_body_gate



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


def _collect_silhouette() -> Callable:
    from pybot.recognition.detector.detector import MobDetector
    return MobDetector._collect_silhouette_candidates


def _collect_marker_square() -> Callable:
    from pybot.recognition.detector.detector import MobDetector
    return MobDetector._collect_marker_square_candidates


def _marker_square_gate() -> Callable:
    from pybot.recognition.detector.detector import MobDetector
    return MobDetector._passes_marker_square_gate


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
                    "use_sprite_grf",
                    "_collect_marker_square_candidates",
                    "_collect_silhouette_candidates",
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
            "body-cluster diversity (boost body+required groups; optional-group boost; press weak/mono)",
            "edge-density boost (skipped for GRF marker squares)",
            "GaussianBlur",
            "upscale",
        ),
        sources=(
            SourceCheck(
                _build_sprite_heatmap,
                (
                    "downscale",
                    "weighted_sprite_palette_heatmap",
                    "use_body_cluster_diversity",
                    "apply_body_cluster_diversity",
                    "marker_square",
                    "_finish_heatmap",
                ),
            ),
            SourceCheck(
                _finish_heatmap,
                (
                    "edge_boost",
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
        title="GRF marker square",
        items=(
            "sprite.grf hunts skip geometry, color-structure, and silhouette",
            "heat CC area vs descriptor square",
            "aspect near 1:1",
            "heatmap fill of the CC bbox",
        ),
        sources=(
            SourceCheck(
                _detect,
                (
                    "use_sprite_grf",
                    "_collect_marker_square_candidates",
                ),
            ),
            SourceCheck(
                _collect_marker_square,
                (
                    "_passes_marker_square_gate",
                ),
            ),
            SourceCheck(
                _marker_square_gate,
                (
                    "_MARKER_AREA_MIN_RATIO",
                    "_MARKER_AREA_MAX_RATIO",
                    "_MARKER_ASPECT_MIN",
                    "_MARKER_ASPECT_MAX",
                    "_MARKER_MIN_HEAT_FILL",
                    "peakRelativeThreshold",
                    "minCenterHeat",
                ),
            ),
        ),
    ),
    PipelineStage(        title="Geometry pre-gate",
        items=(
            "all peaks clear pre-gates (dedup is post-detection)",
            "heat area in [sil_frac/5, 2.0] vs descriptor sprite area",
            "heat aspect vs per-mob descriptor.min_aspect_ratio / max_aspect_ratio",
            "small heat CC: peak-relative heat >= 1.5 * peakRelativeThreshold",
        ),
        sources=(
            SourceCheck(
                _geometry_gate,
                (
                    "sil_frac",
                    "min_area_ratio",
                    "_GEOMETRY_AREA_MAX_RATIO",
                    "descriptor.min_aspect_ratio",
                    "descriptor.max_aspect_ratio",
                ),
            ),
            SourceCheck(
                _collect_silhouette,
                (
                    "_passes_discovery_geometry_gate",
                    "_is_small_heat_cc",
                    "peakRelativeThreshold",
                    "_SMALL_HEAT_RELATIVE_PEAK_MULT",
                ),
            ),


        ),
    ),
    PipelineStage(
        title="Color structure pre-gate",
        items=(
            "all peaks clear pre-gates (dedup is post-detection)",
            "heat-CC crop required palette groups present count",
            "second-largest required-group share among matched pixels",
            "required-group match coverage of crop pixels",
            "dominant+supporting mass body-cluster strong-match fraction",
            "reject when present < minRequiredPaletteGroups (fail-closed)",
            "reject when second_share < minSecondPaletteGroupShare (fail-closed)",
            "reject when coverage < descriptor.min_required_palette_coverage (fail-closed)",
            "reject when body_strong < descriptor.min_body_cluster_strong (fail-closed; full-res; desc-sized when heat area < 2*min_area_ratio)",

            "skip gate when descriptor has no required groups",
        ),
        sources=(
            SourceCheck(
                _color_structure_gate,
                (
                    "required_groups_structure",
                    "minRequiredPaletteGroups",
                    "minSecondPaletteGroupShare",
                    "min_required_palette_coverage",
                    "min_body_cluster_strong",
                    "match_palette_required_groups",
                    "max_sprite_palette_distance",
                ),
            ),
            SourceCheck(
                _collect_silhouette,
                ("_passes_color_structure_gate",),
            ),
        ),
    ),
    PipelineStage(
        title="Silhouette gate",
        items=(
            "search around heat CC bbox (not sprite-inflated)",
            "palette binary_raw + dilate(1) -> CC overlapping heat",
            "horizontal MORPH_CLOSE bridge (silhouetteHorizontalBridgeCells)",
            "pre-shrink extract: same min-area + aspect band as heat (fail-closed)",
            "if extract_area_ratio >= 2: shrink to descriptor window on body centroid",
            "if soft/hard >= 2 and cand0 recall >= mode recall: deform best ref into heat within 2 silhouette cells",
            "tight bridged crop resized to descriptor size",
            "candidate_silhouette vs descriptor masks",
            "dual gate recall AND precision",
            "reject solid-fill hard occupancy (>=95% of gate grid)",
            "new peaks: extract body_strong >= 0.5 * descriptor.min_body_cluster_strong",
            "noisy extract flags (bloated / soft-hard) on SilhouetteCheck",
            "pass / fail per blob",
        ),
        sources=(
            SourceCheck(
                _collect_silhouette,
                (
                    "_evaluate_silhouette_gate",
                    "_passes_extract_body_gate",
                    "_noisy_extraction_signal",
                ),
            ),
            SourceCheck(
                _silhouette_gate,
                (
                    "search_region",
                    "sprite_palette_heatmap",
                    "binary_raw",
                    "minSpritePaletteMatch",
                    "dilate",
                    "_passes_size_aspect_vs_descriptor",
                    "_shrink_bloated_extract_to_descriptor",
                    "extract_bbox",
                    "candidate_silhouette",
                    "best_silhouette_match",
                    "silhouette_gate_thresholds",
                    "_SOLID_FILL_HARD_FRACTION",
                    "_maybe_deform_noisy_candidate",
                ),
            ),
            SourceCheck(
                _extract_body_gate,
                (
                    "min_body_cluster_strong",
                    "_EXTRACT_BODY_STRONG_FLOOR_FRAC",
                    "required_groups_structure",
                    "extract_bbox",
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
                    "silhouette_gate_thresholds",
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
