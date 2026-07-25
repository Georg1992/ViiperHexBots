"""Discovery reconciliation — matches detections to existing tracks; lists
new-mob candidates for tracking to create.

Discovery finds NEW mobs and publishes candidate positions.  Tracking owns
all track creation — it ingests candidates on the next fresh frame, runs a
local-follow search to get exact coordinates, and creates the track there.

Dedup uses ``existing_positions`` — known-object (x, y) at frame-capture time
(alive tracks plus recent removal sites).  Absence uses
``existing_track_positions`` — (track_id, x, y) for alive tracks at that same
instant.  A detection within one object radius of a known position is matched
(not a new candidate).  Alive tracks with no matching detection are listed
in ``removed_ids``; the caller increments their ``discovery_miss_count``.
"""

from __future__ import annotations

from dataclasses import dataclass

from pybot.recognition.rules import (
    DiscoveryDetection,
    cluster_living_detections,
    detection_matches_existing,
)
from pybot.recognition.detector.detector import load_detector_config


@dataclass
class DiscoveryReconcileResult:
    """Output of discovery reconciliation.

    new_candidates: detections that did not match any existing track —
        tracking should create tracks for these on the next fresh frame.
    matched_ids: track IDs that were matched by a detection.
    removed_ids: track IDs that had no matching detection (absent).
    matched_count: number of detections that matched an existing track.
    """
    new_candidates: list[DiscoveryDetection]
    matched_ids: list[int]
    removed_ids: list[int]
    matched_count: int


class TrackReconciler:
    """Stateless service that matches detections to tracks and lists absences."""

    @staticmethod
    def match_and_absent(
        detections: list[DiscoveryDetection],
        existing_positions: list[tuple[int, int]],
        existing_track_positions: list[tuple[int, int, int]],
        *,
        detector_config: dict | None = None,
    ) -> DiscoveryReconcileResult:
        """Match detections to existing tracks; return new-candidate detections.

        Does NOT create tracks — that is tracking's job on a fresh frame.
        Does NOT mutate any track fields — the caller applies match/absence.

        Args:
            detections: Raw discovery detections.
            existing_positions: (x, y) of known objects at frame-capture time.
            existing_track_positions: (track_id, x, y) for alive tracks at
                frame-capture time — required, always provided by caller.
            detector_config: Optional detector config dict (loaded from disk
                when omitted).

        Returns:
            DiscoveryReconcileResult with new_candidates, matched_ids, removed_ids.
        """
        unmatched_ids = {entry[0] for entry in existing_track_positions}

        # Working set of "known" positions: seeded with frame-time known
        # objects, extended with each candidate already published in this
        # scan so two detections of one new mob don't both become candidates.
        known_positions: list[tuple[int, int]] = list(existing_positions)

        matched_ids: list[int] = []
        matched_count = 0
        new_candidates: list[DiscoveryDetection] = []

        config = detector_config or load_detector_config()
        cluster_radius = int(config["discoveryClusterRadiusPx"])
        dedup_radius = int(config["trackDedupRadiusPx"])
        radius_sq = dedup_radius * dedup_radius

        clustered = cluster_living_detections(
            detections,
            cluster_radius=cluster_radius,
        )
        for detection in clustered:
            matched_tid = TrackReconciler._match_track_id(
                detection.x,
                detection.y,
                existing_track_positions,
                unmatched_ids,
                radius_sq=radius_sq,
            )
            if matched_tid is not None:
                unmatched_ids.discard(matched_tid)
                matched_ids.append(matched_tid)
                matched_count += 1
                continue

            if detection_matches_existing(
                detection.x,
                detection.y,
                known_positions,
                dedup_radius=dedup_radius,
            ):
                matched_count += 1
                continue

            new_candidates.append(detection)
            known_positions.append((detection.x, detection.y))

        removed_ids = sorted(unmatched_ids)
        return DiscoveryReconcileResult(
            new_candidates=new_candidates,
            matched_ids=matched_ids,
            removed_ids=removed_ids,
            matched_count=matched_count,
        )

    @staticmethod
    def _match_track_id(
        x: int,
        y: int,
        track_positions: list[tuple[int, int, int]],
        unmatched_ids: set[int],
        *,
        radius_sq: int,
    ) -> int | None:
        """Nearest unmatched capture-time track within dedup radius, if any."""
        best_id: int | None = None
        best_dist = radius_sq + 1
        for entry in track_positions:
            track_id, px, py = entry[0], entry[1], entry[2]
            if track_id not in unmatched_ids:
                continue
            dx = x - px
            dy = y - py
            dist = (dx * dx) + (dy * dy)
            if dist <= radius_sq and dist < best_dist:
                best_dist = dist
                best_id = track_id
        return best_id
