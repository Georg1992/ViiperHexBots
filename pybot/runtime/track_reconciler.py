"""Discovery reconciliation — create new tracks; list off-screen absences.

Discovery finds NEW mobs. Position updates still belong to tracking
(LocalTracker); this service never moves existing tracks.

Dedup uses ``existing_positions`` — known-object (x, y) at frame-capture time
(alive tracks plus recent removal sites). Absence uses
``existing_track_positions`` — (track_id, x, y) for alive tracks at that same
instant. A detection within one object radius of a known position (or of a
track just created earlier in this same scan) is skipped; only genuinely new
detections create a track. Alive tracks with no matching detection are listed
in ``removed_ids``; the caller decides which of those actually left the hunt
ROI and may be dropped.
"""

from __future__ import annotations

from collections.abc import Callable

from pybot.recognition.rules import (
    DiscoveryDetection,
    MobTrack,
    ReconcileSummary,
    cluster_living_detections,
    detection_matches_existing,
    is_alive,
)
from pybot.recognition.detector.detector import load_detector_config


class TrackReconciler:
    """Stateless service that adds new-mob tracks and lists absent track ids."""

    @staticmethod
    def reconcile(
        tracks: list[MobTrack],
        detections: list[DiscoveryDetection],
        existing_positions: list[tuple[int, int]],
        *,
        mob_name: str,
        now_tick: int,
        create_track_fn: Callable[..., MobTrack],
        detector_config: dict | None = None,
        existing_track_positions: list[tuple[int, int, int]] | None = None,
    ) -> ReconcileSummary:
        """Create tracks for new detections; list tracks missing from this scan.

        Args:
            tracks: Current track list (appended to in-place via create_track_fn).
            detections: Raw discovery detections.
            existing_positions: (x, y) of known objects at frame-capture time.
            mob_name: Mob name for new tracks.
            now_tick: Current monotonic tick.
            create_track_fn: Callable with signature
                ``(mob_name, x, y, confidence, candidate_scale, tick) -> MobTrack``
                that creates and appends a new track to the list.
            existing_track_positions: (track_id, x, y) for alive tracks at
                frame-capture time. When omitted, derived from ``tracks``.

        Returns:
            ReconcileSummary with created/matched/absent statistics.
        """
        tracks_before = len(tracks)
        alive_before = sum(1 for t in tracks if is_alive(t))

        track_positions = (
            list(existing_track_positions)
            if existing_track_positions is not None
            else [(t.id, t.x, t.y) for t in tracks if is_alive(t)]
        )
        unmatched_ids = {track_id for track_id, _x, _y in track_positions}

        # Working set of "known" positions: seeded with frame-time known
        # objects, extended with each track created in this scan so two
        # detections of one new mob don't both spawn a track.
        known_positions: list[tuple[int, int]] = list(existing_positions)

        matched_count = 0
        created_ids: list[int] = []

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
                track_positions,
                unmatched_ids,
                radius_sq=radius_sq,
            )
            if matched_tid is not None:
                unmatched_ids.discard(matched_tid)
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

            new_track = create_track_fn(
                mob_name,
                detection.x,
                detection.y,
                detection.confidence,
                detection.candidate_scale,
                now_tick,
            )
            created_ids.append(new_track.id)
            known_positions.append((detection.x, detection.y))

        removed_ids = sorted(unmatched_ids)
        return ReconcileSummary(
            tracks_before=tracks_before,
            tracks_after=len(tracks),
            alive_before=alive_before,
            alive_after=sum(1 for t in tracks if is_alive(t)),
            created_ids=created_ids,
            removed_ids=removed_ids,
            matched_count=matched_count,
            added_count=len(created_ids),
            removed_count=len(removed_ids),
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
        for track_id, px, py in track_positions:
            if track_id not in unmatched_ids:
                continue
            dx = x - px
            dy = y - py
            dist = (dx * dx) + (dy * dy)
            if dist <= radius_sq and dist < best_dist:
                best_dist = dist
                best_id = track_id
        return best_id
