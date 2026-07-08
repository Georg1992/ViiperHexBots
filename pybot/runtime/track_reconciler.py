"""Discovery reconciliation service — create-only.

Discovery has exactly one job: find NEW mobs. Position updates and
liveness/removal belong to tracking (LocalTracker), which runs on its own
thread and keeps track coordinates fresh. So this service never mutates or
removes existing tracks.

Dedup is done against ``existing_positions`` — the (x, y) of known objects
sampled at the instant the discovery frame was captured. That keeps detections
and the positions they are compared against on one time reference even though
tracking is concurrently moving the live tracks. A detection within one object
radius of a known position (or of a track just created earlier in this same
scan) is skipped; only genuinely new detections create a track.
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
    """Stateless service that adds new-mob tracks from discovery detections."""

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
    ) -> ReconcileSummary:
        """Create tracks for detections that don't belong to an existing object.

        Args:
            tracks: Current track list (appended to in-place via create_track_fn).
            detections: Raw discovery detections.
            existing_positions: (x, y) of known objects at frame-capture time.
            mob_name: Mob name for new tracks.
            now_tick: Current monotonic tick.
            create_track_fn: Callable with signature
                ``(mob_name, x, y, confidence, candidate_scale, tick) -> MobTrack``
                that creates and appends a new track to the list.

        Returns:
            ReconcileSummary with created/matched statistics (never removes).
        """
        tracks_before = len(tracks)
        alive_before = sum(1 for t in tracks if is_alive(t))

        # Working set of "known" positions: seeded with frame-time track
        # positions, extended with each track created in this scan so two
        # detections of one new mob don't both spawn a track.
        known_positions: list[tuple[int, int]] = list(existing_positions)

        matched_count = 0
        created_ids: list[int] = []

        config = detector_config or load_detector_config()
        cluster_radius = int(config["discoveryClusterRadiusPx"])
        dedup_radius = int(config["trackDedupRadiusPx"])

        clustered = cluster_living_detections(
            detections,
            cluster_radius=cluster_radius,
        )
        for detection in clustered:
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

        return ReconcileSummary(
            tracks_before=tracks_before,
            tracks_after=len(tracks),
            alive_before=alive_before,
            alive_after=sum(1 for t in tracks if is_alive(t)),
            created_ids=created_ids,
            matched_count=matched_count,
            added_count=len(created_ids),
        )
