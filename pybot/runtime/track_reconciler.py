"""Discovery reconciliation — create new tracks; list absences for tracking.

Discovery finds NEW mobs and publishes soft position priors on match.
Authoritative position updates and all track removal belong to tracking;
this service never overwrites existing track ``x``/``y`` and never deletes.

Dedup uses ``existing_positions`` — known-object (x, y) at frame-capture time
(alive tracks plus recent removal sites). Absence uses
``existing_track_positions`` — (track_id, x, y) for alive tracks at that same
instant. A detection within one object radius of a known position (or of a
track just created earlier in this same scan) is skipped; only genuinely new
detections create a track. Alive tracks with no matching detection are listed
in ``removed_ids``; the caller marks them ``discovery_absent`` (notification).

Tracks already flagged ``discovery_death`` are excluded from living match and
from absence listing — tracking owns their removal via the death flag.
"""

from __future__ import annotations

from collections.abc import Callable

from pybot.recognition.rules import (
    DiscoveryDetection,
    MobTrack,
    ReconcileSummary,
    apply_discovery_observation,
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
        tracks_by_id = {t.id: t for t in tracks if is_alive(t)}
        # Death-flagged tracks are tracking's responsibility; do not bind living
        # detections to them or mark them discovery_absent in this pass.
        death_flagged_ids = {
            tid for tid, track in tracks_by_id.items() if track.discovery_death
        }
        matchable_positions = [
            entry for entry in track_positions if entry[0] not in death_flagged_ids
        ]
        unmatched_ids = {entry[0] for entry in matchable_positions}
        death_capture_xy = {
            (int(entry[1]), int(entry[2]))
            for entry in track_positions
            if entry[0] in death_flagged_ids
        }

        # Working set of "known" positions: seeded with frame-time known
        # objects, extended with each track created in this scan so two
        # detections of one new mob don't both spawn a track. Death-flagged
        # capture sites are omitted so a nearby living peak can still create.
        known_positions: list[tuple[int, int]] = [
            (int(x), int(y))
            for x, y in existing_positions
            if (int(x), int(y)) not in death_capture_xy
        ]

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
                matchable_positions,
                unmatched_ids,
                radius_sq=radius_sq,
            )
            if matched_tid is not None:
                unmatched_ids.discard(matched_tid)
                matched_count += 1
                matched_track = tracks_by_id.get(matched_tid)
                if matched_track is not None:
                    apply_discovery_observation(
                        matched_track,
                        x=detection.x,
                        y=detection.y,
                        now_tick=now_tick,
                    )
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
