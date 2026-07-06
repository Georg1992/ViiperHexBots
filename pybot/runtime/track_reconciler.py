"""Stateless discovery reconciliation service.

Extracted from :class:`pybot.runtime.hunt_tracks.HuntTracks` to satisfy
the Single Responsibility Principle.  HuntTracks remains the thread-safe
store; this class handles the pure logic of matching discovery
detections to existing tracks, creating new tracks, and removing stale
unmatched tracks.
"""

from __future__ import annotations

from collections.abc import Callable

from pybot.runtime._mob_rec_path import import_hunt_track_rules
_hunt = import_hunt_track_rules()

DiscoveryDetection = _hunt.DiscoveryDetection
MobTrack = _hunt.MobTrack
ReconcileSummary = _hunt.ReconcileSummary
cluster_living_detections = _hunt.cluster_living_detections
find_nearest_track_for_detection = _hunt.find_nearest_track_for_detection
is_alive = _hunt.is_alive
reconcile_detection = _hunt.reconcile_detection
reconcile_unmatched_tracks = _hunt.reconcile_unmatched_tracks


class TrackReconciler:
    """Stateless service that reconciles discovery detections with a track list.

    Mutates the track list in-place: matches existing tracks, creates
    new ones via *create_track_fn*, and removes stale unmatched tracks.
    """

    @staticmethod
    def reconcile(
        tracks: list[MobTrack],
        detections: list[DiscoveryDetection],
        *,
        mob_name: str,
        now_tick: int,
        create_track_fn: Callable[..., MobTrack],
    ) -> ReconcileSummary:
        """Reconcile discovery detections with existing tracks.

        Args:
            tracks: Current track list (mutated in-place).
            detections: Raw discovery detections.
            mob_name: Mob name for new tracks.
            now_tick: Current monotonic tick.
            create_track_fn: Callable with signature
                ``(mob_name, x, y, confidence, candidate_scale, tick) -> MobTrack``
                that creates and appends a new track to the list.

        Returns:
            ReconcileSummary with match/create/remove statistics.
        """
        tracks_before = len(tracks)
        alive_before = sum(1 for t in tracks if is_alive(t))

        matched_ids: list[int] = []
        created_ids: list[int] = []
        removed_ids: list[int] = []
        matched_track_ids: set[int] = set()
        added_count = 0

        clustered = cluster_living_detections(detections)
        for detection in clustered:
            match = find_nearest_track_for_detection(
                tracks,
                detection.x,
                detection.y,
                now_tick,
                matched_track_ids,
            )
            if match is not None:
                matched_track_ids.add(match.id)
                matched_ids.append(match.id)
                reconcile_detection(match, detection, now_tick=now_tick)
                continue

            new_track = create_track_fn(
                mob_name,
                detection.x,
                detection.y,
                detection.confidence,
                detection.candidate_scale,
                now_tick,
            )
            matched_track_ids.add(new_track.id)
            created_ids.append(new_track.id)
            added_count += 1

        for track_id in reconcile_unmatched_tracks(tracks, matched_track_ids):
            removed_ids.append(track_id)
        if removed_ids:
            TrackReconciler._remove_by_ids(tracks, set(removed_ids))

        return ReconcileSummary(
            tracks_before=tracks_before,
            tracks_after=len(tracks),
            alive_before=alive_before,
            alive_or_pending_after=sum(
                1 for t in tracks if is_alive(t)
            ),
            matched_ids=matched_ids,
            created_ids=created_ids,
            removed_ids=removed_ids,
            matched_count=len(matched_ids),
            removed_count=len(removed_ids),
            added_count=added_count,
        )


    @staticmethod
    def _remove_by_ids(
        tracks: list[MobTrack],
        remove_ids: set[int],
    ) -> None:
        """Remove tracks with the given IDs from the list (mutates in-place)."""
        if not remove_ids:
            return
        tracks[:] = [t for t in tracks if t.id not in remove_ids]
