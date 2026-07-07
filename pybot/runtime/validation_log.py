"""Structured hunt validation logs"""

from __future__ import annotations

from pybot.runtime.hunt_tracks import HuntTracks, monotonic_ms
from pybot.runtime.logging import HuntLogger


def _join_ids(ids: list[int] | None) -> str:
    if not ids:
        return "-"
    return ",".join(str(track_id) for track_id in ids)


class HuntValidationLogger:
    def __init__(self, logger: HuntLogger, tracks: HuntTracks, *, enabled: bool = True) -> None:
        self._logger = logger
        self._tracks = tracks
        self._enabled = enabled

    def _emit(self, event: str, **fields: object) -> None:
        if not self._enabled:
            return
        parts = " ".join(f"{key}={value}" for key, value in fields.items())
        self._logger.behavior(f"[VAL] {event} {parts} tick={monotonic_ms()}")

    def log_area_reset(self, reason: str = "area_reset") -> None:
        self._emit("area_reset", screenId=self._tracks.area_epoch, reason=reason)

    def log_discovery_scan(
        self,
        *,
        raw_count: int,
        filtered_count: int,
        duration_ms: int,
        summary,
    ) -> None:
        self._emit(
            "discovery_scan",
            screenId=self._tracks.area_epoch,
            rawCount=raw_count,
            filteredCount=filtered_count,
            durationMs=duration_ms,
            addedCount=summary.added_count,
            matchedCount=summary.matched_count,
            createdIds=_join_ids(summary.created_ids),
            tracksBefore=summary.tracks_before,
            tracksAfter=summary.tracks_after,
            aliveAfter=summary.alive_after,
        )

    def log_no_target_decision(
        self,
        decision: str,
        reason: str,
        *,
        alive_count: int,
        area_clear: bool,
        has_discovery_since_reset: bool,
    ) -> None:
        self._emit(
            "no_target",
            screenId=self._tracks.area_epoch,
            decision=decision,
            reason=reason,
            aliveCount=alive_count,
            areaClear=int(area_clear),
            hasDiscoverySinceReset=int(has_discovery_since_reset),
        )
