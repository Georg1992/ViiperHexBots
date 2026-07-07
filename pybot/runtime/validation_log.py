"""Structured hunt validation logs"""

from __future__ import annotations

from pybot.runtime.hunt_tracks import HuntTracks, MobTrackSnapshot, monotonic_ms
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
        self._state_tick_seq = 0

    def _emit(self, event: str, **fields: object) -> None:
        if not self._enabled:
            return
        parts = " ".join(f"{key}={value}" for key, value in fields.items())
        self._logger.behavior(f"[VAL] {event} {parts} tick={monotonic_ms()}")

    def log_area_reset(self, reason: str = "area_reset") -> None:
        self._state_tick_seq = 0
        self._emit("area_reset", screenId=self._tracks.area_epoch, reason=reason)

    def log_discovery_scan(
        self,
        *,
        raw_count: int,
        filtered_count: int,
        added_count: int,
        duration_ms: int,
        summary,
    ) -> None:
        fields = {
            "screenId": self._tracks.area_epoch,
            "rawCount": raw_count,
            "filteredCount": filtered_count,
            "addedCount": added_count,
            "durationMs": duration_ms,
        }
        if summary is not None:
            fields.update(
                {
                    "matchedCount": summary.matched_count,
                    "removedCount": summary.removed_count,
                    "matchedIds": _join_ids(summary.matched_ids),
                    "createdIds": _join_ids(summary.created_ids),
                    "removedIds": _join_ids(summary.removed_ids),
                    "tracksBefore": summary.tracks_before,
                    "tracksAfter": summary.tracks_after,
                    "aliveOrPendingAfter": summary.alive_or_pending_after,
                }
            )
        self._emit("discovery_scan", **fields)

    def log_state_tick(
        self,
        *,
        requested_ids: list[int],
        observation_count: int,
        duration_ms: int,
        every_n: int = 1,
    ) -> None:
        if not self._enabled:
            return
        self._state_tick_seq += 1
        if every_n > 1 and self._state_tick_seq % every_n != 0:
            return
        self._emit(
            "state_tick",
            screenId=self._tracks.area_epoch,
            seq=self._state_tick_seq,
            requestedIds=_join_ids(requested_ids),
            observationCount=observation_count,
            durationMs=duration_ms,
        )

    def log_state_observation(
        self,
        *,
        track_id: int,
        obs_state: str,
        old_snap: MobTrackSnapshot | None,
        obs_x: int,
        obs_y: int,
        duration_ms: int = 0,
        coord_updates: int = 0,
    ) -> None:
        track = self._tracks.get_track_by_id(track_id)
        old_state = old_snap.state if old_snap else "?"
        old_x = old_snap.x if old_snap else "?"
        old_y = old_snap.y if old_snap else "?"
        new_state = track.state if track else "removed"
        new_x = track.x if track else obs_x
        new_y = track.y if track else obs_y
        self._emit(
            "state_obs",
            screenId=self._tracks.area_epoch,
            id=track_id,
            obsState=obs_state,
            oldState=old_state,
            newState=new_state,
            oldX=old_x,
            oldY=old_y,
            newX=new_x,
            newY=new_y,
            durationMs=duration_ms,
            coordUpdates=coord_updates,
        )

    def log_track_batch(
        self,
        *,
        track_ids: list[int],
        found_count: int,
        duration_ms: int,
        needs_confirm_ids: list[int],
        every_n: int = 1,
        coord_updates: int = 0,
        results_detail: str = "",
    ) -> None:
        if not self._enabled:
            return
        self._state_tick_seq += 1
        if every_n > 1 and self._state_tick_seq % every_n != 0:
            return
        fields = {
            "screenId": self._tracks.area_epoch,
            "seq": self._state_tick_seq,
            "trackIds": _join_ids(track_ids),
            "foundCount": found_count,
            "coordUpdates": coord_updates,
            "durationMs": duration_ms,
            "confirmIds": _join_ids(needs_confirm_ids),
        }
        if results_detail:
            fields["results"] = results_detail
        self._emit("track_batch", **fields)

    def log_local_track_observation(
        self,
        *,
        track_id: int,
        found: bool,
        old_snap: MobTrackSnapshot | None,
        x: int,
        y: int,
        miss_reason: str = "",
    ) -> None:
        track = self._tracks.get_track_by_id(track_id)
        old_x = old_snap.x if old_snap else "?"
        old_y = old_snap.y if old_snap else "?"
        new_x = track.x if track else x
        new_y = track.y if track else y
        self._emit(
            "local_track",
            screenId=self._tracks.area_epoch,
            id=track_id,
            found=int(found),
            missReason=miss_reason or "-",
            oldX=old_x,
            oldY=old_y,
            newX=new_x,
            newY=new_y,
        )

    def log_attack_decision(
        self,
        track_id: int,
        block_reason: str,
        *,
        x: int = 0,
        y: int = 0,
        coord_age_ms: int = 0,
        attack_count: int = 0,
        state: str = "",
    ) -> None:
        self._emit(
            "attack_decision",
            screenId=self._tracks.area_epoch,
            trackId=track_id,
            blockReason=block_reason or "-",
            x=x,
            y=y,
            coordAgeMs=coord_age_ms,
            attackCount=attack_count,
            state=state,
        )

    def log_attack_engage(
        self,
        track_id: int,
        x: int,
        y: int,
        *,
        coord_age_ms: int,
        attack_count: int,
    ) -> None:
        self._emit(
            "attack_engage",
            screenId=self._tracks.area_epoch,
            trackId=track_id,
            x=x,
            y=y,
            coordAgeMs=coord_age_ms,
            attackCount=attack_count,
        )

    def log_no_target_decision(
        self,
        decision: str,
        reason: str,
        *,
        attackable_count: int,
        alive_or_pending_count: int,
        area_clear: bool,
        vision_busy: bool,
        direct_state_pending: bool,
        has_discovery_since_reset: bool,
    ) -> None:
        self._emit(
            "no_target",
            screenId=self._tracks.area_epoch,
            decision=decision,
            reason=reason,
            attackableCount=attackable_count,
            aliveOrPendingCount=alive_or_pending_count,
            areaClear=int(area_clear),
            visionBusy=int(vision_busy),
            directStatePending=int(direct_state_pending),
            hasDiscoverySinceReset=int(has_discovery_since_reset),
        )
