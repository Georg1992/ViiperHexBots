"""Round-robin attack target selection."""

from __future__ import annotations

from pybot.runtime._mob_rec_path import import_hunt_track_rules
_hunt = import_hunt_track_rules()
MobTrack = _hunt.MobTrack
select_target_id = _hunt.select_target_id


class HuntPolicy:
    def __init__(self) -> None:
        self._last_attack_target_id = 0

    def reset(self) -> None:
        self._last_attack_target_id = 0

    def note_attack_target(self, track_id: int) -> None:
        self._last_attack_target_id = track_id

    def select_target(self, tracks: list[MobTrack], now_tick: int) -> int:
        target_id = select_target_id(
            tracks,
            now_tick,
            last_attack_target_id=self._last_attack_target_id,
        )
        return target_id

    @property
    def last_attack_target_id(self) -> int:
        return self._last_attack_target_id
