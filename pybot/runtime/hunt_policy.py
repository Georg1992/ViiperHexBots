"""Round-robin attack target selection."""

from __future__ import annotations

from pybot.recognition.rules import MobTrack, select_target_id


class HuntPolicy:
    def __init__(self) -> None:
        self._last_attack_target_id = 0
        self._max_attacks: int | None = None

    def reset(self) -> None:
        self._last_attack_target_id = 0
        self._max_attacks = None

    def note_attack_target(self, track_id: int) -> None:
        self._last_attack_target_id = track_id

    def select_target(self, tracks: list[MobTrack], now_tick: int) -> int:
        return select_target_id(
            tracks,
            now_tick,
            last_attack_target_id=self._last_attack_target_id,
            max_attacks=self._max_attacks,
        )

    def set_max_attacks(self, max_attacks: int) -> None:
        self._max_attacks = max_attacks

    @property
    def last_attack_target_id(self) -> int:
        return self._last_attack_target_id
