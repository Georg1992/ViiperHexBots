"""Pure post-attack observation rules.

This module classifies the evidence available after a skill press. It does
not read vitals, write tracks, or log; callers own those side effects.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from pybot.runtime.constants import SP_IDLE_MAX_OBSERVATION_AGE_MS


class CombatObservation(str, Enum):
    """Classification of one skill press based on SP observations."""

    HIT = "hit"
    IDLE = "idle"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class CombatObservationResult:
    """Classification plus a diagnostic reason for inconclusive evidence."""

    outcome: CombatObservation
    reason: str = ""

    @property
    def was_idle(self) -> bool | None:
        """Compatibility value consumed by the track policy."""
        if self.outcome is CombatObservation.IDLE:
            return True
        if self.outcome is CombatObservation.HIT:
            return False
        return None


class CombatObserver:
    """Classify SP evidence without depending on runtime state or logging."""

    def __init__(self, *, max_observation_age_ms: int = SP_IDLE_MAX_OBSERVATION_AGE_MS) -> None:
        self._max_observation_age_ms = max(0, int(max_observation_age_ms))

    def classify_sp(
        self,
        *,
        pre_sp: int | None,
        post_sp: int | None,
        pre_observed_ms: int,
        post_observed_ms: int,
        pre_changed_ms: int,
        post_changed_ms: int,
        sample_now_ms: int,
    ) -> CombatObservationResult:
        """Classify a skill press as hit, idle, or unknown.

        Equal, fresh SP with no intervening value change is considered idle.
        Missing, stale, reversed, increased, or transiently changed values are
        deliberately inconclusive so track counters cannot be corrupted.
        """
        if pre_sp is None or post_sp is None or post_observed_ms <= pre_observed_ms:
            reason = "sp-unread" if pre_sp is None or post_sp is None else "vitals-stale"
            return CombatObservationResult(CombatObservation.UNKNOWN, reason)

        if post_sp < pre_sp:
            return CombatObservationResult(CombatObservation.HIT)

        if post_sp > pre_sp:
            return CombatObservationResult(CombatObservation.UNKNOWN, "sp-increased")

        if post_changed_ms > pre_changed_ms:
            return CombatObservationResult(CombatObservation.UNKNOWN, "sp-changed-during-window")

        if sample_now_ms - post_observed_ms > self._max_observation_age_ms:
            return CombatObservationResult(CombatObservation.UNKNOWN, "obs-stale")

        return CombatObservationResult(CombatObservation.IDLE)


__all__ = [
    "CombatObservation",
    "CombatObservationResult",
    "CombatObserver",
]
