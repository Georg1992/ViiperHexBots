"""DangerDetector — isolated danger observer.

Runs in its own worker thread.  Polls ``PlayerVitals`` for HP and
reads ``CharacterState`` for surround detection.

Inputs:
* ``run()`` — polls ``hp_pair()`` from PlayerVitals; HP drops trigger
  teleport; drops at/below 50% of max are logged as critical
* ``run()`` (continued) — polls ``CharacterState.is_surrounded`` and
  teleports when the character is boxed in by mobs

Heal-skill reads ``has_nearby_threat`` / ``has_recent_damage`` to heal
only when safe. Teleport execution delegated to :class:`TeleportController`.
Danger teleport has priority over healing (allowed while heal ops are held).
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

from pybot.runtime.constants import WORKER_POLL_INTERVAL_S

if TYPE_CHECKING:
    from pybot.game_state import PlayerVitals
    from pybot.runtime.teleport import TeleportController
    from pybot.runtime.character_state import CharacterState

CRITICAL_HP_RATIO = 0.5


class DangerDetector:
    """Observes HP and CharacterState; triggers danger teleport automatically."""

    def __init__(
        self,
        ctx,
        teleport: TeleportController,
        character_state: CharacterState,
        vitals: PlayerVitals,
    ) -> None:
        self._ctx = ctx
        self._teleport = teleport
        self._character_state = character_state
        self._vitals = vitals
        self._prev_hp: int | None = None
        self._damage_lock = threading.Lock()
        self._damage_detected: bool = False
        self._last_damage_mono: float | None = None

    # ── Worker loop ───────────────────────────────────────────────

    def run(self) -> None:
        """Ongoing loop: poll HP + surround state, detect danger, sleep."""
        while not self._ctx.is_stopped():
            self._poll_hp()
            # Surround teleports while danger is allowed (incl. during heal).
            if self._ctx.should_allow_danger_teleport():
                self._poll_surround()
            self._ctx.stop_event.wait(WORKER_POLL_INTERVAL_S)

    def _poll_hp(self) -> None:
        """Read HP from shared vitals; record drops; teleport when allowed."""
        hp, hp_max = self._vitals.hp_pair()
        if hp is None:
            self._prev_hp = None
            return

        if self._prev_hp is not None and hp < self._prev_hp:
            is_critical = (
                hp_max is not None
                and hp_max > 0
                and hp / hp_max <= CRITICAL_HP_RATIO
            )
            with self._damage_lock:
                self._damage_detected = True
                self._last_damage_mono = time.monotonic()
            # Always record the drop (sit/heal read the latch). Danger TP
            # is allowed during heal; sit/storage/pause still block it.
            if self._ctx.should_allow_danger_teleport():
                reason = "critical_hp" if is_critical else "hp_drop"
                if self._teleport.danger_teleport(reason=reason):
                    # Only clear after a real escape — failed/missing key
                    # must keep blocking heal.
                    self._clear_damage_after_teleport()
        self._prev_hp = hp

    def _poll_surround(self) -> None:
        """Check CharacterState for surround danger and teleport."""
        if self._character_state.is_surrounded:
            reason = self._character_state.surrounded_reason
            if self._teleport.danger_teleport(reason=f"surrounded {reason}"):
                self._clear_damage_after_teleport()

    def _clear_damage_after_teleport(self) -> None:
        """Allow heal-skill to start after a successful danger TP."""
        with self._damage_lock:
            self._last_damage_mono = None
            self._damage_detected = False

    def is_surrounded(self) -> bool:
        """True when CharacterState reports the character is boxed in."""
        return bool(self._character_state.is_surrounded)

    def has_nearby_threat(self) -> bool:
        """True when mobs are near the character (tracks or visual blobs).

        Formal surround (opposite sides) is not required — any nearby mob
        means combat is unsafe for heal-skill.
        """
        state = self._character_state
        return (
            bool(state.is_surrounded)
            or int(state.nearby_mob_count) > 0
            or int(state.nearby_any_mobs_count) > 0
        )

    def has_recent_damage(self, within_s: float) -> bool:
        """True if damage latch is set or an HP drop was recent.

        The latch stays set until a successful danger teleport (or sit pop),
        so missing wing keys cannot open a heal window mid-fight.
        """
        with self._damage_lock:
            if self._damage_detected:
                return True
            if self._last_damage_mono is None:
                return False
            return (time.monotonic() - self._last_damage_mono) < within_s

    def pop_damage_detected(self) -> bool:
        """Return and clear the damage-detected flag for consumers (sit worker)."""
        with self._damage_lock:
            if self._damage_detected:
                self._damage_detected = False
                return True
            return False
