"""Mob-specific custom behaviors injected into the hunt loop.

Each mob can have a :class:`MobBehavior` subclass that hooks into key
points of the attack cycle without modifying the core workers.

Only mobs listed in ``_BEHAVIOR_REGISTRY`` have custom behavior. Today
that is Anubis alone (kiting).

Kiting
------
After each attack, move the cursor in the opposite direction from the
mob and left-click — the character walks away, avoiding melee hits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from pybot.game_state import PlayerVitals
from pybot.runtime.config import CustomBehaviorRuntime
from pybot.runtime.hunt_tracks import monotonic_ms

if TYPE_CHECKING:
    from pybot.runtime.danger_detector import DangerDetector
    from pybot.runtime.input.input_backend import InputBackend


def kite_away_from_mobs(
    char_x: int,
    char_y: int,
    input_backend: InputBackend,
    *,
    all_mobs: list[tuple[int, int]],
) -> bool:
    """Walk away from the average position of all currently tracked mobs."""
    if not all_mobs:
        return False
    n = len(all_mobs)
    center_x = sum(mx for mx, _my in all_mobs) // n
    center_y = sum(my for _mx, my in all_mobs) // n
    dx = char_x - center_x
    dy = char_y - center_y
    if dx == 0 and dy == 0:
        return False
    return input_backend.move_and_click(char_x + dx, char_y + dy)


class MobBehavior:
    """Default no-op behavior for mobs without custom logic."""

    def prepare_target(
        self,
        target_id: int,
        target_x: int,
        target_y: int,
        input_backend: InputBackend,
        *,
        target_debuffed: bool,
        mark_debuffed: Callable[[], bool],
    ) -> bool:
        """Prepare a target before its first attack."""
        del target_id, target_x, target_y, input_backend, target_debuffed, mark_debuffed
        return True

    def before_attack(
        self,
        char_x: int,
        char_y: int,
        input_backend: InputBackend,
        *,
        all_mobs: list[tuple[int, int]],
    ) -> bool:
        """Run configured pre-attack actions; default is a no-op."""
        return False

    def kite_after_attack(
        self,
        char_x: int,
        char_y: int,
        input_backend: InputBackend,
        *,
        all_mobs: list[tuple[int, int]],
    ) -> bool:
        """Move the character away from nearby mobs after attacking.

        Called *after* the skill delay wait.  *all_mobs* contains the
        screen positions of every currently tracked mob.

        Default is a no-op.
        Return ``True`` when the backend was instructed to click away.
        """
        return False


class AnubisBehavior(MobBehavior):
    """Anubis kites: walk away from the mob after each attack."""

    def kite_after_attack(
        self,
        char_x: int,
        char_y: int,
        input_backend: InputBackend,
        *,
        all_mobs: list[tuple[int, int]],
    ) -> bool:
        if not all_mobs:
            return False

        # Preserve the legacy Anubis input sequence (move, then click).
        n = len(all_mobs)
        center_x = sum(mx for mx, _my in all_mobs) // n
        center_y = sum(my for _mx, my in all_mobs) // n
        dx = char_x - center_x
        dy = char_y - center_y
        if dx == 0 and dy == 0:
            return False
        return input_backend.move_and_click(char_x + dx, char_y + dy)


class ConfiguredMobBehavior(MobBehavior):
    """Generic per-mob cycle: debuff, safe self-heal, kite, then attack."""

    def __init__(
        self,
        settings: CustomBehaviorRuntime,
        vitals: PlayerVitals,
        danger: DangerDetector,
        legacy_behavior: MobBehavior | None = None,
    ) -> None:
        self._settings = settings
        self._vitals = vitals
        self._danger = danger
        self._legacy_behavior = legacy_behavior
        self._last_kite_ms = 0

    def prepare_target(
        self,
        target_id: int,
        target_x: int,
        target_y: int,
        input_backend: InputBackend,
        *,
        target_debuffed: bool,
        mark_debuffed: Callable[[], bool],
    ) -> bool:
        del target_id
        if getattr(self._settings, "debuff_scan_code", 0) <= 0 or target_debuffed:
            return True
        if not input_backend.skill_click_at(
            self._settings.debuff_scan_code, target_x, target_y
        ):
            return False
        return mark_debuffed()

    def before_attack(
        self,
        char_x: int,
        char_y: int,
        input_backend: InputBackend,
        *,
        all_mobs: list[tuple[int, int]],
    ) -> bool:
        del all_mobs
        hp, hp_max = self._vitals.hp_pair()
        if (
            self._settings.heal_scan_code > 0
            and hp is not None
            and hp_max is not None
            and hp < hp_max
            and self._danger.is_safe_for_heal()
        ):
            return input_backend.skill_click_at(
                self._settings.heal_scan_code, char_x, char_y
            )
        return False

    def kite_after_attack(
        self,
        char_x: int,
        char_y: int,
        input_backend: InputBackend,
        *,
        all_mobs: list[tuple[int, int]],
    ) -> bool:
        """Kite in the inter-attack window, before the next heal/attack."""
        now = monotonic_ms()
        if self._settings.kiting_tick_ms > 0:
            if now - self._last_kite_ms < self._settings.kiting_tick_ms:
                return False
            acted = kite_away_from_mobs(
                char_x, char_y, input_backend, all_mobs=all_mobs
            )
            if acted:
                self._last_kite_ms = now
            return acted
        if self._legacy_behavior is not None:
            # Keep Anubis's original post-attack kite timing when its cog has
            # no custom kite interval configured.
            return self._legacy_behavior.kite_after_attack(
                char_x, char_y, input_backend, all_mobs=all_mobs
            )
        return False


# ── Registry ──────────────────────────────────────────────────────
# Only Anubis has custom hunt behavior. Do not register other mobs
# here unless they intentionally get custom hooks + UI marking.

_BEHAVIOR_REGISTRY: dict[str, MobBehavior] = {
    "anubis": AnubisBehavior(),
}


def get_configured_mob_behavior(
    settings: CustomBehaviorRuntime,
    vitals: PlayerVitals,
    danger: DangerDetector,
    *,
    legacy_behavior: MobBehavior | None = None,
) -> MobBehavior:
    """Build the generic configured behavior for one mob."""
    return ConfiguredMobBehavior(
        settings, vitals, danger, legacy_behavior=legacy_behavior
    )


def get_mob_behavior(mob_name: str) -> MobBehavior:
    """Return the :class:`MobBehavior` for *mob_name*, or the default no-op."""
    key = mob_name.strip().lower()
    return _BEHAVIOR_REGISTRY.get(key, MobBehavior())


def mob_has_custom_behavior(mob_name: str) -> bool:
    """True only for mobs with a registered custom hunt behavior."""
    return mob_name.strip().lower() in _BEHAVIOR_REGISTRY
