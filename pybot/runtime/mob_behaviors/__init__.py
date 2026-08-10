"""Mob-specific custom behaviors injected into the hunt loop.

Each mob can have a :class:`MobBehavior` subclass that hooks into key
points of the attack cycle without modifying the core workers.

Only mobs listed in ``_BEHAVIOR_REGISTRY`` have custom behavior. Today
that is Anubis alone (kiting).

Kiting
------
After each attack, move the cursor in the opposite direction from the
mob and double-click — the character walks away, avoiding melee hits.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Callable

from pybot.config.runtime import CustomBehaviorRuntime
from pybot.runtime.constants import CELL_SIZE_PX

DEFAULT_KITE_DISTANCE_PX = CELL_SIZE_PX * 5
_KITE_DIRECTIONS: tuple[tuple[int, int], ...] = (
    (0, -1),
    (1, -1),
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
)
from pybot.runtime.hunt_tracks import monotonic_ms

if TYPE_CHECKING:
    from pybot.runtime.input.input_backend import InputBackend


def _kite_direction(
    char_x: int,
    char_y: int,
    all_mobs: list[tuple[int, int]],
) -> tuple[float, float] | None:
    """Return a normalized direction toward the least occupied compass sector."""
    if not all_mobs:
        return None
    center_x = sum(mx for mx, _my in all_mobs) / len(all_mobs)
    center_y = sum(my for _mx, my in all_mobs) / len(all_mobs)
    away_x = char_x - center_x
    away_y = char_y - center_y
    distance = math.hypot(away_x, away_y)
    if distance > 0.0:
        return away_x / distance, away_y / distance

    # If the mob center overlaps the character, choose the direction with the
    # fewest mobs in its forward half-plane. The fixed compass order makes a
    # fully symmetric/centered frame deterministic rather than skipping kite.
    best_direction = _KITE_DIRECTIONS[0]
    best_score: tuple[int, float] | None = None
    for dx, dy in _KITE_DIRECTIONS:
        forward_distances = [
            math.hypot(mx - char_x, my - char_y)
            for mx, my in all_mobs
            if (mx - char_x) * dx + (my - char_y) * dy > 0
        ]
        score = (len(forward_distances), sum(forward_distances))
        if best_score is None or score < best_score:
            best_score = score
            best_direction = (dx, dy)
    direction_length = math.hypot(*best_direction)
    return best_direction[0] / direction_length, best_direction[1] / direction_length


def kite_away_from_mobs(
    char_x: int,
    char_y: int,
    input_backend: InputBackend,
    *,
    all_mobs: list[tuple[int, int]],
    distance_px: int = DEFAULT_KITE_DISTANCE_PX,
) -> bool:
    """Click a configured distance from the character toward open space."""
    direction = _kite_direction(char_x, char_y, all_mobs)
    if direction is None:
        return False
    distance_px = max(1, int(distance_px))
    target_x = round(char_x + direction[0] * distance_px)
    target_y = round(char_y + direction[1] * distance_px)
    # Keep the movement command atomic. The concrete backend emits both
    # clicks under its shared input lock and returns immediately after the
    # second release so the attack loop can continue without an extra wait.
    # Inspect the concrete type so MagicMock/lightweight doubles cannot
    # fabricate the new method and silently change their legacy call contract.
    double_click = getattr(type(input_backend), "move_and_double_click", None)
    if callable(double_click):
        return input_backend.move_and_double_click(target_x, target_y)
    # Narrow legacy test/custom backends may not expose the new primitive.
    return input_backend.move_and_click(target_x, target_y)


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

        return kite_away_from_mobs(
            char_x,
            char_y,
            input_backend,
            all_mobs=all_mobs,
            distance_px=DEFAULT_KITE_DISTANCE_PX,
        )


class ConfiguredMobBehavior(MobBehavior):
    """Generic per-mob cycle: debuff, kite, and attack.

    Character healing is owned by the synchronous hunt-loop recovery step;
    this behavior only handles target preparation and movement.
    """

    def __init__(
        self,
        settings: CustomBehaviorRuntime,
        legacy_behavior: MobBehavior | None = None,
    ) -> None:
        self._settings = settings
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
        if (
            getattr(self._settings, "debuff_scan_code", 0) <= 0
            or not str(getattr(self._settings, "debuff_button", "") or "").strip()
            or target_debuffed
        ):
            return True
        if not input_backend.skill_click_at(
            self._settings.debuff_scan_code,
            target_x,
            target_y,
            move_delay_s=0.0,
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
        # Character healing is handled by the hunt-loop recovery step; this
        # hook only handles target-independent behavior before an attack.
        del char_x, char_y, input_backend, all_mobs
        return False

    def kite_after_attack(
        self,
        char_x: int,
        char_y: int,
        input_backend: InputBackend,
        *,
        all_mobs: list[tuple[int, int]],
    ) -> bool:
        """Kite in the inter-attack window before the next attack."""
        now = monotonic_ms()
        if self._settings.kiting_tick_ms > 0:
            if now - self._last_kite_ms < self._settings.kiting_tick_ms:
                return False
            acted = kite_away_from_mobs(
                char_x,
                char_y,
                input_backend,
                all_mobs=all_mobs,
                distance_px=getattr(
                    self._settings,
                    "kite_distance_px",
                    DEFAULT_KITE_DISTANCE_PX,
                ),
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
    *,
    legacy_behavior: MobBehavior | None = None,
) -> MobBehavior:
    """Build the generic configured behavior for one mob."""
    return ConfiguredMobBehavior(settings, legacy_behavior=legacy_behavior)


def get_mob_behavior(mob_name: str) -> MobBehavior:
    """Return the registered behavior, or one explicit default no-op."""
    key = mob_name.strip().lower()
    behavior = _BEHAVIOR_REGISTRY.get(key)
    return MobBehavior() if behavior is None else behavior


def mob_has_custom_behavior(mob_name: str) -> bool:
    """True only for mobs with a registered custom hunt behavior."""
    return mob_name.strip().lower() in _BEHAVIOR_REGISTRY
