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

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pybot.runtime.input.input_backend import InputBackend


class MobBehavior:
    """Default no-op behavior for mobs without custom logic."""

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

        # Average position of all mobs → vector away from their center.
        n = len(all_mobs)
        center_x = sum(mx for mx, _my in all_mobs) // n
        center_y = sum(_my for _mx, my in all_mobs) // n

        dx = char_x - center_x
        dy = char_y - center_y
        kite_x = char_x + dx
        kite_y = char_y + dy
        input_backend.move_mouse(kite_x, kite_y)
        return input_backend.left_click()


# ── Registry ──────────────────────────────────────────────────────
# Only Anubis has custom hunt behavior. Do not register other mobs
# here unless they intentionally get custom hooks + UI marking.

_BEHAVIOR_REGISTRY: dict[str, MobBehavior] = {
    "anubis": AnubisBehavior(),
}


def get_mob_behavior(mob_name: str) -> MobBehavior:
    """Return the :class:`MobBehavior` for *mob_name*, or the default no-op."""
    key = mob_name.strip().lower()
    return _BEHAVIOR_REGISTRY.get(key, MobBehavior())


def mob_has_custom_behavior(mob_name: str) -> bool:
    """True only for mobs with a registered custom hunt behavior."""
    return mob_name.strip().lower() in _BEHAVIOR_REGISTRY
