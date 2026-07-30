"""Mob-specific custom behaviors injected into the hunt loop.

Each mob can have a :class:`MobBehavior` subclass that hooks into key
points of the attack cycle without modifying the core workers.

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

    def has_custom_behavior(self) -> bool:
        """Whether this mob has any non-default hooks (used by the GUI)."""
        return False

    def kite_after_attack(
        self,
        mob_x: int,
        mob_y: int,
        char_x: int,
        char_y: int,
        input_backend: InputBackend,
    ) -> bool:
        """Move the character away from the mob after attacking.

        Called *after* the skill delay wait.  Default is a no-op.
        Return ``True`` when the backend was instructed to click away.
        """
        return False


class AnubisBehavior(MobBehavior):
    """Anubis kites: walk away from the mob after each attack."""

    def has_custom_behavior(self) -> bool:
        return True

    def kite_after_attack(
        self,
        mob_x: int,
        mob_y: int,
        char_x: int,
        char_y: int,
        input_backend: InputBackend,
    ) -> bool:
        # Vector from mob → character; continue past character to move away.
        dx = char_x - mob_x
        dy = char_y - mob_y
        kite_x = char_x + dx
        kite_y = char_y + dy
        input_backend.move_mouse(kite_x, kite_y)
        return input_backend.left_click()


# ── Registry ──────────────────────────────────────────────────────
# Map lower-case descriptor names to behavior instances.
# Add new mobs here as their custom behaviors are implemented.

_BEHAVIOR_REGISTRY: dict[str, MobBehavior] = {
    "anubis": AnubisBehavior(),
}


def get_mob_behavior(mob_name: str) -> MobBehavior:
    """Return the :class:`MobBehavior` for *mob_name*, or the default no-op."""
    key = mob_name.strip().lower()
    return _BEHAVIOR_REGISTRY.get(key, MobBehavior())
