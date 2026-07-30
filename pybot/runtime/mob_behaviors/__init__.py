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

    def is_surrounded(
        self,
        char_x: int,
        char_y: int,
        all_mobs: list[tuple[int, int]],
    ) -> tuple[bool, str]:
        """Check whether mobs box the character in from opposite sides.

        Called before each attack.  Returns ``(in_danger, reason)``.
        Default: never surrounded.
        """
        return False, ""

    def execute_danger_teleport(
        self,
        ctx,
        input_backend: InputBackend,
        *,
        reason: str = "",
    ) -> None:
        """Press teleport key, wait, then clear tracks immediately.

        Called by :class:`DangerDetector` only — no other module touches
        teleport for danger.  Tracks are cleared right after the teleport
        settles so stale mob positions are never used post-teleport.

        *ctx* must provide ``config``, ``logger``, ``stop_event``, and
        ``area_reset``.
        """
        tp_scan = ctx.config.teleport_scan_code or ctx.config.creamy_tp_scan_code
        prefix = f"{reason} " if reason else ""
        ctx.logger.behavior(
            f"[DANGER] {prefix}teleport_scan={tp_scan} — teleporting"
        )
        try:
            input_backend.teleport_key(tp_scan)
        except Exception as exc:
            ctx.logger.behavior(f"[DANGER] teleport input error: {exc}")
        ctx.stop_event.wait(ctx.config.teleport_duration_ms / 1000.0)
        ctx.area_reset(reason="danger_teleport")


class AnubisBehavior(MobBehavior):
    """Anubis kites: walk away from the mob after each attack."""

    def has_custom_behavior(self) -> bool:
        return True

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

    def is_surrounded(
        self,
        char_x: int,
        char_y: int,
        all_mobs: list[tuple[int, int]],
    ) -> tuple[bool, str]:
        """Anubis is surrounded when mobs exist on opposite sides.

        Left+Right or Above+Below — no safe kite direction exists.
        """
        if len(all_mobs) < 2:
            return False, ""
        left = any(mx < char_x for mx, _my in all_mobs)
        right = any(mx > char_x for mx, _my in all_mobs)
        above = any(my < char_y for _mx, my in all_mobs)
        below = any(my > char_y for _mx, my in all_mobs)
        if left and right:
            return True, "left+right"
        if above and below:
            return True, "above+below"
        return False, ""


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
