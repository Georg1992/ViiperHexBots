"""Corpse-heat death-site bookkeeping for hunt tracking.

A death site is a short-lived screen position used to absorb rediscovered
corpse heat.  The store deliberately knows nothing about tracks, discovery,
or threading; its owner controls synchronization around calls.
"""

from __future__ import annotations


class DeathSiteStore:
    """Track temporary death positions and absorb nearby corpse detections."""

    def __init__(self, *, radius_px: int, cooldown_ms: int) -> None:
        self._radius_px = max(0, int(radius_px))
        self._cooldown_ms = max(0, int(cooldown_ms))
        self._sites: list[tuple[int, int, int]] = []

    def clear(self) -> None:
        """Forget all sites, normally when entering a new hunt area."""
        self._sites.clear()

    def record(self, x: int, y: int, removed_tick: int) -> None:
        """Record a confirmed death position."""
        self._prune(removed_tick)
        self._sites.append((int(x), int(y), int(removed_tick)))

    def absorb_heat(self, x: int, y: int, now_tick: int) -> bool:
        """Refresh the nearest matching site and report whether heat was absorbed."""
        now_tick = int(now_tick)
        x = int(x)
        y = int(y)
        self._prune(now_tick)
        radius_sq = self._radius_px * self._radius_px
        best_index: int | None = None
        best_distance = 0
        for index, (site_x, site_y, _removed_tick) in enumerate(self._sites):
            dx = x - site_x
            dy = y - site_y
            distance = (dx * dx) + (dy * dy)
            if distance <= radius_sq and (
                best_index is None or distance < best_distance
            ):
                best_index = index
                best_distance = distance

        if best_index is None:
            return False
        self._sites[best_index] = (x, y, now_tick)
        return True

    def active_count(self, now_tick: int) -> int:
        """Return the number of non-expired sites at ``now_tick``."""
        self._prune(int(now_tick))
        return len(self._sites)

    def _prune(self, now_tick: int) -> None:
        self._sites = [
            (x, y, removed_tick)
            for x, y, removed_tick in self._sites
            if now_tick - removed_tick <= self._cooldown_ms
        ]


__all__ = ["DeathSiteStore"]
