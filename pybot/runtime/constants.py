"""Hunt timing constants"""

from __future__ import annotations

from pybot.runtime._mob_rec_path import import_hunt_track_rules
_hunt = import_hunt_track_rules()

HUNT_OBJECT_RADIUS = _hunt.HUNT_OBJECT_RADIUS
HUNT_TRACK_LOST_LIMIT = _hunt.HUNT_TRACK_LOST_LIMIT

HUNT_DISCOVERY_INTERVAL_MS = 3000
HUNT_TELEPORT_DURATION_MS = 1000
CELL_SIZE_PX = 64
DEFAULT_SEARCH_RANGE_CELLS = 16

__all__ = [
    "HUNT_DISCOVERY_INTERVAL_MS",
    "HUNT_OBJECT_RADIUS",
    "HUNT_TELEPORT_DURATION_MS",
    "HUNT_TRACK_LOST_LIMIT",
    "CELL_SIZE_PX",
    "DEFAULT_SEARCH_RANGE_CELLS",
]
