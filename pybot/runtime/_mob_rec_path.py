"""Add mob-recognition to sys.path and provide clean imports for hunt_track_rules.

Module-level side-effect
    Importing this module adds ``mob-recognition/`` and
    ``mob-recognition/simple/`` to ``sys.path`` so that bare
    imports like ``from capture import capture_region`` work.

Preferred usage
    Instead of relying on the side-effect, call
    :func:`import_hunt_track_rules` from the modules that need it::

        from pybot.runtime._mob_rec_path import import_hunt_track_rules
        hunt_rules = import_hunt_track_rules()
        HUNT_TRACK_MISS_LIMIT = hunt_rules.HUNT_TRACK_MISS_LIMIT

    This makes the dependency explicit and avoids ``# noqa: F401`` /
    ``# noqa: E402`` suppression comments.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MOB_RECOGNITION_DIR = PROJECT_ROOT / "mob-recognition"
MOB_RECOGNITION_SIMPLE_DIR = MOB_RECOGNITION_DIR / "simple"

for path in (MOB_RECOGNITION_DIR, MOB_RECOGNITION_SIMPLE_DIR):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)


def import_hunt_track_rules():
    """Import and return the ``hunt_track_rules`` module.

    ``sys.path`` has already been set up by the module-level code above,
    so ``hunt_track_rules`` (which lives in ``mob-recognition/simple/``)
    is importable.
    """
    return importlib.import_module("hunt_track_rules")
