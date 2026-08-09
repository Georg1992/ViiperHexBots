"""Runtime composition ownership invariants."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from pybot.runtime.danger_detector import DangerDetector
from pybot.runtime.hunt_runtime import _build_core_workers
from pybot.runtime.workers.attack_loop import AttackLoop


class RuntimeCompositionTests(unittest.TestCase):
    def test_core_worker_list_contains_observers_only(self) -> None:
        """Conditional actions are advanced by GameplayLoop, not own threads."""
        ctx = MagicMock()
        ctx.capture.get_hunt_roi.return_value = None
        hunt_mode = MagicMock()
        input_backend = MagicMock()
        teleport = MagicMock()
        vitals = MagicMock()
        mob_behavior = MagicMock()
        danger = MagicMock(spec=DangerDetector)

        workers, attack = _build_core_workers(
            ctx,
            hunt_mode,
            input_backend,
            teleport,
            vitals,
            mob_behavior,
            danger,
        )

        self.assertEqual([name for name, _fn in workers], ["danger", "coord", "discovery"])
        self.assertIsInstance(attack, AttackLoop)
        self.assertNotIn("sit", [name for name, _fn in workers])
        self.assertNotIn("storage", [name for name, _fn in workers])
        self.assertNotIn("timers", [name for name, _fn in workers])
        self.assertNotIn("buffs", [name for name, _fn in workers])
        self.assertNotIn("hp_restore", [name for name, _fn in workers])


if __name__ == "__main__":
    unittest.main()
