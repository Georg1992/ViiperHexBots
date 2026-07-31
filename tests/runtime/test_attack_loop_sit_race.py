"""Attack loop must not continue after sit claims the combat gate mid-delay."""

from __future__ import annotations

import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from pybot.game_state import PlayerVitals
from pybot.runtime.workers.attack_loop import AttackLoop


class AttackLoopSitRaceTests(unittest.TestCase):
    def test_attack_skips_kite_when_sit_blocks_combat_during_skill_delay(self) -> None:
        ctx = MagicMock()
        ctx.config = SimpleNamespace(skill_scan_code=16, skill_delay_ms=50)
        ctx.stop_event = threading.Event()
        ctx.should_run_combat.side_effect = [True, False]
        snap = SimpleNamespace(x=10, y=20)
        ctx.tracks.snapshot_for_track.return_value = snap
        ctx.character_screen_pos.return_value = (0, 0)
        ctx.logger = MagicMock()

        input_backend = MagicMock()
        mob_behavior = MagicMock()
        vitals = PlayerVitals()
        vitals.publish_sp(100, 200)
        loop = AttackLoop(
            ctx,
            MagicMock(),
            input_backend,
            mob_behavior=mob_behavior,
            vitals=vitals,
        )

        # First should_run_combat is unused by _attack_one; gate flips during delay.
        ctx.should_run_combat.side_effect = None
        ctx.should_run_combat.return_value = False
        loop._attack_one(1, 1)

        input_backend.skill_click.assert_called_once()
        mob_behavior.kite_after_attack.assert_not_called()


if __name__ == "__main__":
    unittest.main()
