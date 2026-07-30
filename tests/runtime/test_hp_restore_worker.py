"""HP Restore worker — press HP key when HP falls below threshold."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from pybot.game_state import PlayerVitals
from pybot.runtime.constants import HP_RESTORE_RATIO
from pybot.runtime.runtime_context import HuntRuntimeContext
from pybot.runtime.workers.hp_restore_worker import HpRestoreWorker


class HpRestoreWorkerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = MagicMock()
        self.config.hp_button = "f1"
        self.config.hp_scan_code = 59
        self.config.heal_skill = False
        self.ctx = HuntRuntimeContext(
            config=self.config,
            logger=MagicMock(),
            tracks=MagicMock(),
            policy=MagicMock(),
            capture=MagicMock(),
            detector=MagicMock(),
            tracker=MagicMock(),
            validation=MagicMock(),
            control=MagicMock(),
            overlay=MagicMock(),
        )
        self.ctx.capture.is_valid.return_value = True
        self.input = MagicMock()
        self.vitals = PlayerVitals()

    def test_skips_when_no_hp_key(self) -> None:
        self.config.hp_scan_code = 0
        worker = HpRestoreWorker(self.ctx, self.input, self.vitals)
        worker.run()
        self.input.teleport_key.assert_not_called()

    def test_presses_when_hp_below_threshold(self) -> None:
        self.vitals.publish_hp(40, 100)
        worker = HpRestoreWorker(self.ctx, self.input, self.vitals)

        def stop_after_press(*_a, **_k):
            self.ctx.stop_event.set()
            return True

        self.input.teleport_key.side_effect = stop_after_press
        worker.run()
        self.input.teleport_key.assert_called_with(59)
        self.assertLess(40 / 100, HP_RESTORE_RATIO)

    def test_no_press_when_hp_ok(self) -> None:
        self.vitals.publish_hp(80, 100)
        worker = HpRestoreWorker(self.ctx, self.input, self.vitals)

        def stop_soon(*_a, **_k):
            self.ctx.stop_event.set()
            return False

        self.ctx.stop_event.wait = stop_soon  # type: ignore[method-assign]
        worker.run()
        self.input.teleport_key.assert_not_called()


if __name__ == "__main__":
    unittest.main()
