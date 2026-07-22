"""HP Restore worker — press HP key when vision HP is under threshold."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from pybot.recognition.ui.status_panel import StatusPanelValues
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
        self.ctx.capture.capture_client.return_value = MagicMock(size=1)
        self.input = MagicMock()

    def test_skips_when_no_hp_key(self) -> None:
        self.config.hp_scan_code = 0
        worker = HpRestoreWorker(self.ctx, self.input)
        worker.run()
        self.input.teleport_key.assert_not_called()

    @patch("pybot.runtime.workers.hp_restore_worker.read_status_panel")
    def test_presses_when_hp_below_threshold(self, read_hp: MagicMock) -> None:
        read_hp.return_value = StatusPanelValues(
            hp=40,
            hp_max=100,
            sp=50,
            sp_max=100,
            weight=10,
            weight_max=100,
            panel_origin=(0, 0),
        )
        worker = HpRestoreWorker(self.ctx, self.input)

        def stop_after_press(*_a, **_k):
            self.ctx.stop_event.set()
            return True

        self.input.teleport_key.side_effect = stop_after_press
        worker.run()
        self.input.teleport_key.assert_called_with(59)
        self.assertLess(40 / 100, HP_RESTORE_RATIO)

    @patch("pybot.runtime.workers.hp_restore_worker.read_status_panel")
    def test_no_press_when_hp_ok(self, read_hp: MagicMock) -> None:
        read_hp.return_value = StatusPanelValues(
            hp=80,
            hp_max=100,
            sp=50,
            sp_max=100,
            weight=10,
            weight_max=100,
            panel_origin=(0, 0),
        )
        worker = HpRestoreWorker(self.ctx, self.input)

        def stop_soon(*_a, **_k):
            self.ctx.stop_event.set()
            return False

        self.ctx.stop_event.wait = stop_soon  # type: ignore[method-assign]
        worker.run()
        self.input.teleport_key.assert_not_called()


if __name__ == "__main__":
    unittest.main()
