"""ItemsToStorage / GetFlyWings worker — AHK call-sequence fidelity."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from pybot.app.process_memory import MemorySnapshot
from pybot.config.clients import MemoryAddresses
from pybot.runtime.constants import STORAGE_ENTER_SCAN_CODE
from pybot.runtime.input.input_backend import ShadowInputBackend
from pybot.runtime.runtime_context import HuntRuntimeContext
from pybot.runtime.workers.items_to_storage_worker import ItemsToStorageWorker
from pybot.viiper.keyboard import MOD_LEFT_ALT, vk_to_modifier


class _FakePoller:
    def __init__(self, weight: int, weight_max: int = 100) -> None:
        self.weight = weight
        self.weight_max = weight_max
        self.calls = 0

    def read(self, hwnd: int, addresses: MemoryAddresses) -> MemorySnapshot:
        del hwnd, addresses
        self.calls += 1
        return MemorySnapshot(
            weight=self.weight, weight_max=self.weight_max, ok=True
        )


class _RecordingInput(ShadowInputBackend):
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def move_mouse(self, x: int, y: int) -> bool:
        self.calls.append(("move", x, y))
        return True

    def left_click(self) -> bool:
        self.calls.append(("left_click",))
        return True

    def set_left_button(self, down: bool) -> bool:
        self.calls.append(("left_button", down))
        return True

    def alt_right_click(self) -> bool:
        self.calls.append(("alt_rmb",))
        return True

    def alt_right_clicks(self, times: int = 1) -> bool:
        for _ in range(times):
            if not self.alt_right_click():
                return False
        return True

    def key_tap(
        self,
        scan_code: int,
        *,
        press_s: float = 0.05,
        after_s: float = 0.30,
    ) -> bool:
        self.calls.append(("key_tap", scan_code, press_s, after_s))
        return True

    def type_text(self, text: str) -> bool:
        self.calls.append(("type", text))
        return True

    def toggle_inventory(self) -> bool:
        self.calls.append(("toggle_inv",))
        return True

    def play_key_chain(
        self, steps: tuple[tuple[str, int, int], ...]
    ) -> bool:
        self.calls.append(("play_chain", steps))
        return bool(steps)


class ItemsToStorageWorkerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = MagicMock()
        self.config.hwnd = 1
        self.config.open_storage_steps = (("f8", 66, 0),)
        self.config.weight_modifier = 80
        self.config.take_fly_wings = True
        self.config.fly_wings_amount = 100
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
        self.frame = np.zeros((240, 320, 3), dtype=np.uint8)
        self.ctx.capture.capture_client.return_value = self.frame
        self.ctx.capture.get_client_rect_screen.return_value = (100, 50, 800, 600)
        self.input = _RecordingInput()
        self.memory = MemoryAddresses(current_weight=1, max_weight=2)

    def _worker(self, poller: _FakePoller) -> ItemsToStorageWorker:
        return ItemsToStorageWorker(
            self.ctx, self.input, self.memory, poller=poller
        )

    def test_vk_menu_maps_to_left_alt(self) -> None:
        self.assertEqual(vk_to_modifier(0x12), MOD_LEFT_ALT)

    def test_exclusive_ops_block_should_run_workers(self) -> None:
        self.assertTrue(self.ctx.should_run_workers())
        self.assertTrue(self.ctx.begin_exclusive_ops())
        self.assertFalse(self.ctx.should_run_workers())
        self.assertFalse(self.ctx.try_begin_exclusive_ops())
        self.ctx.end_exclusive_ops()
        self.assertTrue(self.ctx.should_run_workers())

    def test_note_teleport_decrements_wingcount(self) -> None:
        self.ctx.wingcount = 3
        self.ctx.note_teleport_for_wings()
        self.assertEqual(self.ctx.wingcount, 2)
        self.config.open_storage_steps = ()
        self.ctx.note_teleport_for_wings()
        self.assertEqual(self.ctx.wingcount, 2)
        self.config.open_storage_steps = (("f8", 66, 0),)
        self.config.take_fly_wings = False
        self.ctx.note_teleport_for_wings()
        self.assertEqual(self.ctx.wingcount, 2)

    @patch("pybot.runtime.workers.items_to_storage_worker.time.sleep", return_value=None)
    @patch(
        "pybot.runtime.workers.items_to_storage_worker._cursor_pos",
        return_value=(150, 100),
    )
    @patch("pybot.runtime.workers.items_to_storage_worker.require_template")
    @patch("pybot.runtime.workers.items_to_storage_worker.cell_contains_template")
    @patch("pybot.runtime.workers.items_to_storage_worker.find_template")
    def test_items_to_storage_sequence(
        self,
        find_tpl: MagicMock,
        cell_tpl: MagicMock,
        require_tpl: MagicMock,
        _cursor: MagicMock,
        _sleep: MagicMock,
    ) -> None:
        require_tpl.side_effect = lambda _f, name, **_kw: {
            "use": (10, 10),
            "cell1": (20, 20),
            "eqp": (30, 10),
            "etc": (40, 10),
            "close": (50, 50),
            "wing": (60, 60),
        }[name]
        # empty after one deposit per tab; etc tab empty after ok+deposit
        empty_reads = iter([False, True, False, True, False, True])
        cell_tpl.side_effect = lambda _f, name, *_a, **_k: (
            next(empty_reads) if name == "empty_cell" else False
        )
        find_tpl.return_value = None

        worker = self._worker(_FakePoller(90))
        worker.items_to_storage()

        kinds = [c[0] for c in self.input.calls]
        self.assertEqual(kinds.count("toggle_inv"), 2)
        self.assertIn(("left_click",), self.input.calls)
        self.assertIn(("play_chain", (("f8", 66, 0),)), self.input.calls)
        self.assertGreaterEqual(kinds.count("alt_rmb"), 3)
        # USE / EQP / ETC tabs clicked
        self.assertEqual(kinds.count("left_click"), 4)  # 3 tabs + close

    @patch("pybot.runtime.workers.items_to_storage_worker.time.sleep", return_value=None)
    @patch(
        "pybot.runtime.workers.items_to_storage_worker._cursor_pos",
        return_value=(150, 100),
    )
    @patch("pybot.runtime.workers.items_to_storage_worker.require_template")
    @patch("pybot.runtime.workers.items_to_storage_worker.cell_contains_template")
    def test_items_to_storage_ok_dialog_uses_enter_284(
        self,
        cell_tpl: MagicMock,
        require_tpl: MagicMock,
        _cursor: MagicMock,
        _sleep: MagicMock,
    ) -> None:
        require_tpl.return_value = (10, 10)
        # USE empty immediately, EQP empty immediately, ETC: not empty then empty
        empty_seq = iter([True, True, False, True])
        cell_tpl.side_effect = lambda _f, name, *_a, **_k: (
            next(empty_seq) if name == "empty_cell" else False
        )

        with patch(
            "pybot.runtime.workers.items_to_storage_worker.find_template",
            side_effect=lambda _f, name, **_k: (1, 1) if name == "ok" else None,
        ):
            worker = self._worker(_FakePoller(90))
            worker.items_to_storage()

        self.assertIn(
            ("key_tap", STORAGE_ENTER_SCAN_CODE, 0.05, 0.0),
            self.input.calls,
        )

    @patch("pybot.runtime.workers.items_to_storage_worker.time.sleep", return_value=None)
    @patch(
        "pybot.runtime.workers.items_to_storage_worker._cursor_pos",
        return_value=(150, 100),
    )
    @patch("pybot.runtime.workers.items_to_storage_worker.require_template")
    @patch("pybot.runtime.workers.items_to_storage_worker.cell_contains_template")
    def test_get_fly_wings_sequence(
        self,
        cell_tpl: MagicMock,
        require_tpl: MagicMock,
        _cursor: MagicMock,
        _sleep: MagicMock,
    ) -> None:
        require_tpl.return_value = (10, 10)
        # One wing in the grid, then clear (avoids infinite re-deposit passes).
        cell_tpl.side_effect = [True] + [False] * 500
        self.config.fly_wings_amount = 150

        worker = self._worker(_FakePoller(10))
        self.assertEqual(self.ctx.wingcount, 0)
        worker.get_fly_wings()

        kinds = [c[0] for c in self.input.calls]
        self.assertIn("toggle_inv", kinds)
        self.assertIn(("left_click",), self.input.calls)
        self.assertIn(("play_chain", (("f8", 66, 0),)), self.input.calls)
        self.assertEqual(self.input.calls.count(("alt_rmb",)), 1)
        self.assertIn(("left_button", True), self.input.calls)
        self.assertIn(("left_button", False), self.input.calls)
        self.assertIn(("type", "150"), self.input.calls)
        self.assertEqual(self.ctx.wingcount, 150)
        # Use tab before inventory wing check / storage open
        toggle_i = kinds.index("toggle_inv")
        use_click_i = self.input.calls.index(("left_click",))
        chain_i = self.input.calls.index(("play_chain", (("f8", 66, 0),)))
        self.assertLess(toggle_i, use_click_i)
        self.assertLess(use_click_i, chain_i)
        self.ctx.logger.behavior.assert_any_call(
            "[STORAGE] GetFlyWings click Use tab (use_img)"
        )
        self.ctx.logger.behavior.assert_any_call(
            "[STORAGE] GetFlyWings scan Use grid 6x6 for wings"
        )
        self.ctx.logger.behavior.assert_any_call(
            "[STORAGE] GetFlyWings Use wing at col=0 row=0 — Alt+RMB deposit"
        )

    def test_weight_threshold_gate(self) -> None:
        worker = self._worker(_FakePoller(79, 100))
        self.config.weight_modifier = 80
        self.assertFalse(worker._weight_over_threshold())
        worker = self._worker(_FakePoller(80, 100))
        self.assertTrue(worker._weight_over_threshold())
        self.config.weight_modifier = 49
        self.assertFalse(worker._weight_over_threshold())


if __name__ == "__main__":
    unittest.main()
