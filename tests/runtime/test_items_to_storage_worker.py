"""ItemsToStorage / GetFlyWings worker — AHK call-sequence fidelity."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from pybot.game_state import MemorySnapshot
from pybot.config.clients import MemoryAddresses
from pybot.recognition.ui.inventory import InventoryPanelHit, InventoryUiError
from pybot.runtime.constants import STORAGE_ENTER_SCAN_CODE
from pybot.runtime.input.input_backend import ShadowInputBackend
from pybot.runtime.runtime_context import HuntRuntimeContext
from pybot.runtime.workers.items_to_storage_worker import (
    ItemsToStorageWorker,
    StorageCriticalHpError,
)
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


def _fake_panel(x: int = 0, y: int = 0) -> InventoryPanelHit:
    return InventoryPanelHit(x=x, y=y, width=312, height=254)


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
            self.ctx,
            self.input,
            self.memory,
            hunt_mode=MagicMock(),
            poller=poller,
        )

    def test_vk_menu_maps_to_left_alt(self) -> None:
        self.assertEqual(vk_to_modifier(0x12), MOD_LEFT_ALT)

    def test_sit_ops_block_should_run_workers(self) -> None:
        self.assertTrue(self.ctx.should_run_workers())
        self.assertTrue(self.ctx.should_run_combat())
        self.assertTrue(self.ctx.begin_sit_ops())
        self.assertFalse(self.ctx.should_run_workers())
        self.assertFalse(self.ctx.should_run_combat())
        self.assertFalse(self.ctx.try_begin_sit_ops())
        self.assertFalse(self.ctx.try_begin_storage_ops())
        self.ctx.end_sit_ops()
        self.assertTrue(self.ctx.should_run_workers())
        self.assertTrue(self.ctx.should_run_combat())

    def test_storage_ops_pause_combat_not_timers(self) -> None:
        self.assertTrue(self.ctx.begin_storage_ops())
        self.assertTrue(self.ctx.should_run_workers())
        self.assertFalse(self.ctx.should_run_combat())
        self.assertFalse(self.ctx.should_run_discovery())
        self.assertFalse(self.ctx.should_run_tracking())
        self.assertFalse(self.ctx.try_begin_storage_ops())
        self.assertFalse(self.ctx.try_begin_sit_ops())
        self.ctx.end_storage_ops()
        self.assertTrue(self.ctx.should_run_combat())
        self.assertTrue(self.ctx.should_run_discovery())
        self.assertTrue(self.ctx.should_run_tracking())

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
    @patch(
        "pybot.runtime.workers.items_to_storage_worker.ItemsToStorageWorker._close_menus"
    )
    @patch(
        "pybot.runtime.workers.items_to_storage_worker.ItemsToStorageWorker._ensure_storage_open"
    )
    @patch(
        "pybot.runtime.workers.items_to_storage_worker.ItemsToStorageWorker._ensure_inventory_open",
        return_value=_fake_panel(),
    )
    @patch(
        "pybot.runtime.workers.items_to_storage_worker.ItemsToStorageWorker._wait_for_inventory_panel",
        return_value=(_fake_panel(), np.zeros((10, 10, 3), dtype=np.uint8)),
    )
    @patch("pybot.runtime.workers.items_to_storage_worker.require_inventory_panel")
    @patch("pybot.runtime.workers.items_to_storage_worker.slot_looks_empty")
    @patch("pybot.runtime.workers.items_to_storage_worker.require_template")
    @patch(
        "pybot.runtime.workers.items_to_storage_worker.slot_contains_template",
        return_value=False,
    )
    @patch("pybot.runtime.workers.items_to_storage_worker.find_template")
    def test_items_to_storage_sequence(
        self,
        find_tpl: MagicMock,
        _slot_wing: MagicMock,
        require_tpl: MagicMock,
        slot_empty: MagicMock,
        require_panel: MagicMock,
        _wait_panel: MagicMock,
        _ensure_inv_open: MagicMock,
        ensure_stor_open: MagicMock,
        close_menus: MagicMock,
        _cursor: MagicMock,
        _sleep: MagicMock,
    ) -> None:
        require_panel.return_value = _fake_panel()
        require_tpl.side_effect = lambda _f, name, **_kw: {
            "use": (10, 10),
            "eqp": (30, 10),
            "etc": (40, 10),
            "close": (50, 50),
            "wing": (60, 60),
        }[name]
        slot_empty.side_effect = (
            # Use / Eqp / Etc: one occupied slot then a full clear scan each.
            [False] + [True] * 48
            + [False] + [True] * 48
            + [False] + [True] * 48
        )
        find_tpl.side_effect = (
            lambda _f, name, **_kw: (10, 10) if name == "use" else None
        )

        worker = self._worker(_FakePoller(90))
        worker.items_to_storage()

        kinds = [c[0] for c in self.input.calls]
        ensure_stor_open.assert_called()
        close_menus.assert_called()
        self.assertIn(("left_click",), self.input.calls)
        self.assertGreaterEqual(kinds.count("alt_rmb"), 3)
        self.assertEqual(kinds.count("left_click"), 3)  # use/eqp/etc

    @patch("pybot.runtime.workers.items_to_storage_worker.time.sleep", return_value=None)
    @patch(
        "pybot.runtime.workers.items_to_storage_worker._cursor_pos",
        return_value=(150, 100),
    )
    @patch(
        "pybot.runtime.workers.items_to_storage_worker.ItemsToStorageWorker._close_menus"
    )
    @patch(
        "pybot.runtime.workers.items_to_storage_worker.ItemsToStorageWorker._ensure_storage_open"
    )
    @patch(
        "pybot.runtime.workers.items_to_storage_worker.ItemsToStorageWorker._ensure_inventory_open",
        return_value=_fake_panel(),
    )
    @patch(
        "pybot.runtime.workers.items_to_storage_worker.ItemsToStorageWorker._wait_for_inventory_panel",
        return_value=(_fake_panel(), np.zeros((10, 10, 3), dtype=np.uint8)),
    )
    @patch("pybot.runtime.workers.items_to_storage_worker.require_inventory_panel")
    @patch("pybot.runtime.workers.items_to_storage_worker.slot_looks_empty")
    @patch("pybot.runtime.workers.items_to_storage_worker.require_template")
    @patch(
        "pybot.runtime.workers.items_to_storage_worker.slot_contains_template",
        return_value=False,
    )
    def test_items_to_storage_ok_dialog_uses_enter_284(
        self,
        _slot_wing: MagicMock,
        require_tpl: MagicMock,
        slot_empty: MagicMock,
        require_panel: MagicMock,
        _wait_panel: MagicMock,
        _ensure_inv_open: MagicMock,
        _ensure_stor_open: MagicMock,
        _close_menus: MagicMock,
        _cursor: MagicMock,
        _sleep: MagicMock,
    ) -> None:
        require_panel.return_value = _fake_panel()
        require_tpl.return_value = (10, 10)
        # Use clear, Eqp clear, Etc: one item then clear (OK dialog on deposit).
        slot_empty.side_effect = (
            [True] * 48 + [True] * 48 + [False] + [True] * 48
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
        return_value=(156, 82),
    )
    @patch(
        "pybot.runtime.workers.items_to_storage_worker.ItemsToStorageWorker._close_menus"
    )
    @patch(
        "pybot.runtime.workers.items_to_storage_worker.ItemsToStorageWorker._ensure_storage_open"
    )
    @patch(
        "pybot.runtime.workers.items_to_storage_worker.ItemsToStorageWorker._ensure_inventory_open",
        return_value=_fake_panel(),
    )
    @patch(
        "pybot.runtime.workers.items_to_storage_worker.ItemsToStorageWorker._wait_for_inventory_panel",
        return_value=(_fake_panel(), np.zeros((10, 10, 3), dtype=np.uint8)),
    )
    @patch("pybot.runtime.workers.items_to_storage_worker.require_inventory_panel")
    @patch("pybot.runtime.workers.items_to_storage_worker.slot_looks_empty")
    @patch("pybot.runtime.workers.items_to_storage_worker.require_template")
    @patch("pybot.runtime.workers.items_to_storage_worker.slot_contains_template")
    @patch("pybot.runtime.workers.items_to_storage_worker.find_template")
    def test_items_to_storage_skips_use_tab_wings(
        self,
        find_tpl: MagicMock,
        slot_wing: MagicMock,
        require_tpl: MagicMock,
        slot_empty: MagicMock,
        require_panel: MagicMock,
        _wait_panel: MagicMock,
        _ensure_inv_open: MagicMock,
        _ensure_stor_open: MagicMock,
        _close_menus: MagicMock,
        _cursor: MagicMock,
        _sleep: MagicMock,
    ) -> None:
        require_panel.return_value = _fake_panel()
        require_tpl.return_value = (10, 10)
        find_tpl.return_value = None
        # Use scan1: wing at (0,0), potion at (1,0) → deposit potion.
        # Use/Eqp/Etc clear scans afterward.
        slot_empty.side_effect = (
            [False, False] + [True] * 48 + [True] * 48 + [True] * 48
        )
        slot_wing.side_effect = [True, False]

        worker = self._worker(_FakePoller(90))
        worker.items_to_storage()

        self.assertEqual(self.input.calls.count(("alt_rmb",)), 1)
        self.assertIn(("move", 178, 92), self.input.calls)
        # Off-screen clear before Use-tab scans (client origin 100,50 → 98,48).
        self.assertIn(("move", 98, 48), self.input.calls)

    @patch("pybot.runtime.workers.items_to_storage_worker.time.sleep", return_value=None)
    @patch(
        "pybot.runtime.workers.items_to_storage_worker._cursor_pos",
        return_value=(150, 100),
    )
    @patch(
        "pybot.runtime.workers.items_to_storage_worker.ItemsToStorageWorker._close_menus"
    )
    @patch(
        "pybot.runtime.workers.items_to_storage_worker.ItemsToStorageWorker._ensure_storage_open"
    )
    @patch(
        "pybot.runtime.workers.items_to_storage_worker.ItemsToStorageWorker._ensure_inventory_open",
        return_value=_fake_panel(),
    )
    @patch(
        "pybot.runtime.workers.items_to_storage_worker.ItemsToStorageWorker._wait_for_inventory_panel",
        return_value=(_fake_panel(), np.zeros((10, 10, 3), dtype=np.uint8)),
    )
    @patch("pybot.runtime.workers.items_to_storage_worker.require_inventory_panel")
    @patch("pybot.runtime.workers.items_to_storage_worker.find_storage_wing")
    @patch("pybot.runtime.workers.items_to_storage_worker.find_wings_in_use_grid")
    @patch("pybot.runtime.workers.items_to_storage_worker.require_template")
    def test_get_fly_wings_sequence(
        self,
        require_tpl: MagicMock,
        find_wings: MagicMock,
        find_storage: MagicMock,
        require_panel: MagicMock,
        _wait_panel: MagicMock,
        ensure_inv_open: MagicMock,
        ensure_stor_open: MagicMock,
        close_menus: MagicMock,
        _cursor: MagicMock,
        _sleep: MagicMock,
    ) -> None:
        panel = _fake_panel()
        require_panel.return_value = panel
        require_tpl.return_value = (10, 10)
        find_wings.side_effect = [[(0, 0, 46, 42)], []]
        find_storage.return_value = (200, 100)
        self.config.fly_wings_amount = 150

        worker = self._worker(_FakePoller(10))
        self.assertEqual(self.ctx.wingcount, 0)
        worker.get_fly_wings()

        ensure_inv_open.assert_called()
        ensure_stor_open.assert_called()
        close_menus.assert_called()
        self.assertIn(("move", 146, 92), self.input.calls)
        self.assertIn(("move", 98, 48), self.input.calls)
        self.assertEqual(self.input.calls.count(("alt_rmb",)), 1)
        self.assertIn(("type", "150"), self.input.calls)
        self.assertEqual(self.ctx.wingcount, 150)
        self.assertFalse(self.ctx.fly_wings_exhausted)
        # Restock-only: Use selected once at session open (soft select may skip).
        # left_click count depends on whether use_img was found (mocked True → 1).

    @patch("pybot.runtime.workers.items_to_storage_worker.time.sleep", return_value=None)
    @patch(
        "pybot.runtime.workers.items_to_storage_worker.ItemsToStorageWorker._close_menus"
    )
    @patch(
        "pybot.runtime.workers.items_to_storage_worker.ItemsToStorageWorker._ensure_inventory_open",
        return_value=_fake_panel(),
    )
    @patch(
        "pybot.runtime.workers.items_to_storage_worker.ItemsToStorageWorker._open_storage",
        side_effect=InventoryUiError("storage open failed"),
    )
    @patch(
        "pybot.runtime.workers.items_to_storage_worker.find_template",
        return_value=None,
    )
    def test_storage_session_closes_menus_on_ui_miss(
        self,
        _find_tpl: MagicMock,
        _open_storage: MagicMock,
        _ensure_inv_open: MagicMock,
        close_menus: MagicMock,
        _sleep: MagicMock,
    ) -> None:
        worker = self._worker(_FakePoller(10))
        with self.assertRaises(InventoryUiError):
            worker.storage_session(dump=False, restock=True)
        close_menus.assert_called()

    @patch("pybot.runtime.workers.items_to_storage_worker.time.sleep", return_value=None)
    @patch(
        "pybot.runtime.workers.items_to_storage_worker.ItemsToStorageWorker._close_menus"
    )
    @patch(
        "pybot.runtime.workers.items_to_storage_worker.ItemsToStorageWorker._ensure_storage_open"
    )
    @patch(
        "pybot.runtime.workers.items_to_storage_worker.ItemsToStorageWorker._ensure_inventory_open",
        return_value=_fake_panel(),
    )
    @patch(
        "pybot.runtime.workers.items_to_storage_worker.find_template",
        return_value=None,
    )
    @patch("pybot.runtime.workers.items_to_storage_worker.read_status_panel")
    def test_restock_force_closes_only_on_critical_hp(
        self,
        read_hp: MagicMock,
        _find_tpl: MagicMock,
        _ensure_inv_open: MagicMock,
        _ensure_stor_open: MagicMock,
        close_menus: MagicMock,
        _sleep: MagicMock,
    ) -> None:
        from pybot.recognition.ui.status_panel import StatusPanelValues

        read_hp.return_value = StatusPanelValues(
            hp=40,
            hp_max=100,
            sp=50,
            sp_max=100,
            weight=10,
            weight_max=100,
            panel_origin=(0, 0),
        )
        worker = self._worker(_FakePoller(10))
        with self.assertRaises(StorageCriticalHpError):
            worker.storage_session(dump=False, restock=True)
        # Critical path does not use the normal session close — caller force-closes.
        close_menus.assert_not_called()

    @patch("pybot.runtime.workers.items_to_storage_worker.time.sleep", return_value=None)
    @patch(
        "pybot.runtime.workers.items_to_storage_worker.ItemsToStorageWorker._close_menus"
    )
    @patch(
        "pybot.runtime.workers.items_to_storage_worker.ItemsToStorageWorker._ensure_storage_open"
    )
    @patch(
        "pybot.runtime.workers.items_to_storage_worker.ItemsToStorageWorker._ensure_inventory_open",
        return_value=_fake_panel(),
    )
    @patch(
        "pybot.runtime.workers.items_to_storage_worker.ItemsToStorageWorker._wait_for_inventory_panel",
        return_value=(_fake_panel(), np.zeros((10, 10, 3), dtype=np.uint8)),
    )
    @patch("pybot.runtime.workers.items_to_storage_worker.require_inventory_panel")
    @patch("pybot.runtime.workers.items_to_storage_worker.find_storage_wing")
    @patch("pybot.runtime.workers.items_to_storage_worker.find_wings_in_use_grid")
    @patch("pybot.runtime.workers.items_to_storage_worker.require_template")
    def test_get_fly_wings_abandons_when_storage_empty(
        self,
        require_tpl: MagicMock,
        find_wings: MagicMock,
        find_storage: MagicMock,
        require_panel: MagicMock,
        _wait_panel: MagicMock,
        _ensure_inv_open: MagicMock,
        _ensure_stor_open: MagicMock,
        close_menus: MagicMock,
        _sleep: MagicMock,
    ) -> None:
        require_panel.return_value = _fake_panel()
        require_tpl.return_value = (10, 10)
        find_wings.return_value = []
        find_storage.return_value = None
        self.config.fly_wings_amount = 150
        self.config.creamy_tp_button = "w"
        self.config.creamy_tp_scan_code = 17

        worker = self._worker(_FakePoller(10))
        worker.get_fly_wings()

        close_menus.assert_called()
        self.assertTrue(self.ctx.fly_wings_exhausted)
        self.assertEqual(self.ctx.wingcount, 0)
        self.assertFalse(self.ctx.should_restock_fly_wings())
        self.assertEqual(self.ctx.active_teleport_button(), "w")
        self.assertNotIn(("type", "150"), self.input.calls)

    def test_exhausted_wings_use_creamy_tp(self) -> None:
        self.config.teleport_button = "q"
        self.config.teleport_scan_code = 16
        self.config.creamy_tp_button = "w"
        self.config.creamy_tp_scan_code = 17
        self.config.take_fly_wings = True
        self.config.open_storage_steps = (("f8", 66, 0),)
        self.ctx.wingcount = 0
        self.assertTrue(self.ctx.should_restock_fly_wings())
        self.ctx.mark_fly_wings_exhausted()
        self.assertFalse(self.ctx.should_restock_fly_wings())
        self.assertEqual(self.ctx.active_teleport_scan_code(), 17)
        self.ctx.note_teleport_for_wings()
        self.assertEqual(self.ctx.wingcount, 0)

    def test_take_fly_wings_off_uses_creamy_tp(self) -> None:
        self.config.teleport_button = "q"
        self.config.teleport_scan_code = 16
        self.config.creamy_tp_button = "w"
        self.config.creamy_tp_scan_code = 17
        self.config.take_fly_wings = False
        self.assertEqual(self.ctx.active_teleport_button(), "w")
        self.assertEqual(self.ctx.active_teleport_scan_code(), 17)

    def test_take_fly_wings_off_without_creamy_uses_mob_teleport(self) -> None:
        self.config.teleport_button = "q"
        self.config.teleport_scan_code = 16
        self.config.active_teleport_button.return_value = "q"
        self.config.active_teleport_scan_code.return_value = 16
        self.config.creamy_tp_button = ""
        self.config.creamy_tp_scan_code = 0
        self.config.take_fly_wings = False
        self.assertEqual(self.ctx.active_teleport_button(), "q")
        self.assertEqual(self.ctx.active_teleport_scan_code(), 16)

    def test_exhausted_wings_without_creamy_keep_mob_teleport(self) -> None:
        self.config.teleport_button = "q"
        self.config.teleport_scan_code = 16
        self.config.active_teleport_button.return_value = "q"
        self.config.active_teleport_scan_code.return_value = 16
        self.config.creamy_tp_button = ""
        self.config.creamy_tp_scan_code = 0
        self.config.take_fly_wings = True
        self.config.open_storage_steps = (("f8", 66, 0),)
        self.ctx.mark_fly_wings_exhausted()
        self.assertEqual(self.ctx.active_teleport_button(), "q")
        self.assertEqual(self.ctx.active_teleport_scan_code(), 16)

    @patch("pybot.runtime.workers.items_to_storage_worker.time.sleep", return_value=None)
    @patch("pybot.runtime.workers.items_to_storage_worker.is_storage_open")
    @patch("pybot.runtime.workers.items_to_storage_worker.is_inventory_open")
    @patch("pybot.runtime.workers.items_to_storage_worker.require_inventory_panel")
    def test_menu_validation_open_closed(
        self,
        require_panel: MagicMock,
        inv_open: MagicMock,
        stor_open: MagicMock,
        _sleep: MagicMock,
    ) -> None:
        require_panel.return_value = _fake_panel()
        worker = self._worker(_FakePoller(10))

        inv_open.return_value = False
        stor_open.return_value = False
        with self.assertRaisesRegex(Exception, "inventory open"):
            worker._wait_menu_state(
                menu="inventory", want_open=True, label="inventory open", timeout_s=0.0
            )

        inv_open.return_value = True
        frame = worker._wait_menu_state(
            menu="inventory", want_open=True, label="inventory open", timeout_s=0.5
        )
        self.assertIs(frame, self.frame)

        stor_open.return_value = True
        worker._wait_menu_state(
            menu="storage", want_open=True, label="storage open", timeout_s=0.5
        )
        stor_open.return_value = False
        worker._wait_menu_state(
            menu="storage", want_open=False, label="storage closed", timeout_s=0.5
        )

    def test_weight_threshold_gate(self) -> None:
        worker = self._worker(_FakePoller(79, 100))
        self.config.weight_modifier = 80
        self.assertFalse(worker._weight_over_threshold())
        worker = self._worker(_FakePoller(80, 100))
        self.assertTrue(worker._weight_over_threshold())
        self.config.weight_modifier = 49
        self.assertFalse(worker._weight_over_threshold())

    def test_fly_wings_would_hit_threshold_triggers_dump(self) -> None:
        # weight 70, max 100, gate 80% → threshold 80.
        # 150 wings * 5 = 750 → projected 820 >= 80 → dump before restock.
        self.config.weight_modifier = 80
        self.config.fly_wings_amount = 150
        worker = self._worker(_FakePoller(70, 100))
        self.assertFalse(worker._weight_over_threshold())
        self.assertTrue(worker._fly_wings_would_hit_threshold())

        # Small restock that stays under threshold: 1 wing * 5 → 75 < 80.
        self.config.fly_wings_amount = 1
        self.assertFalse(worker._fly_wings_would_hit_threshold())

    @patch("pybot.runtime.workers.items_to_storage_worker.time.sleep", return_value=None)
    @patch(
        "pybot.runtime.workers.items_to_storage_worker._cursor_pos",
        return_value=(150, 100),
    )
    @patch(
        "pybot.runtime.workers.items_to_storage_worker.ItemsToStorageWorker._close_menus"
    )
    @patch(
        "pybot.runtime.workers.items_to_storage_worker.ItemsToStorageWorker._ensure_storage_open"
    )
    @patch(
        "pybot.runtime.workers.items_to_storage_worker.ItemsToStorageWorker._ensure_inventory_open",
        return_value=_fake_panel(),
    )
    @patch(
        "pybot.runtime.workers.items_to_storage_worker.ItemsToStorageWorker._wait_for_inventory_panel",
        return_value=(_fake_panel(), np.zeros((10, 10, 3), dtype=np.uint8)),
    )
    @patch("pybot.runtime.workers.items_to_storage_worker.require_inventory_panel")
    @patch("pybot.runtime.workers.items_to_storage_worker.slot_looks_empty")
    @patch("pybot.runtime.workers.items_to_storage_worker.find_storage_wing")
    @patch("pybot.runtime.workers.items_to_storage_worker.find_wings_in_use_grid")
    @patch(
        "pybot.runtime.workers.items_to_storage_worker.slot_contains_template",
        return_value=False,
    )
    @patch("pybot.runtime.workers.items_to_storage_worker.require_template")
    @patch("pybot.runtime.workers.items_to_storage_worker.find_template")
    def test_merged_dump_and_restock_opens_storage_once(
        self,
        find_tpl: MagicMock,
        require_tpl: MagicMock,
        _slot_wing: MagicMock,
        find_wings: MagicMock,
        find_storage: MagicMock,
        slot_empty: MagicMock,
        require_panel: MagicMock,
        _wait_panel: MagicMock,
        _ensure_inv_open: MagicMock,
        ensure_stor_open: MagicMock,
        close_menus: MagicMock,
        _cursor: MagicMock,
        _sleep: MagicMock,
    ) -> None:
        require_panel.return_value = _fake_panel()
        require_tpl.return_value = (10, 10)
        find_tpl.return_value = None
        find_wings.side_effect = [[], []]
        find_storage.return_value = (200, 100)
        slot_empty.side_effect = (
            # Use / Eqp / Etc dump grids, then restock path does not use slot_empty.
            [False] + [True] * 48
            + [False] + [True] * 48
            + [False] + [True] * 48
        )
        self.config.fly_wings_amount = 150

        worker = self._worker(_FakePoller(70, 100))
        worker.storage_session(dump=True, restock=True)

        self.assertEqual(ensure_stor_open.call_count, 1)
        close_menus.assert_called_once()
        self.assertEqual(self.ctx.wingcount, 150)
        self.assertIn(("type", "150"), self.input.calls)


if __name__ == "__main__":
    unittest.main()
