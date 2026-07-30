"""ItemsToStorage / GetFlyWings worker — AHK call-sequence fidelity."""

from __future__ import annotations

import unittest
from contextlib import ExitStack
from unittest.mock import DEFAULT, MagicMock, patch

import numpy as np

from pybot.game_state import MemorySnapshot
from pybot.config.clients import MemoryAddresses
from pybot.recognition.ui.inventory import InventoryPanelHit, InventoryUiError
from pybot.recognition.ui.status_panel import StatusPanelValues
from pybot.runtime.input.input_backend import ShadowInputBackend
from pybot.runtime.runtime_context import HuntRuntimeContext
from pybot.runtime.workers.items_to_storage_worker import (
    ItemsToStorageWorker,
    StorageCriticalHpError,
)
from pybot.viiper.keyboard import MOD_LEFT_ALT, vk_to_modifier

_WORKER = "pybot.runtime.workers.items_to_storage_worker"

# ── common patches shared across storage / restock tests ─────────────

# Patch targets present in nearly every storage/restock test.
# Each key is a short name; _enter_worker_patches prepends _WORKER.
# Wrapped in a factory so every call gets fresh MagicMock instances.


def _default_worker_patch_targets():
    """Return a fresh dict of {target: mock_or_DEFAULT} for storage tests."""
    return {
        "time.sleep": MagicMock(return_value=None),
        "_cursor_pos": MagicMock(return_value=(150, 100)),
        "ItemsToStorageWorker._close_menus": MagicMock(),
        "ItemsToStorageWorker._ensure_storage_open": MagicMock(),
        "ItemsToStorageWorker._ensure_inventory_open": MagicMock(
            return_value=InventoryPanelHit(x=0, y=0, width=312, height=254)
        ),
        "ItemsToStorageWorker._wait_for_inventory_panel": MagicMock(
            return_value=(
                InventoryPanelHit(x=0, y=0, width=312, height=254),
                np.zeros((10, 10, 3), dtype=np.uint8),
            )
        ),
        "require_inventory_panel": MagicMock(),
    }


def _enter_worker_patches(**extras):
    """Enter all common storage-worker patches, plus any extras.

    Returns ``(stack, mocks)`` where *stack* is an ``ExitStack`` that undoes
    patches on exit, and *mocks* is a dict keyed by the short attribute name
    (e.g. ``"_close_menus"``, ``"require_inventory_panel"``).

    Every call creates **fresh** ``MagicMock`` instances, so tests are
    isolated and call-count assertions don't leak.

    Pass ``DEFAULT`` in *extras* for a bare ``MagicMock()``, or an explicit
    ``MagicMock(...)`` for a pre-configured one.
    """
    stack = ExitStack()
    mocks: dict[str, MagicMock] = {}

    merged = _default_worker_patch_targets()
    for k, v in extras.items():
        merged[k] = v

    for name, value in merged.items():
        if value is DEFAULT:
            value = MagicMock()
        full_target = name if name.startswith(_WORKER) else f"{_WORKER}.{name}"
        mock = stack.enter_context(patch(full_target, value))
        # Short key for lookup: drop _WORKER prefix if present.
        short = name[len(_WORKER) + 1:] if name.startswith(_WORKER) else name
        mocks[short] = mock
    return stack, mocks


# ── test helpers ────────────────────────────────────────────────────


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


# ── tests ───────────────────────────────────────────────────────────


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

    # ── unit / no-patch tests ──────────────────────────────────────

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

    def test_active_teleport_delegates_to_config(self) -> None:
        self.config.active_teleport_button.return_value = "q"
        self.config.active_teleport_scan_code.return_value = 16
        self.assertEqual(self.ctx.active_teleport_button(), "q")
        self.assertEqual(self.ctx.active_teleport_scan_code(), 16)

    def test_weight_threshold_gate(self) -> None:
        worker = self._worker(_FakePoller(79, 100))
        self.config.weight_modifier = 80
        self.assertFalse(worker._weight_over_threshold())
        worker = self._worker(_FakePoller(80, 100))
        self.assertTrue(worker._weight_over_threshold())
        self.config.weight_modifier = 49
        self.assertFalse(worker._weight_over_threshold())

    def test_fly_wings_would_hit_threshold_triggers_dump(self) -> None:
        self.config.weight_modifier = 80
        self.config.fly_wings_amount = 150
        worker = self._worker(_FakePoller(70, 100))
        self.assertFalse(worker._weight_over_threshold())
        self.assertTrue(worker._fly_wings_would_hit_threshold())
        self.config.fly_wings_amount = 1
        self.assertFalse(worker._fly_wings_would_hit_threshold())

    # ── items-to-storage (dump) ────────────────────────────────────

    def test_items_to_storage_sequence(self) -> None:
        stack, m = _enter_worker_patches(
            slot_looks_empty=DEFAULT,
            require_template=DEFAULT,
            slot_contains_template=DEFAULT,
            find_template=DEFAULT,
        )
        with stack:
            m["require_inventory_panel"].return_value = _fake_panel()
            m["require_template"].side_effect = lambda _f, name, **_kw: {
                "use": (10, 10), "eqp": (30, 10), "etc": (40, 10),
                "close": (50, 50), "wing": (60, 60),
            }[name]
            m["slot_looks_empty"].side_effect = (
                [False, True, False, True, False, True]
            )
            m["slot_contains_template"].return_value = False
            m["find_template"].side_effect = lambda _f, name, **_kw: {
                "use": (10, 10), "eqp": (30, 10), "etc": (40, 10),
            }.get(name)

            worker = self._worker(_FakePoller(90))
            worker.items_to_storage()

            kinds = [c[0] for c in self.input.calls]
            m["ItemsToStorageWorker._ensure_storage_open"].assert_called()
            m["ItemsToStorageWorker._close_menus"].assert_called()
            self.assertIn(("left_click",), self.input.calls)
            self.assertGreaterEqual(kinds.count("alt_rmb"), 3)
            self.assertEqual(kinds.count("left_click"), 3)

    def test_items_to_storage_no_ok_dialog_moves_cursor(self) -> None:
        stack, m = _enter_worker_patches(
            slot_looks_empty=DEFAULT,
            require_template=DEFAULT,
            slot_contains_template=DEFAULT,
        )
        with stack:
            m["require_inventory_panel"].return_value = _fake_panel()
            m["require_template"].return_value = (10, 10)
            m["slot_looks_empty"].side_effect = [True, True, False, True]
            m["slot_contains_template"].return_value = False

            worker = self._worker(_FakePoller(90))
            worker.items_to_storage()

            kinds = [c[0] for c in self.input.calls]
            self.assertNotIn("key_tap", kinds)

    # ── get-fly-wings (restock) ────────────────────────────────────

    def test_get_fly_wings_sequence(self) -> None:
        stack, m = _enter_worker_patches(
            find_storage_wing=DEFAULT,
            find_wings_in_use_grid=DEFAULT,
            require_template=DEFAULT,
        )
        with stack:
            panel = _fake_panel()
            m["require_inventory_panel"].return_value = panel
            m["require_template"].return_value = (10, 10)
            m["find_wings_in_use_grid"].side_effect = [[(0, 0, 46, 42)], []]
            m["find_storage_wing"].return_value = (200, 100)
            m["ItemsToStorageWorker._ensure_inventory_open"].return_value = panel
            self.config.fly_wings_amount = 150

            worker = self._worker(_FakePoller(10))
            self.assertEqual(self.ctx.wingcount, 0)
            worker.get_fly_wings()

            m["ItemsToStorageWorker._ensure_inventory_open"].assert_called()
            m["ItemsToStorageWorker._ensure_storage_open"].assert_called()
            m["ItemsToStorageWorker._close_menus"].assert_called()
            self.assertIn(("move", 146, 92), self.input.calls)
            self.assertIn(("move", 98, 48), self.input.calls)
            self.assertEqual(self.input.calls.count(("alt_rmb",)), 1)
            self.assertIn(("type", "150"), self.input.calls)
            self.assertEqual(self.ctx.wingcount, 150)
            self.assertFalse(self.ctx.fly_wings_exhausted)

    def test_get_fly_wings_abandons_when_storage_empty(self) -> None:
        stack, m = _enter_worker_patches(
            find_storage_wing=DEFAULT,
            find_wings_in_use_grid=DEFAULT,
            require_template=DEFAULT,
        )
        with stack:
            m["require_inventory_panel"].return_value = _fake_panel()
            m["require_template"].return_value = (10, 10)
            m["find_wings_in_use_grid"].return_value = []
            m["find_storage_wing"].return_value = None
            self.config.fly_wings_amount = 150
            self.config.creamy_tp_button = "w"
            self.config.creamy_tp_scan_code = 17

            worker = self._worker(_FakePoller(10))
            worker.get_fly_wings()

            m["ItemsToStorageWorker._close_menus"].assert_called()
            self.assertTrue(self.ctx.fly_wings_exhausted)
            self.assertEqual(self.ctx.wingcount, 0)
            self.assertFalse(self.ctx.should_restock_fly_wings())
            self.assertNotIn(("type", "150"), self.input.calls)

    # ── storage session edge cases ─────────────────────────────────

    def test_storage_session_closes_menus_on_ui_miss(self) -> None:
        stack, m = _enter_worker_patches(
            find_template=DEFAULT,
            **{f"{_WORKER}.ItemsToStorageWorker._open_storage": MagicMock(
                side_effect=InventoryUiError("storage open failed")
            )},
        )
        with stack:
            m["find_template"].return_value = None

            worker = self._worker(_FakePoller(10))
            with self.assertRaises(InventoryUiError):
                worker.storage_session(dump=False, restock=True)
            m["ItemsToStorageWorker._close_menus"].assert_called()

    def test_restock_force_closes_only_on_critical_hp(self) -> None:
        stack, m = _enter_worker_patches(
            find_template=DEFAULT,
            read_status_panel=DEFAULT,
        )
        with stack:
            m["find_template"].return_value = None
            m["read_status_panel"].return_value = StatusPanelValues(
                hp=40, hp_max=100, sp=50, sp_max=100,
                weight=10, weight_max=100, panel_origin=(0, 0),
            )

            worker = self._worker(_FakePoller(10))
            with self.assertRaises(StorageCriticalHpError):
                worker.storage_session(dump=False, restock=True)
            m["ItemsToStorageWorker._close_menus"].assert_not_called()

    def test_menu_validation_open_closed(self) -> None:
        stack, m = _enter_worker_patches(
            is_storage_open=DEFAULT,
            is_inventory_open=DEFAULT,
        )
        with stack:
            m["require_inventory_panel"].return_value = _fake_panel()
            worker = self._worker(_FakePoller(10))

            m["is_inventory_open"].return_value = False
            m["is_storage_open"].return_value = False
            with self.assertRaisesRegex(Exception, "inventory open"):
                worker._wait_menu_state(
                    menu="inventory", want_open=True, label="inventory open",
                    timeout_s=0.0,
                )

            m["is_inventory_open"].return_value = True
            frame = worker._wait_menu_state(
                menu="inventory", want_open=True, label="inventory open",
                timeout_s=0.5,
            )
            self.assertIs(frame, self.frame)

            m["is_storage_open"].return_value = True
            worker._wait_menu_state(
                menu="storage", want_open=True, label="storage open",
                timeout_s=0.5,
            )
            m["is_storage_open"].return_value = False
            worker._wait_menu_state(
                menu="storage", want_open=False, label="storage closed",
                timeout_s=0.5,
            )

    def test_merged_dump_and_restock_opens_storage_once(self) -> None:
        stack, m = _enter_worker_patches(
            slot_looks_empty=DEFAULT,
            find_storage_wing=DEFAULT,
            find_wings_in_use_grid=DEFAULT,
            slot_contains_template=DEFAULT,
            require_template=DEFAULT,
            find_template=DEFAULT,
        )
        with stack:
            m["require_inventory_panel"].return_value = _fake_panel()
            m["require_template"].return_value = (10, 10)
            m["find_template"].return_value = None
            m["find_wings_in_use_grid"].side_effect = [[], []]
            m["find_storage_wing"].return_value = (200, 100)
            m["slot_contains_template"].return_value = False
            m["slot_looks_empty"].side_effect = (
                [False, True, False, True, False, True]
            )
            self.config.fly_wings_amount = 150

            worker = self._worker(_FakePoller(70, 100))
            worker.storage_session(dump=True, restock=True)

            self.assertEqual(
                m["ItemsToStorageWorker._ensure_storage_open"].call_count, 1
            )
            m["ItemsToStorageWorker._close_menus"].assert_called_once()
            self.assertEqual(self.ctx.wingcount, 150)
            self.assertIn(("type", "150"), self.input.calls)


# ── _read_weight unit tests (OCR and memory modes) ──────────────


class ItemsToStorageWorkerReadWeightTests(unittest.TestCase):
    """Unit tests for ``_read_weight()`` covering both memory and OCR paths."""

    def setUp(self) -> None:
        self.config = MagicMock()
        self.config.hwnd = 1
        self.config.weight_modifier = 80
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
        self.input = _RecordingInput()

    def _worker(
        self,
        poller: _FakePoller,
        memory: MemoryAddresses | None = None,
    ) -> ItemsToStorageWorker:
        if memory is None:
            memory = MemoryAddresses(current_weight=1, max_weight=2)
        return ItemsToStorageWorker(
            self.ctx,
            self.input,
            memory,
            hunt_mode=MagicMock(),
            poller=poller,
        )

    # ── memory mode ─────────────────────────────────────────────────

    def test_memory_mode_returns_weight(self) -> None:
        """snap.ok=True with valid weights → (weight, weight_max)"""
        worker = self._worker(_FakePoller(450, 500))
        result = worker._read_weight()
        self.assertEqual(result, (450, 500))

    def test_memory_mode_returns_none_when_snap_not_ok(self) -> None:
        """snap.ok=False → None, logs once"""

        class _FailPoller:
            def read(self, hwnd, addresses) -> MemorySnapshot:
                return MemorySnapshot(ok=False, error="read failed")

        worker = self._worker(_FailPoller())
        result = worker._read_weight()
        self.assertIsNone(result)
        # Should have logged once
        self.assertEqual(
            self.ctx.logger.behavior.call_count, 1
        )
        log_msg = self.ctx.logger.behavior.call_args[0][0]
        self.assertIn("memory_weight_unavailable", log_msg)

    def test_memory_mode_returns_none_when_weight_none(self) -> None:
        """snap.weight=None → None"""

        class _NoneWeightPoller:
            def read(self, hwnd, addresses) -> MemorySnapshot:
                return MemorySnapshot(
                    weight=None, weight_max=500, ok=True
                )

        worker = self._worker(_NoneWeightPoller())
        result = worker._read_weight()
        self.assertIsNone(result)

    def test_memory_mode_returns_none_when_max_weight_none(self) -> None:
        """snap.weight_max=None → None"""

        class _NoneMaxPoller:
            def read(self, hwnd, addresses) -> MemorySnapshot:
                return MemorySnapshot(
                    weight=450, weight_max=None, ok=True
                )

        worker = self._worker(_NoneMaxPoller())
        result = worker._read_weight()
        self.assertIsNone(result)

    def test_memory_mode_returns_none_when_max_weight_zero(self) -> None:
        """snap.weight_max=0 → None"""

        class _ZeroMaxPoller:
            def read(self, hwnd, addresses) -> MemorySnapshot:
                return MemorySnapshot(
                    weight=450, weight_max=0, ok=True
                )

        worker = self._worker(_ZeroMaxPoller())
        result = worker._read_weight()
        self.assertIsNone(result)

    # ── OCR mode ────────────────────────────────────────────────────

    def test_ocr_mode_falls_through_when_no_memory_addresses(self) -> None:
        """current_weight=0 → uses OCR path"""
        # Memory addresses all zero → OCR path
        memory = MemoryAddresses(current_weight=0, max_weight=0)
        worker = self._worker(_FakePoller(450, 500), memory=memory)

        # _capture_client() returns self.frame, read_status_panel
        # returns valid OCR data.  Both are mocked on the context.
        with patch(
            f"{_WORKER}.read_status_panel",
            return_value=StatusPanelValues(
                hp=100,
                hp_max=100,
                sp=100,
                sp_max=100,
                weight=300,
                weight_max=400,
                panel_origin=(0, 0),
            ),
        ):
            result = worker._read_weight()

        self.assertEqual(result, (300, 400))

    def test_ocr_mode_returns_none_on_capture_error(self) -> None:
        """_capture_client raises InventoryUiError → None, no log"""
        memory = MemoryAddresses(current_weight=0, max_weight=0)
        worker = self._worker(_FakePoller(450, 500), memory=memory)

        # Make capture_client raise
        self.ctx.capture.capture_client.side_effect = InventoryUiError(
            "capture failed"
        )
        result = worker._read_weight()
        self.assertIsNone(result)
        # OCR mode doesn't log on capture failure
        self.ctx.logger.behavior.assert_not_called()

    def test_ocr_mode_returns_none_when_panel_none(self) -> None:
        """read_status_panel returns None → None, log once"""
        memory = MemoryAddresses(current_weight=0, max_weight=0)
        worker = self._worker(_FakePoller(450, 500), memory=memory)

        with patch(f"{_WORKER}.read_status_panel", return_value=None):
            result = worker._read_weight()

        self.assertIsNone(result)
        self.assertEqual(self.ctx.logger.behavior.call_count, 1)
        log_msg = self.ctx.logger.behavior.call_args[0][0]
        self.assertIn("ocr_weight_unavailable", log_msg)

    def test_ocr_mode_returns_none_when_weight_none(self) -> None:
        """StatusPanelValues.weight=None → None"""
        memory = MemoryAddresses(current_weight=0, max_weight=0)
        worker = self._worker(_FakePoller(450, 500), memory=memory)

        with patch(
            f"{_WORKER}.read_status_panel",
            return_value=StatusPanelValues(
                hp=100,
                hp_max=100,
                sp=100,
                sp_max=100,
                weight=None,
                weight_max=400,
                panel_origin=(0, 0),
            ),
        ):
            result = worker._read_weight()

        self.assertIsNone(result)

    def test_ocr_mode_returns_none_when_max_weight_none(self) -> None:
        """StatusPanelValues.weight_max=None → None"""
        memory = MemoryAddresses(current_weight=0, max_weight=0)
        worker = self._worker(_FakePoller(450, 500), memory=memory)

        with patch(
            f"{_WORKER}.read_status_panel",
            return_value=StatusPanelValues(
                hp=100,
                hp_max=100,
                sp=100,
                sp_max=100,
                weight=300,
                weight_max=None,
                panel_origin=(0, 0),
            ),
        ):
            result = worker._read_weight()

        self.assertIsNone(result)

    def test_ocr_mode_returns_none_when_max_weight_zero(self) -> None:
        """StatusPanelValues.weight_max=0 → None"""
        memory = MemoryAddresses(current_weight=0, max_weight=0)
        worker = self._worker(_FakePoller(450, 500), memory=memory)

        with patch(
            f"{_WORKER}.read_status_panel",
            return_value=StatusPanelValues(
                hp=100,
                hp_max=100,
                sp=100,
                sp_max=100,
                weight=300,
                weight_max=0,
                panel_origin=(0, 0),
            ),
        ):
            result = worker._read_weight()

        self.assertIsNone(result)

    def test_ocr_mode_returns_weight_on_success(self) -> None:
        """Valid StatusPanelValues → (weight, weight_max)"""
        memory = MemoryAddresses(current_weight=0, max_weight=0)
        worker = self._worker(_FakePoller(450, 500), memory=memory)

        with patch(
            f"{_WORKER}.read_status_panel",
            return_value=StatusPanelValues(
                hp=100,
                hp_max=100,
                sp=100,
                sp_max=100,
                weight=250,
                weight_max=350,
                panel_origin=(0, 0),
            ),
        ):
            result = worker._read_weight()

        self.assertEqual(result, (250, 350))

    # ── logging dedup ───────────────────────────────────────────────

    def test_memory_failure_logs_once_then_skips_repeat(self) -> None:
        """Second consecutive failure doesn't re-log"""

        class _FailPoller:
            def read(self, hwnd, addresses) -> MemorySnapshot:
                return MemorySnapshot(ok=False, error="fail")

        worker = self._worker(_FailPoller())
        worker._read_weight()
        worker._read_weight()
        worker._read_weight()
        self.assertEqual(self.ctx.logger.behavior.call_count, 1)

    def test_ocr_failure_logs_once_then_skips_repeat(self) -> None:
        """Second consecutive OCR failure doesn't re-log"""
        memory = MemoryAddresses(current_weight=0, max_weight=0)
        worker = self._worker(_FakePoller(450, 500), memory=memory)

        with patch(f"{_WORKER}.read_status_panel", return_value=None):
            worker._read_weight()
            worker._read_weight()
            worker._read_weight()

        self.assertEqual(self.ctx.logger.behavior.call_count, 1)

    def test_success_clears_fail_log_so_next_failure_relogs(self) -> None:
        """Success resets last_fail_log → next failure logs again"""
        memory = MemoryAddresses(current_weight=0, max_weight=0)
        worker = self._worker(_FakePoller(450, 500), memory=memory)

        # First call: fail
        with patch(f"{_WORKER}.read_status_panel", return_value=None):
            worker._read_weight()

        self.assertEqual(self.ctx.logger.behavior.call_count, 1)

        # Second call: succeed (clears last_fail_log)
        with patch(
            f"{_WORKER}.read_status_panel",
            return_value=StatusPanelValues(
                hp=100,
                hp_max=100,
                sp=100,
                sp_max=100,
                weight=250,
                weight_max=350,
                panel_origin=(0, 0),
            ),
        ):
            result = worker._read_weight()

        self.assertEqual(result, (250, 350))
        # Success doesn't increment behavior call count
        self.assertEqual(self.ctx.logger.behavior.call_count, 1)

        # Third call: fail again (should re-log because last_fail_log
        # was cleared by the success)
        with patch(f"{_WORKER}.read_status_panel", return_value=None):
            worker._read_weight()

        self.assertEqual(self.ctx.logger.behavior.call_count, 2)


if __name__ == "__main__":
    unittest.main()
