"""ItemsToStorage / GetFlyWings worker — AHK call-sequence fidelity.

Weight and HP are read from PlayerVitals (published by the UI).  Tests that
need specific weight/HP values publish into a real PlayerVitals instance.
"""

from __future__ import annotations

import unittest
from contextlib import ExitStack
from unittest.mock import DEFAULT, MagicMock, patch

import numpy as np

from pybot.game_state import PlayerVitals
from pybot.recognition.ui.inventory import InventoryPanelHit, InventoryUiError
from pybot.runtime.input.input_backend import ShadowInputBackend
from pybot.runtime.runtime_context import HuntRuntimeContext
from pybot.runtime.workers.items_to_storage_worker import (
    ItemsToStorageWorker,
)
from pybot.viiper.keyboard import MOD_LEFT_ALT, vk_to_modifier

_WORKER = "pybot.runtime.workers.items_to_storage_worker"
_INVENTORY = "pybot.recognition.ui.inventory"
_AUTOMATION = "pybot.recognition.ui.inventory_automation"

# Where each inventory symbol is bound for patching.
_PATCH_TARGETS = {
    "require_inventory_panel": f"{_WORKER}.require_inventory_panel",
    "find_template": [
        f"{_WORKER}.find_template",
        f"{_AUTOMATION}.find_template",
    ],
    "find_storage_wing": f"{_WORKER}.find_storage_wing",
    "find_wings_in_use_grid": f"{_WORKER}.find_wings_in_use_grid",
    "slot_looks_empty": f"{_WORKER}.slot_looks_empty",
    "slot_contains_template": f"{_WORKER}._slot_contains_template",
    "require_template": f"{_AUTOMATION}.require_template",
    "is_inventory_open": f"{_AUTOMATION}.is_inventory_open",
    "is_storage_open": f"{_AUTOMATION}.is_storage_open",
}


def _default_worker_patch_targets():
    """Return a fresh dict of {target: mock_or_DEFAULT} for storage tests."""
    return {
        "time.sleep": MagicMock(return_value=None),
        "key_name_to_scan_code": MagicMock(return_value=0x1C),
        "InventoryAutomation.close_menus": MagicMock(),
        "InventoryAutomation.ensure_storage_open": MagicMock(),
        "InventoryAutomation.ensure_inventory_open": MagicMock(
            return_value=InventoryPanelHit(x=0, y=0, width=312, height=254)
        ),
        "InventoryAutomation.wait_for_inventory_panel": MagicMock(
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
    (e.g. ``"InventoryAutomation.close_menus"``, ``"require_inventory_panel"``).

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
        if name.startswith("pybot."):
            targets = [name]
            short = name.rsplit(".", 1)[-1]
        elif name.startswith("InventoryAutomation."):
            targets = [f"{_AUTOMATION}.{name}"]
            short = name
        elif name in _PATCH_TARGETS:
            targets = (
                _PATCH_TARGETS[name]
                if isinstance(_PATCH_TARGETS[name], list)
                else [_PATCH_TARGETS[name]]
            )
            short = name
        elif name.startswith(_WORKER):
            targets = [name]
            short = name[len(_WORKER) + 1 :]
        else:
            targets = [f"{_WORKER}.{name}"]
            short = name
        mock = None
        for full_target in targets:
            mock = stack.enter_context(patch(full_target, value))
        mocks[short] = mock
    return stack, mocks


# ── test helpers ────────────────────────────────────────────────────


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
        self.config.weight_modifier = 85
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
        self.vitals = PlayerVitals()

    def _worker(self, *, hunt_mode=None, teleport=None) -> ItemsToStorageWorker:
        from pybot.runtime.teleport import TeleportController
        tport = teleport or TeleportController(self.ctx, self.input, MagicMock())
        return ItemsToStorageWorker(
            self.ctx,
            self.input,
            tport,
            vitals=self.vitals,
            hunt_mode=hunt_mode,
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


    def test_teleport_settle_pauses_hunt_observation_but_not_ui_vitals(self) -> None:
        """Black/loading RO frames are ignored while UI feeds remain separate."""
        self.ctx.mark_running()
        self.assertTrue(self.ctx.should_run_discovery())
        self.assertTrue(self.ctx.should_run_tracking())

        self.ctx.discovery_suspend.set()
        self.assertFalse(self.ctx.should_run_combat())
        self.assertFalse(self.ctx.should_run_timers())
        self.assertFalse(self.ctx.should_run_discovery())
        self.assertFalse(self.ctx.should_run_tracking())

        # UI status/memory readers publish through PlayerVitals independently;
        # this gate only controls hunt observation and gameplay workers.
        self.vitals.publish_hp(80, 100)
        self.vitals.publish_sp(40, 100)
        self.assertEqual(self.vitals.hp_pair(), (80, 100))
        self.assertEqual(self.vitals.sp_pair(), (40, 100))

        self.ctx.discovery_suspend.clear()
        self.assertTrue(self.ctx.should_run_discovery())
        self.assertTrue(self.ctx.should_run_tracking())

    def test_storage_ops_pause_combat_not_timers(self) -> None:
        self.assertTrue(self.ctx.begin_storage_ops())
        self.assertTrue(self.ctx.should_run_workers())
        self.assertFalse(self.ctx.should_run_combat())
        self.assertTrue(self.ctx.should_run_discovery())
        self.assertTrue(self.ctx.should_run_tracking())
        self.assertFalse(self.ctx.try_begin_storage_ops())
        self.assertFalse(self.ctx.try_begin_sit_ops())
        self.assertFalse(self.ctx.try_begin_heal_ops())
        self.ctx.end_storage_ops()
        self.assertTrue(self.ctx.should_run_combat())
        self.assertTrue(self.ctx.should_run_discovery())
        self.assertTrue(self.ctx.should_run_tracking())




    def test_heal_ops_pause_combat_but_not_timers_then_resume(self) -> None:
        self.ctx.mark_running()
        self.assertTrue(self.ctx.begin_heal_ops())
        self.assertTrue(self.ctx.should_run_workers())
        self.assertFalse(self.ctx.should_run_combat())
        self.assertTrue(self.ctx.should_run_timers())
        # Discovery/tracking/timers stay up so state remains fresh during heal.
        self.assertTrue(self.ctx.should_run_discovery())
        self.assertTrue(self.ctx.should_run_tracking())
        self.assertFalse(self.ctx.resume_gate.is_set())
        self.assertFalse(self.ctx.try_begin_heal_ops())
        self.assertFalse(self.ctx.try_begin_sit_ops())
        self.assertFalse(self.ctx.try_begin_storage_ops())
        self.ctx.end_heal_ops()
        self.assertTrue(self.ctx.should_run_combat())
        self.assertTrue(self.ctx.should_run_timers())
        self.assertTrue(self.ctx.should_run_discovery())
        self.assertTrue(self.ctx.should_run_tracking())
        self.assertTrue(self.ctx.resume_gate.is_set())
        self.assertTrue(self.ctx.wait_while_combat_blocked(0.2))

    def test_storage_ops_clear_resume_gate_until_end(self) -> None:
        self.ctx.mark_running()
        self.assertTrue(self.ctx.begin_storage_ops())
        self.assertFalse(self.ctx.resume_gate.is_set())
        self.assertTrue(self.ctx.should_run_timers())
        self.ctx.end_storage_ops()
        self.assertTrue(self.ctx.resume_gate.is_set())
        self.assertTrue(self.ctx.should_run_combat())

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

    def test_weight_threshold_gate_defaults_to_85_percent_maximum(self) -> None:
        worker = self._worker()
        self.config.weight_modifier = 85
        self.vitals.publish_weight(84, 100)
        self.assertFalse(worker._weight_over_threshold())
        self.vitals.publish_weight(85, 100)
        self.assertTrue(worker._weight_over_threshold())
        # Runtime values above the UI maximum are fail-safe clamped to 85%.
        self.config.weight_modifier = 95
        self.assertTrue(worker._weight_over_threshold())
        self.vitals.publish_weight(84, 100)
        self.assertFalse(worker._weight_over_threshold())
        self.config.weight_modifier = 49
        self.assertFalse(worker._weight_over_threshold())

    def test_storage_stays_pending_when_discovery_sees_visible_mob(self) -> None:
        mode = MagicMock()
        mode.discovery_scan_age_ms = 0
        mode.discovery_confirmed_clear = True
        self.ctx.tracks.get_area_clear_candidate.return_value = type(
            "Clear", (), {"clear": False}
        )()
        worker = self._worker(hunt_mode=mode)
        self.vitals.publish_weight(85, 100)
        self.assertTrue(worker.storage_due())
        self.assertFalse(worker.can_execute_now())
        self.assertFalse(worker.process_pending())

    def test_storage_is_deferred_until_fresh_empty_discovery(self) -> None:
        mode = MagicMock()
        mode.discovery_scan_age_ms = 0
        mode.discovery_confirmed_clear = True
        self.ctx.tracks.get_area_clear_candidate.return_value = type(
            "Clear", (), {"clear": True}
        )()
        teleport = MagicMock()
        worker = self._worker(hunt_mode=mode, teleport=teleport)
        self.vitals.publish_weight(85, 100)
        self.assertTrue(worker.storage_due())
        self.assertTrue(worker.can_execute_now())
        with patch.object(worker, "storage_session") as storage_session:
            storage_session.return_value = None
            self.assertTrue(worker.process_pending())
        storage_session.assert_called_once_with(dump=True, restock=True)
        self.assertFalse(self.ctx.storage_event.is_set())

    def test_active_storage_ignores_late_visible_mob_without_teleporting(self) -> None:
        """A mob appearing after admission does not cancel active storage."""
        mode = MagicMock()
        mode.discovery_scan_age_ms = 0
        mode.discovery_confirmed_clear = True
        self.ctx.tracks.get_area_clear_candidate.return_value = type(
            "Clear", (), {"clear": True}
        )()
        teleport = MagicMock()
        worker = self._worker(hunt_mode=mode, teleport=teleport)
        self.vitals.publish_weight(85, 100)

        with patch.object(worker, "storage_session") as storage_session:
            self.assertTrue(worker.process_pending())

        storage_session.assert_called_once_with(dump=True, restock=True)
        self.assertFalse(self.ctx.storage_event.is_set())

    def test_fly_wings_would_hit_threshold_triggers_dump(self) -> None:
        self.config.weight_modifier = 85
        self.config.fly_wings_amount = 150
        worker = self._worker()
        self.vitals.publish_weight(70, 100)
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

            self.vitals.publish_weight(90, 100)
            self.vitals.publish_hp(1000, 1000)
            worker = self._worker()
            worker.storage_session(dump=True, restock=False)

            kinds = [c[0] for c in self.input.calls]
            m["InventoryAutomation.ensure_storage_open"].assert_called()
            m["InventoryAutomation.close_menus"].assert_called()
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

            self.vitals.publish_weight(90, 100)
            self.vitals.publish_hp(1000, 1000)
            worker = self._worker()
            worker.storage_session(dump=True, restock=False)

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
            m["InventoryAutomation.ensure_inventory_open"].return_value = panel
            self.config.fly_wings_amount = 150

            self.vitals.publish_weight(10, 100)
            self.vitals.publish_hp(1000, 1000)
            worker = self._worker()
            self.assertEqual(self.ctx.wingcount, 0)
            worker.storage_session(dump=False, restock=True)

            m["InventoryAutomation.ensure_inventory_open"].assert_called()
            m["InventoryAutomation.ensure_storage_open"].assert_called()
            m["InventoryAutomation.close_menus"].assert_called()
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

            self.vitals.publish_weight(10, 100)
            self.vitals.publish_hp(1000, 1000)
            worker = self._worker()
            worker.storage_session(dump=False, restock=True)

            m["InventoryAutomation.close_menus"].assert_called()
            self.assertTrue(self.ctx.fly_wings_exhausted)
            self.assertEqual(self.ctx.wingcount, 0)
            self.assertFalse(self.ctx.should_restock_fly_wings())
            self.assertNotIn(("type", "150"), self.input.calls)

    # ── wing-key-only config (no creamy TP) ─────────────────────────

    def test_restock_cycle_with_only_wing_key_no_creamy(self) -> None:
        """GetFlyWings runs when only the wing key is bound (no Creamy TP).

        User contract: Take Fly Wings checked, Teleport Key set, Creamy TP Key
        empty, weight dump disabled. Restock must still trigger, run, and
        re-arm for a later session once the wings are drained by wing-key
        teleports.
        """
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
            m["InventoryAutomation.ensure_inventory_open"].return_value = panel

            # Only the fly-wing teleport key; no Creamy TP binding anywhere.
            self.config.teleport_button = "q"
            self.config.teleport_scan_code = 16
            self.config.creamy_tp_button = ""
            self.config.creamy_tp_scan_code = 0
            self.config.take_fly_wings = True
            self.config.fly_wings_amount = 150
            # Weight dump disabled: restock-only storage sessions.
            self.config.weight_modifier = 49

            self.vitals.publish_weight(10, 100)
            self.vitals.publish_hp(1000, 1000)
            worker = self._worker()

            # Restock is due — no creamy dependency in any gate.
            self.assertTrue(self.ctx.should_restock_fly_wings())
            self.assertTrue(worker.storage_due())
            self.assertEqual(worker.storage_request(), (False, True))

            worker.storage_session(dump=False, restock=True)

            m["InventoryAutomation.ensure_storage_open"].assert_called()
            m["InventoryAutomation.close_menus"].assert_called()
            self.assertIn(("type", "150"), self.input.calls)
            self.assertEqual(self.ctx.wingcount, 150)
            self.assertFalse(self.ctx.fly_wings_exhausted)

            # Repeated-call safety: wing-key teleports drain the counter; at 0
            # the deferred action becomes due again for a later session.
            for _ in range(149):
                self.ctx.note_teleport_for_wings()
            self.assertEqual(self.ctx.wingcount, 1)
            self.assertFalse(worker.storage_due())
            self.ctx.note_teleport_for_wings()
            self.assertEqual(self.ctx.wingcount, 0)
            self.assertTrue(worker.storage_due())
            self.assertEqual(worker.storage_request(), (False, True))

    # ── storage session edge cases ─────────────────────────────────

    def test_storage_session_closes_menus_on_ui_miss(self) -> None:
        stack, m = _enter_worker_patches(
            find_template=DEFAULT,
            **{"InventoryAutomation.ensure_storage_open": MagicMock(
                side_effect=InventoryUiError("storage open failed")
            )},
        )
        with stack:
            m["find_template"].return_value = None

            self.vitals.publish_hp(1000, 1000)
            worker = self._worker()
            with self.assertRaises(InventoryUiError):
                worker.storage_session(dump=False, restock=True)
            m["InventoryAutomation.close_menus"].assert_called()

    def test_menu_validation_open_closed(self) -> None:
        stack, m = _enter_worker_patches(
            is_storage_open=DEFAULT,
            is_inventory_open=DEFAULT,
        )
        with stack:
            m["require_inventory_panel"].return_value = _fake_panel()
            self.vitals.publish_hp(1000, 1000)
            worker = self._worker()

            m["is_inventory_open"].return_value = False
            m["is_storage_open"].return_value = False
            with self.assertRaisesRegex(Exception, "inventory open"):
                worker._ui.wait_menu_state(
                    menu="inventory", want_open=True, label="inventory open",
                    timeout_s=0.0,
                )

            m["is_inventory_open"].return_value = True
            frame = worker._ui.wait_menu_state(
                menu="inventory", want_open=True, label="inventory open",
                timeout_s=0.5,
            )
            self.assertIs(frame, self.frame)

            m["is_storage_open"].return_value = True
            worker._ui.wait_menu_state(
                menu="storage", want_open=True, label="storage open",
                timeout_s=0.5,
            )
            m["is_storage_open"].return_value = False
            worker._ui.wait_menu_state(
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

            self.vitals.publish_weight(70, 100)
            self.vitals.publish_hp(1000, 1000)
            worker = self._worker()
            worker.storage_session(dump=True, restock=True)

            self.assertEqual(
                m["InventoryAutomation.ensure_storage_open"].call_count, 1
            )
            m["InventoryAutomation.close_menus"].assert_called_once()
            self.assertEqual(self.ctx.wingcount, 150)
            self.assertIn(("type", "150"), self.input.calls)


if __name__ == "__main__":
    unittest.main()
