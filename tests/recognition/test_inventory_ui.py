"""Inventory UI template matching tests."""

from __future__ import annotations

import unittest

import cv2
import numpy as np

from pybot.paths import PROJECT_ROOT
from pybot.recognition.ui.inventory import (
    INV_COLS,
    INV_ROWS,
    INV_SLOT_AIM_DX,
    INV_SLOT_AIM_DY,
    INV_SLOT_ORIGIN_X,
    INV_SLOT_ORIGIN_Y,
    INV_SLOT_PITCH,
    TEMPLATE_FILES,
    InventoryUiError,
    cell_contains_template,
    clear_template_cache,
    find_inventory_panel,
    find_storage_wing,
    find_template,
    find_wings_in_use_grid,
    require_inventory_panel,
    require_template,
    slot_contains_template,
    slot_looks_empty,
    template_path,
)


class InventoryUiTests(unittest.TestCase):
    def setUp(self) -> None:
        clear_template_cache()

    def tearDown(self) -> None:
        clear_template_cache()

    def test_all_template_files_exist(self) -> None:
        for name in TEMPLATE_FILES:
            path = template_path(name)
            self.assertTrue(path.is_file(), str(path))
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            self.assertIsNotNone(img)
            self.assertGreater(img.size, 0)
        panel = PROJECT_ROOT / "assets" / "UI" / "InventoryPanel.png"
        self.assertTrue(panel.is_file())

    def test_find_template_in_synthetic_frame(self) -> None:
        tpl = cv2.imread(str(template_path("use")), cv2.IMREAD_COLOR)
        self.assertIsNotNone(tpl)
        frame = np.zeros((tpl.shape[0] + 40, tpl.shape[1] + 60, 3), dtype=np.uint8)
        frame[:] = (32, 32, 32)
        x0, y0 = 20, 10
        frame[y0 : y0 + tpl.shape[0], x0 : x0 + tpl.shape[1]] = tpl
        self.assertEqual(find_template(frame, "use"), (x0, y0))
        self.assertEqual(require_template(frame, "use"), (x0, y0))

    def test_require_template_raises_when_missing(self) -> None:
        frame = np.zeros((80, 80, 3), dtype=np.uint8)
        with self.assertRaisesRegex(InventoryUiError, "use"):
            require_template(frame, "use")

    def test_cell_contains_template(self) -> None:
        tpl = cv2.imread(str(template_path("wing")), cv2.IMREAD_COLOR)
        self.assertIsNotNone(tpl)
        frame = np.full((200, 200, 3), 32, dtype=np.uint8)
        cx, cy = 100, 100
        x0 = cx - tpl.shape[1] // 2
        y0 = cy - tpl.shape[0] // 2
        frame[y0 : y0 + tpl.shape[0], x0 : x0 + tpl.shape[1]] = tpl
        self.assertTrue(cell_contains_template(frame, "wing", cx, cy))
        self.assertFalse(cell_contains_template(frame, "wing", 10, 10))

    def test_inventory_panel_asset_slot_geometry(self) -> None:
        panel = cv2.imread(
            str(PROJECT_ROOT / "assets" / "UI" / "InventoryPanel.png"),
            cv2.IMREAD_COLOR,
        )
        self.assertIsNotNone(panel)
        hit = require_inventory_panel(panel)
        self.assertEqual(hit.x, 0)
        self.assertEqual(hit.y, 0)
        self.assertEqual(hit.slot_center(0, 0), (INV_SLOT_ORIGIN_X, INV_SLOT_ORIGIN_Y))
        self.assertEqual(
            hit.slot_center(1, 0),
            (INV_SLOT_ORIGIN_X + INV_SLOT_PITCH, INV_SLOT_ORIGIN_Y),
        )
        wings = find_wings_in_use_grid(panel, hit)
        self.assertEqual(
            wings,
            [(0, 0, INV_SLOT_ORIGIN_X + INV_SLOT_AIM_DX, INV_SLOT_ORIGIN_Y + INV_SLOT_AIM_DY)],
        )
        self.assertEqual(
            hit.slot_aim(0, 0),
            (INV_SLOT_ORIGIN_X + INV_SLOT_AIM_DX, INV_SLOT_ORIGIN_Y + INV_SLOT_AIM_DY),
        )
        self.assertFalse(slot_looks_empty(panel, INV_SLOT_ORIGIN_X, INV_SLOT_ORIGIN_Y))
        self.assertTrue(slot_looks_empty(panel, INV_SLOT_ORIGIN_X + INV_SLOT_PITCH, INV_SLOT_ORIGIN_Y))

    def test_flywing_inv_screenshot_panel_and_slots(self) -> None:
        path = PROJECT_ROOT / "tests" / "FlyWingINV.png"
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        self.assertIsNotNone(frame)
        panel = require_inventory_panel(frame)
        self.assertEqual(panel.x, 514)
        self.assertEqual(panel.y, 368)
        self.assertEqual(len(list(panel.iter_slot_centers())), INV_COLS * INV_ROWS)
        wings = find_wings_in_use_grid(frame, panel)
        self.assertIn(
            (
                0, 0,
                panel.x + INV_SLOT_ORIGIN_X + INV_SLOT_AIM_DX,
                panel.y + INV_SLOT_ORIGIN_Y + INV_SLOT_AIM_DY,
            ),
            [(c, r, x, y) for c, r, x, y in wings],
        )
        cx, cy = panel.slot_center(0, 0)
        self.assertTrue(slot_contains_template(frame, "wing", cx, cy))
        self.assertIsNone(find_inventory_panel(np.zeros_like(frame)))
        storage_wing = find_storage_wing(frame, panel)
        self.assertIsNotNone(storage_wing)
        self.assertFalse(
            panel.x <= storage_wing[0] < panel.x + panel.width
            and panel.y <= storage_wing[1] < panel.y + panel.height
        )

    def test_menu_open_closed_on_screenshots(self) -> None:
        from pybot.recognition.ui.inventory import is_inventory_open, is_storage_open

        open_frame = cv2.imread(
            str(PROJECT_ROOT / "tests" / "FlyWingINV.png"), cv2.IMREAD_COLOR
        )
        closed_frame = cv2.imread(
            str(PROJECT_ROOT / "tests" / "StatusPanel.png"), cv2.IMREAD_COLOR
        )
        self.assertIsNotNone(open_frame)
        self.assertIsNotNone(closed_frame)
        self.assertTrue(is_inventory_open(open_frame))
        self.assertTrue(is_storage_open(open_frame))
        self.assertFalse(is_inventory_open(closed_frame))
        self.assertFalse(is_storage_open(closed_frame))

    def test_inventory_panel_not_confused_with_storage(self) -> None:
        """Master Storage title chrome must not win over Inventory."""
        frame = cv2.imread(
            str(PROJECT_ROOT / "tests" / "FlyWingINV.png"), cv2.IMREAD_COLOR
        )
        self.assertIsNotNone(frame)
        panel = require_inventory_panel(frame)
        self.assertEqual(panel.x, 514)
        self.assertEqual(panel.y, 368)
        self.assertFalse(panel.x == 1074 and panel.y == 302)
        wings = find_wings_in_use_grid(frame, panel)
        self.assertTrue(any(c == 0 and r == 0 for c, r, _x, _y in wings))


if __name__ == "__main__":
    unittest.main()
