"""Inventory UI template matching tests."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

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


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    clear_template_cache()
    yield
    clear_template_cache()


def test_all_template_files_exist() -> None:
    for name in TEMPLATE_FILES:
        path = template_path(name)
        assert path.is_file(), path
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        assert img is not None and img.size > 0
    panel = PROJECT_ROOT / "assets" / "UI" / "InventoryPanel.png"
    assert panel.is_file()


def test_find_template_in_synthetic_frame() -> None:
    tpl = cv2.imread(str(template_path("use")), cv2.IMREAD_COLOR)
    assert tpl is not None
    frame = np.zeros((tpl.shape[0] + 40, tpl.shape[1] + 60, 3), dtype=np.uint8)
    frame[:] = (32, 32, 32)
    x0, y0 = 20, 10
    frame[y0 : y0 + tpl.shape[0], x0 : x0 + tpl.shape[1]] = tpl
    assert find_template(frame, "use") == (x0, y0)
    assert require_template(frame, "use") == (x0, y0)


def test_require_template_raises_when_missing() -> None:
    frame = np.zeros((80, 80, 3), dtype=np.uint8)
    with pytest.raises(InventoryUiError, match="use"):
        require_template(frame, "use")


def test_cell_contains_template() -> None:
    tpl = cv2.imread(str(template_path("wing")), cv2.IMREAD_COLOR)
    assert tpl is not None
    frame = np.full((200, 200, 3), 32, dtype=np.uint8)
    cx, cy = 100, 100
    x0 = cx - tpl.shape[1] // 2
    y0 = cy - tpl.shape[0] // 2
    frame[y0 : y0 + tpl.shape[0], x0 : x0 + tpl.shape[1]] = tpl
    assert cell_contains_template(frame, "wing", cx, cy)
    assert not cell_contains_template(frame, "wing", 10, 10)


def test_inventory_panel_asset_slot_geometry() -> None:
    panel = cv2.imread(
        str(PROJECT_ROOT / "assets" / "UI" / "InventoryPanel.png"),
        cv2.IMREAD_COLOR,
    )
    assert panel is not None
    hit = require_inventory_panel(panel)
    assert hit.x == 0 and hit.y == 0
    assert hit.slot_center(0, 0) == (INV_SLOT_ORIGIN_X, INV_SLOT_ORIGIN_Y)
    assert hit.slot_center(1, 0) == (
        INV_SLOT_ORIGIN_X + INV_SLOT_PITCH,
        INV_SLOT_ORIGIN_Y,
    )
    wings = find_wings_in_use_grid(panel, hit)
    assert wings == [
        (
            0,
            0,
            INV_SLOT_ORIGIN_X + INV_SLOT_AIM_DX,
            INV_SLOT_ORIGIN_Y + INV_SLOT_AIM_DY,
        ),
    ]
    assert hit.slot_aim(0, 0) == (
        INV_SLOT_ORIGIN_X + INV_SLOT_AIM_DX,
        INV_SLOT_ORIGIN_Y + INV_SLOT_AIM_DY,
    )
    assert not slot_looks_empty(panel, INV_SLOT_ORIGIN_X, INV_SLOT_ORIGIN_Y)
    assert slot_looks_empty(
        panel, INV_SLOT_ORIGIN_X + INV_SLOT_PITCH, INV_SLOT_ORIGIN_Y
    )


def test_flywing_inv_screenshot_panel_and_slots() -> None:
    path = PROJECT_ROOT / "tests" / "FlyWingINV.png"
    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    assert frame is not None
    panel = require_inventory_panel(frame)
    assert panel.x == 514 and panel.y == 368
    assert len(list(panel.iter_slot_centers())) == INV_COLS * INV_ROWS
    wings = find_wings_in_use_grid(frame, panel)
    assert (
        0,
        0,
        panel.x + INV_SLOT_ORIGIN_X + INV_SLOT_AIM_DX,
        panel.y + INV_SLOT_ORIGIN_Y + INV_SLOT_AIM_DY,
    ) in [(c, r, x, y) for c, r, x, y in wings]
    cx, cy = panel.slot_center(0, 0)
    assert slot_contains_template(frame, "wing", cx, cy)
    assert find_inventory_panel(np.zeros_like(frame)) is None
    storage_wing = find_storage_wing(frame, panel)
    assert storage_wing is not None
    assert not (
        panel.x <= storage_wing[0] < panel.x + panel.width
        and panel.y <= storage_wing[1] < panel.y + panel.height
    )


def test_menu_open_closed_on_screenshots() -> None:
    from pybot.recognition.ui.inventory import is_inventory_open, is_storage_open

    open_frame = cv2.imread(
        str(PROJECT_ROOT / "tests" / "FlyWingINV.png"), cv2.IMREAD_COLOR
    )
    closed_frame = cv2.imread(
        str(PROJECT_ROOT / "tests" / "StatusPanel.png"), cv2.IMREAD_COLOR
    )
    assert open_frame is not None and closed_frame is not None
    assert is_inventory_open(open_frame) is True
    assert is_storage_open(open_frame) is True
    assert is_inventory_open(closed_frame) is False
    assert is_storage_open(closed_frame) is False
