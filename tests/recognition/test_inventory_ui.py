"""Inventory UI template matching tests."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from pybot.recognition.ui.inventory import (
    TEMPLATE_FILES,
    cell_contains_template,
    clear_template_cache,
    find_template,
    require_template,
    template_path,
)
from pybot.recognition.ui.inventory import InventoryUiError


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
    # wing fits in the 40×40 cell window; empty_cell.bmp is solid white (unstable CCOEFF).
    tpl = cv2.imread(str(template_path("wing")), cv2.IMREAD_COLOR)
    assert tpl is not None
    frame = np.full((200, 200, 3), 32, dtype=np.uint8)
    cx, cy = 100, 100
    x0 = cx - tpl.shape[1] // 2
    y0 = cy - tpl.shape[0] // 2
    frame[y0 : y0 + tpl.shape[0], x0 : x0 + tpl.shape[1]] = tpl
    assert cell_contains_template(frame, "wing", cx, cy)
    assert not cell_contains_template(frame, "wing", 10, 10)


def test_flywing_inv_screenshot_finds_wing() -> None:
    """Client crop of wing_img must match inventory/storage on FlyWingINV.png."""
    from pybot.paths import PROJECT_ROOT

    path = PROJECT_ROOT / "tests" / "FlyWingINV.png"
    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    assert frame is not None
    loc = require_template(frame, "wing")
    assert loc == (564, 394)
    cell1 = require_template(frame, "cell1")
    assert cell_contains_template(
        frame, "wing", cell1[0] + 20, cell1[1] + 20
    )
