"""Pure helpers for formatting status values shown by the main window."""

from __future__ import annotations

from pybot.recognition.ui.status_panel import StatusPanelValues


def format_pair(current: int | None, maximum: int | None) -> str:
    """Format a current/maximum value pair without any Tk dependencies."""
    if current is None and maximum is None:
        return "—"
    if maximum is None:
        return str(current)
    if current is None:
        return f"—/{maximum}"
    return f"{current}/{maximum}"


def status_panel_numbers(
    values: StatusPanelValues,
) -> tuple[int, int, int, int, int | None, int | None]:
    """Return the fields used to detect whether an OCR result changed."""
    return (
        values.hp,
        values.hp_max,
        values.sp,
        values.sp_max,
        values.weight,
        values.weight_max,
    )


__all__ = ["format_pair", "status_panel_numbers"]
