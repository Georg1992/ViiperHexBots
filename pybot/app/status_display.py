"""Pure helpers for formatting status values shown by the main window."""

from __future__ import annotations


def format_pair(current: int | None, maximum: int | None) -> str:
    """Format a current/maximum value pair without any Tk dependencies."""
    if current is None and maximum is None:
        return "—"
    if maximum is None:
        return str(current)
    if current is None:
        return f"—/{maximum}"
    return f"{current}/{maximum}"


__all__ = ["format_pair"]
