"""Main-window lifecycle control state tests."""

from __future__ import annotations

from types import SimpleNamespace

import tkinter as tk

from pybot.app.bot_lifecycle import BotState
from pybot.app.main_window import MainWindow


class _Widget:
    def __init__(self) -> None:
        self.options: dict[str, object] = {}

    def configure(self, **options: object) -> None:
        self.options.update(options)


def test_paused_state_enables_stop_and_continue() -> None:
    window = SimpleNamespace(
        bot_button=_Widget(),
        continue_button=_Widget(),
        bot_status=_Widget(),
        status_indicator=_Widget(),
    )

    MainWindow._on_bot_state_changed(window, BotState.PAUSED)

    assert window.bot_button.options == {"text": "Stop Bot", "state": tk.NORMAL}
    assert window.continue_button.options == {"state": tk.NORMAL}
    assert window.bot_status.options == {"text": "Paused (game focus lost)"}
    assert window.status_indicator.options == {
        "text": " PAUSED ",
        "bg": "#f9a825",
    }
