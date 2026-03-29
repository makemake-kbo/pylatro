"""SettingsScreen — user settings modal."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Button, Static

from ..settings import KeyMode


class SettingsScreen(Screen):
    """Settings: key mode toggle."""

    BINDINGS = [
        Binding("escape", "dismiss_screen", "Back"),
    ]

    DEFAULT_CSS = """
    SettingsScreen {
        align: center middle;
        background: rgba(0, 0, 0, 0.7);
    }
    #settings-container {
        width: 40;
        height: auto;
        background: #1c1c3a;
        border: heavy #5c5c8a;
        padding: 2;
    }
    #settings-container Static {
        text-align: center;
        width: 100%;
    }
    #settings-container Button {
        width: 100%;
        margin: 1 2;
    }
    """

    def compose(self) -> ComposeResult:
        settings = self.app.settings
        with Vertical(id="settings-container"):
            yield Static("Settings")
            yield Static("")
            yield Button(
                f"Key Mode: {settings.key_mode.value}",
                id="keymode-btn",
                variant="default",
            )
            yield Button("Back", id="back-btn", variant="default")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        match event.button.id:
            case "keymode-btn":
                settings = self.app.settings
                if settings.key_mode == KeyMode.VIM:
                    settings.key_mode = KeyMode.ARROWS
                else:
                    settings.key_mode = KeyMode.VIM
                settings.save()
                event.button.label = f"Key Mode: {settings.key_mode.value}"
            case "back-btn":
                self.app.pop_screen()

    def action_dismiss_screen(self) -> None:
        self.app.pop_screen()
