"""GameOverScreen, win/loss display with stats."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Center, Vertical
from textual.screen import Screen
from textual.widgets import Button, Static


class GameOverScreen(Screen):
    """Game over: shows win/loss, stats, and return to menu."""

    BINDINGS = [
        Binding("enter", "return_to_menu", "Menu"),
    ]

    DEFAULT_CSS = """
    GameOverScreen {
        align: center middle;
    }
    #gameover-box {
        width: 50;
        height: auto;
        background: #1c1c3a;
        border: heavy #5c5c8a;
        padding: 2;
        align: center middle;
    }
    #gameover-box Static {
        text-align: center;
        width: 100%;
    }
    #gameover-box Button {
        width: 100%;
        margin: 2 4 0 4;
    }
    """

    def __init__(self, won: bool = False, **kwargs) -> None:
        super().__init__(**kwargs)
        self.won = won

    def compose(self) -> ComposeResult:
        ctrl = self.app.controller
        state = ctrl.state

        with Center(), Vertical(id="gameover-box"):
            if self.won:
                yield Static("YOU WIN!", id="result-text")
            else:
                yield Static("GAME OVER", id="result-text")

            if state:
                yield Static(f"Seed: {state.seed}")
                yield Static(f"Ante: {state.round_resets.ante}")
                yield Static(f"Round: {state.round}")
                yield Static(f"Jokers: {len(state.jokers)}")
                yield Static(f"Dollars: ${state.dollars}")
                yield Static(f"Hands Played: {state.hands_played}")

            yield Button("Return to Menu", id="menu-btn", variant="primary")

    def on_mount(self) -> None:
        self.query_one("#result-text", Static).styles.color = "#27ae60" if self.won else "#e74c3c"

    def action_return_to_menu(self) -> None:
        from .main_menu import MainMenuScreen

        self.app.switch_screen(MainMenuScreen())

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "menu-btn":
            self.action_return_to_menu()
