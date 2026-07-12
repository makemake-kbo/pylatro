"""BalatroApp, main Textual application."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import App
from textual.binding import Binding

from pylatro import GameData, load_game_data

from .controller import GameController
from .settings import UserSettings

if TYPE_CHECKING:
    from pylatro.seedwalk import SeedWalk


class BalatroApp(App):
    CSS_PATH = "balatro.tcss"
    TITLE = "pylatro"

    BINDINGS = [
        Binding("q", "quit", "Quit", priority=True),
        Binding("question_mark", "help", "Help"),
        Binding("escape", "back", "Back"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.settings = UserSettings.load()
        self.game_data: GameData | None = None
        self.controller: GameController | None = None
        self.walk: SeedWalk | None = None

    def on_mount(self) -> None:
        self.game_data = load_game_data()
        self.controller = GameController(data=self.game_data)
        from .screens.main_menu import MainMenuScreen

        self.push_screen(MainMenuScreen())

    def action_back(self) -> None:
        if len(self.screen_stack) > 1:
            self.pop_screen()

    def action_help(self) -> None:
        self.notify("pylatro, Terminal Balatro\n[q] Quit  [?] Help  [Esc] Back", title="Help")
