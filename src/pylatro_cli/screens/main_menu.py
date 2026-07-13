"""MainMenuScreen, title screen with PLAY / OPTIONS / QUIT."""

from __future__ import annotations

import random
import string

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Center, Vertical
from textual.screen import Screen
from textual.widgets import Button, Input, Label, Static

TITLE_ART = r"""
             _       _
 _ __  _   _| | __ _| |_ _ __ ___
| '_ \| | | | |/ _` | __| '__/ _ \
| |_) | |_| | | (_| | |_| | | (_) |
| .__/ \__, |_|\__,_|\__|_|  \___/
|_|    |___/
"""


STAKE_NAMES = [
    "White", "Red", "Green", "Blue",
    "Purple", "Orange", "Gold", "Black",
]


class NewRunModal(Screen):
    """Modal dialog for starting a new run."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
    ]

    DEFAULT_CSS = """
    NewRunModal {
        align: center middle;
        background: rgba(0, 0, 0, 0.7);
    }
    #new-run-container {
        width: 50;
        height: auto;
        background: #1c1c3a;
        border: heavy #5c5c8a;
        padding: 2;
    }
    #new-run-container Label {
        margin: 1 0 0 0;
    }
    #new-run-container Input {
        margin: 0 0 1 0;
    }
    #new-run-container Button {
        margin: 1 1 0 0;
    }
    """

    DECK_KEYS = [
        "b_red", "b_blue", "b_yellow", "b_green",
        "b_black", "b_magic", "b_nebula", "b_ghost",
        "b_abandoned", "b_checkered", "b_zodiac", "b_painted",
        "b_anaglyph", "b_plasma", "b_erratic", "b_challenge",
    ]

    def __init__(self) -> None:
        super().__init__()
        self._deck_idx = 0
        self._stake = 1

    def compose(self) -> ComposeResult:
        with Vertical(id="new-run-container"):
            yield Label("New Run", classes="panel-title")
            yield Label("Seed (blank = random):")
            yield Input(placeholder="8-char seed", id="seed-input", max_length=8)
            yield Label("Deck:")
            yield Button(self._deck_name(), id="deck-btn", variant="default")
            yield Label("Stake:")
            yield Button(f"{STAKE_NAMES[self._stake - 1]} Stake", id="stake-btn", variant="default")
            yield Button("Start Run", id="start-btn", variant="primary")
            yield Button("Cancel", id="cancel-btn", variant="default")

    def _deck_name(self) -> str:
        return self.DECK_KEYS[self._deck_idx].replace("b_", "").title() + " Deck"

    def on_button_pressed(self, event: Button.Pressed) -> None:
        match event.button.id:
            case "deck-btn":
                self._deck_idx = (self._deck_idx + 1) % len(self.DECK_KEYS)
                event.button.label = self._deck_name()
            case "stake-btn":
                self._stake = (self._stake % 8) + 1
                event.button.label = f"{STAKE_NAMES[self._stake - 1]} Stake"
            case "start-btn":
                self._start_run()
            case "cancel-btn":
                self.app.pop_screen()

    def _start_run(self) -> None:
        seed_input = self.query_one("#seed-input", Input)
        seed = seed_input.value.strip()
        if not seed:
            seed = "".join(random.choices(string.ascii_uppercase + string.digits, k=8))

        deck_key = self.DECK_KEYS[self._deck_idx]

        app = self.app
        controller = app.controller
        controller.new_run(seed, self._stake, deck_key)

        # Pop this modal, pop main menu, push blind select
        app.pop_screen()  # modal
        app.pop_screen()  # main menu

        from .blind_select import BlindSelectScreen

        app.push_screen(BlindSelectScreen())

    def action_cancel(self) -> None:
        self.app.pop_screen()


class MainMenuScreen(Screen):
    """Main menu with title art and menu buttons."""

    DEFAULT_CSS = """
    MainMenuScreen {
        align: center middle;
    }
    #menu-box {
        width: 50;
        height: auto;
        align: center middle;
    }
    #menu-box Static {
        text-align: center;
        width: 100%;
    }
    #menu-box Button {
        width: 100%;
        margin: 1 4;
    }
    """

    def compose(self) -> ComposeResult:
        with Center(), Vertical(id="menu-box"):
            yield Static(TITLE_ART, classes="menu-title")
            yield Button("Play", id="play-btn", variant="primary")
            yield Button("Seed Walk", id="seedwalk-btn", variant="success")
            yield Button("Settings", id="settings-btn", variant="default")
            yield Button("Quit", id="quit-btn", variant="error")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        match event.button.id:
            case "play-btn":
                self.app.push_screen(NewRunModal())
            case "seedwalk-btn":
                from .seedwalk import SeedWalkSetupModal

                self.app.push_screen(SeedWalkSetupModal())
            case "settings-btn":
                from .settings_screen import SettingsScreen

                self.app.push_screen(SettingsScreen())
            case "quit-btn":
                self.app.exit()
