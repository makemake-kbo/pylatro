"""BoosterPackScreen — modal overlay for opening packs."""

from __future__ import annotations

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.screen import Screen
from textual.widgets import Button, Static

from ..theme import BALATRO_PALETTE


class BoosterPackScreen(Screen):
    """Modal: pick cards from an opened booster pack."""

    BINDINGS = [
        Binding("h", "cursor_left", "Left", show=False),
        Binding("l", "cursor_right", "Right", show=False),
        Binding("left", "cursor_left", "Left", show=False),
        Binding("right", "cursor_right", "Right", show=False),
        Binding("up", "cursor_up", "Up", show=False),
        Binding("down", "cursor_down", "Down", show=False),
        Binding("k", "cursor_up", "Up", show=False),
        Binding("j", "cursor_down", "Down", show=False),
        Binding("enter", "claim", "Claim"),
        Binding("escape", "skip_pack", "Skip"),
    ]

    DEFAULT_CSS = """
    BoosterPackScreen {
        align: center middle;
        background: rgba(0, 0, 0, 0.7);
    }
    #pack-container {
        width: 60;
        height: auto;
        max-height: 80%;
        background: #1c1c3a;
        border: heavy #5c5c8a;
        padding: 2;
    }
    #pack-header {
        text-align: center;
        margin-bottom: 1;
    }
    #pack-cards-display {
        height: auto;
        min-height: 4;
    }
    #pack-buttons {
        layout: horizontal;
        height: 3;
        align: center middle;
        margin-top: 1;
    }
    #pack-buttons Button {
        margin: 0 1;
    }
    """

    cursor: reactive[int] = reactive(0)

    def compose(self) -> ComposeResult:
        with Vertical(id="pack-container"):
            yield Static("", id="pack-header")
            yield Static("", id="pack-cards-display")
            with Horizontal(id="pack-buttons"):
                yield Button("Claim", id="claim-btn", variant="primary")
                yield Button("Skip Pack", id="skip-btn", variant="default")

    def on_mount(self) -> None:
        self._refresh_display()

    def _refresh_display(self) -> None:
        ctrl = self.app.controller
        state = ctrl.state
        if state is None or state.pack is None:
            return

        pack = state.pack
        pack_center = state.data.centers.get(pack.booster_key, {})
        pack_name = pack_center.get("name", pack.booster_key)

        self.query_one("#pack-header", Static).update(
            f"{pack_name} — {pack.choices_remaining} choice(s) remaining"
        )

        # Render pack cards
        t = Text()
        for i, card in enumerate(pack.cards):
            center = state.data.centers.get(card.center_key, {})
            name = center.get("name", card.center_key)

            if i == self.cursor:
                t.append(" >> ", style=BALATRO_PALETTE["card_selected"])
            else:
                t.append("    ")

            type_colors = {"Joker": "#ffc107", "Tarot": "#9c27b0", "Planet": "#2196f3", "Spectral": "#b0bec5", "Base": "#ecf0f1", "Enhanced": "#27ae60"}
            tc = type_colors.get(card.card_type, "#95a5a6")
            t.append(f"[{card.card_type}] ", style=tc)
            t.append(f"{name}\n", style="bold #ecf0f1")

        self.query_one("#pack-cards-display", Static).update(t)

    def watch_cursor(self, value: int) -> None:
        if self.is_mounted:
            self._refresh_display()

    def action_cursor_left(self) -> None:
        if self.cursor > 0:
            self.cursor -= 1

    def action_cursor_right(self) -> None:
        state = self.app.controller.state
        if state and state.pack and self.cursor < len(state.pack.cards) - 1:
            self.cursor += 1

    def action_cursor_up(self) -> None:
        if self.cursor > 0:
            self.cursor -= 1

    def action_cursor_down(self) -> None:
        state = self.app.controller.state
        if state and state.pack and self.cursor < len(state.pack.cards) - 1:
            self.cursor += 1

    def action_claim(self) -> None:
        ctrl = self.app.controller
        state = ctrl.state
        if state is None or state.pack is None:
            return

        if not state.pack.cards:
            return

        if self.cursor >= len(state.pack.cards):
            self.cursor = max(0, len(state.pack.cards) - 1)

        try:
            claimed = ctrl.claim_from_pack(self.cursor)
            name = state.data.centers.get(claimed.center_key, {}).get("name", claimed.center_key)
            self.notify(f"Claimed {name}")
        except Exception as e:
            self.notify(f"Cannot claim: {e}", severity="error")
            return

        # Check if pack is done
        if state.pack is None or state.pack.choices_remaining <= 0:
            ctrl.close_current_pack(skipped=False)
            self.app.pop_screen()
        else:
            self.cursor = min(self.cursor, max(0, len(state.pack.cards) - 1))
            self._refresh_display()

    def action_skip_pack(self) -> None:
        ctrl = self.app.controller
        ctrl.close_current_pack(skipped=True)
        self.app.pop_screen()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        match event.button.id:
            case "claim-btn":
                self.action_claim()
            case "skip-btn":
                self.action_skip_pack()
