"""ConsumableTargetScreen, modal for targeting consumable effects."""

from __future__ import annotations

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.screen import Screen
from textual.widgets import Button, Static

from ..theme import BALATRO_PALETTE, SUIT_SYMBOLS


class ConsumableTargetScreen(Screen):
    """Modal: select target cards/jokers for a consumable."""

    BINDINGS = [
        Binding("h", "cursor_left", "Left", show=False),
        Binding("l", "cursor_right", "Right", show=False),
        Binding("left", "cursor_left", "Left", show=False),
        Binding("right", "cursor_right", "Right", show=False),
        Binding("space", "toggle_select", "Toggle", show=False),
        Binding("shift+left", "move_left", "Move card left"),
        Binding("shift+right", "move_right", "Move card right"),
        Binding("enter", "confirm", "Confirm"),
        Binding("escape", "cancel", "Cancel"),
    ]

    DEFAULT_CSS = """
    ConsumableTargetScreen {
        align: center middle;
        background: rgba(0, 0, 0, 0.7);
    }
    #target-container {
        width: 70;
        height: auto;
        max-height: 80%;
        background: #1c1c3a;
        border: heavy #5c5c8a;
        padding: 2;
    }
    #target-header {
        text-align: center;
        margin-bottom: 1;
    }
    #target-cards-display {
        height: auto;
        min-height: 4;
    }
    #target-buttons {
        layout: horizontal;
        height: 3;
        align: center middle;
        margin-top: 1;
    }
    #target-buttons Button {
        margin: 0 1;
    }
    """

    cursor: reactive[int] = reactive(0)
    selected: reactive[frozenset[int]] = reactive(frozenset())

    def __init__(self, consumable_index: int | None = None, *, pack_index: int | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.consumable_index = consumable_index
        self.pack_index = pack_index

    def compose(self) -> ComposeResult:
        with Vertical(id="target-container"):
            yield Static("", id="target-header")
            yield Static("", id="target-cards-display")
            with Horizontal(id="target-buttons"):
                yield Button("Confirm", id="confirm-btn", variant="primary")
                yield Button("Cancel", id="cancel-btn", variant="default")

    def on_mount(self) -> None:
        self._refresh_display()

    def _refresh_display(self) -> None:
        ctrl = self.app.controller
        state = ctrl.state
        if state is None:
            return

        consumable = (
            state.pack.cards[self.pack_index]
            if self.pack_index is not None
            else state.consumables[self.consumable_index]
        )
        center = state.data.centers.get(consumable.center_key, {})
        name = center.get("name", consumable.center_key)
        desc = center.get("description", "")

        self.query_one("#target-header", Static).update(f"Use {name}\n{desc}\n\nSelect target cards:")

        # Show hand cards as targets
        t = Text()
        for i, card in enumerate(state.hand_cards):
            sym = SUIT_SYMBOLS.get(card.suit, "?")
            is_selected = i in self.selected
            is_cursor = i == self.cursor

            prefix = " >> " if is_cursor else "    "
            marker = "[X]" if is_selected else "[ ]"

            if is_cursor:
                t.append(prefix, style=BALATRO_PALETTE["card_selected"])
            else:
                t.append(prefix)

            t.append(f"{marker} ", style=BALATRO_PALETTE["card_selected"] if is_selected else "#95a5a6")
            t.append(f"{card.rank}{sym}", style="#ecf0f1")
            t.append("\n")

        self.query_one("#target-cards-display", Static).update(t)

    def watch_cursor(self, value: int) -> None:
        if self.is_mounted:
            self._refresh_display()

    def watch_selected(self, value: frozenset[int]) -> None:
        if self.is_mounted:
            self._refresh_display()

    def action_cursor_left(self) -> None:
        if self.cursor > 0:
            self.cursor -= 1

    def action_cursor_right(self) -> None:
        state = self.app.controller.state
        if state and self.cursor < len(state.hand_cards) - 1:
            self.cursor += 1

    def action_toggle_select(self) -> None:
        if self.cursor in self.selected:
            self.selected = self.selected - {self.cursor}
        else:
            self.selected = self.selected | {self.cursor}

    def _move_card(self, offset: int) -> None:
        state = self.app.controller.state
        destination = self.cursor + offset
        if state is None or not 0 <= destination < len(state.hand_cards):
            return
        selected_cards = {id(state.hand_cards[index]) for index in self.selected}
        moved = state.hand_cards.pop(self.cursor)
        state.hand_cards.insert(destination, moved)
        self.selected = frozenset(i for i, card in enumerate(state.hand_cards) if id(card) in selected_cards)
        self.cursor = destination
        self._refresh_display()

    def action_move_left(self) -> None:
        self._move_card(-1)

    def action_move_right(self) -> None:
        self._move_card(1)

    def action_confirm(self) -> None:
        ctrl = self.app.controller
        state = ctrl.state
        if state is None:
            return

        hand_targets = tuple(sorted(self.selected))

        if self.pack_index is not None:
            try:
                ctrl.claim_from_pack(self.pack_index, hand_targets=hand_targets)
            except ValueError as error:
                self.notify(str(error), severity="warning")
                return
            self.dismiss(True)
            return

        if not ctrl.can_use(self.consumable_index, hand_targets=hand_targets):
            self.notify("Cannot use consumable with this selection", severity="warning")
            return

        result = ctrl.use_consumable_on(self.consumable_index, hand_targets=hand_targets)
        name = state.data.centers.get(result.consumable_key, {}).get("name", result.consumable_key)
        self.notify(f"Used {name}")
        self.dismiss(True)

    def action_cancel(self) -> None:
        self.dismiss(False)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        match event.button.id:
            case "confirm-btn":
                self.action_confirm()
            case "cancel-btn":
                self.action_cancel()
