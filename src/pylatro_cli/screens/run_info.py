"""RunInfoScreen — modal showing poker hand levels and deck composition."""

from __future__ import annotations

from collections import Counter

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Static

from pylatro.models import POKER_HANDS

from ..theme import BALATRO_PALETTE, SUIT_COLORS, SUIT_SYMBOLS


class RunInfoScreen(Screen):
    """Modal: poker hand levels + deck composition."""

    BINDINGS = [
        Binding("escape", "dismiss_screen", "Close"),
        Binding("i", "dismiss_screen", "Close"),
    ]

    DEFAULT_CSS = """
    RunInfoScreen {
        align: center middle;
        background: rgba(0, 0, 0, 0.7);
    }
    #run-info-container {
        width: 80;
        height: auto;
        max-height: 90%;
        background: #1c1c3a;
        border: heavy #5c5c8a;
        padding: 2;
    }
    #run-info-panels {
        layout: horizontal;
        height: auto;
    }
    #hands-panel {
        width: 1fr;
        padding: 0 1;
    }
    #deck-panel {
        width: 1fr;
        padding: 0 1;
        border-left: solid #5c5c8a;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical(id="run-info-container"):
            yield Static("Run Info  [Esc/i to close]", id="run-info-header")
            with Horizontal(id="run-info-panels"):
                yield Static("", id="hands-panel")
                yield Static("", id="deck-panel")

    def on_mount(self) -> None:
        ctrl = self.app.controller
        state = ctrl.state
        if state is None:
            return

        # Poker hands panel
        hands_text = Text()
        hands_text.append("Poker Hands\n", style="bold #ecf0f1")
        hands_text.append("─" * 35 + "\n")

        for hand_name in POKER_HANDS:
            hand_data = state.hands.get(hand_name, {})
            level = hand_data.get("level", 1)
            chips = hand_data.get("chips", 0)
            mult = hand_data.get("mult", 0)
            l_chips = hand_data.get("l_chips", 0)
            l_mult = hand_data.get("l_mult", 0)
            total_chips = chips + l_chips * (level - 1)
            total_mult = mult + l_mult * (level - 1)
            played = hand_data.get("played", 0)

            hands_text.append(f"{hand_name:<18}", style="#ecf0f1")
            hands_text.append(f"Lv.{level}", style=BALATRO_PALETTE["card_selected"])
            hands_text.append(f"  {total_chips}", style=BALATRO_PALETTE["chips"])
            hands_text.append(" × ", style="#95a5a6")
            hands_text.append(f"{total_mult}", style=BALATRO_PALETTE["mult"])
            hands_text.append(f"  ({played})\n", style="#636e72")

        self.query_one("#hands-panel", Static).update(hands_text)

        # Deck composition panel
        deck_text = Text()
        deck_text.append("Deck Composition\n", style="bold #ecf0f1")
        deck_text.append("─" * 35 + "\n")
        deck_text.append(f"Total cards: {len(state.deck_cards)}\n\n", style="#95a5a6")

        # Count by suit
        suit_counts = Counter(c.suit for c in state.deck_cards)
        for suit in ("Spades", "Hearts", "Diamonds", "Clubs"):
            sym = SUIT_SYMBOLS.get(suit, "?")
            color = SUIT_COLORS.get(suit, "#ecf0f1")
            count = suit_counts.get(suit, 0)
            deck_text.append(f"  {sym} {suit}: ", style=color)
            deck_text.append(f"{count}\n", style="#ecf0f1")

        deck_text.append("\n")

        # Count by rank
        rank_order = ["A", "K", "Q", "J", "10", "9", "8", "7", "6", "5", "4", "3", "2"]
        rank_counts = Counter(c.rank for c in state.deck_cards)
        for rank in rank_order:
            count = rank_counts.get(rank, 0)
            if count > 0:
                deck_text.append(f"  {rank:>3}: ", style="#95a5a6")
                deck_text.append(f"{count}\n", style="#ecf0f1")

        self.query_one("#deck-panel", Static).update(deck_text)

    def action_dismiss_screen(self) -> None:
        self.app.pop_screen()
