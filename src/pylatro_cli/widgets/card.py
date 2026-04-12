"""PlayingCardWidget — 7-wide × 5-tall Rich renderable card."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.text import Text
from textual.reactive import reactive
from textual.widget import Widget
from textual.message import Message

from ..theme import SUIT_COLORS, SUIT_SYMBOLS, EDITION_COLORS, BALATRO_PALETTE

if TYPE_CHECKING:
    from pylatro.models import PlayingCard


CARD_WIDTH = 9
CARD_HEIGHT = 5


class PlayingCardWidget(Widget):
    """A single playing card rendered as a small Rich block."""

    DEFAULT_CSS = """
    PlayingCardWidget {
        width: 9;
        height: 5;
        margin: 0 0;
    }
    """

    selected: reactive[bool] = reactive(False)
    highlighted: reactive[bool] = reactive(False)

    class Clicked(Message):
        def __init__(self, index: int) -> None:
            super().__init__()
            self.index = index

    def __init__(self, card: PlayingCard, index: int = 0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.card = card
        self.index = index

    def render(self) -> Text:
        card = self.card

        if card.face_down:
            return self._render_face_down()

        suit_color = SUIT_COLORS.get(card.suit, "#ecf0f1")
        suit_sym = SUIT_SYMBOLS.get(card.suit, "?")
        rank = card.rank

        # Border characters
        if self.selected:
            tl, tr, bl, br = "┏", "┓", "┗", "┛"
            h, v = "━", "┃"
            border_color = BALATRO_PALETTE["card_selected"]
        elif self.highlighted:
            tl, tr, bl, br = "┌", "┐", "└", "┘"
            h, v = "─", "│"
            border_color = BALATRO_PALETTE["card_highlighted"]
        else:
            tl, tr, bl, br = "┌", "┐", "└", "┘"
            h, v = "─", "│"
            border_color = BALATRO_PALETTE["card_border"]

        # Edition override for border color
        if card.edition_key and card.edition_key in EDITION_COLORS:
            border_color = EDITION_COLORS[card.edition_key]

        style = f"dim" if card.debuff else ""
        inner_w = CARD_WIDTH - 2

        # Build rank display (left-align, max 2 chars)
        rank_display = rank if len(rank) <= 2 else rank[0]

        lines = []
        # Top border
        lines.append(Text(f"{tl}{h * inner_w}{tr}", style=border_color))

        # Row 1: rank + suit top-left
        row1 = f"{rank_display:<2}{' ' * (inner_w - 3)}{suit_sym}"
        line1 = Text(f"{v}", style=border_color)
        line1.append(rank_display, style=f"{suit_color} {style}".strip())
        line1.append(" " * (inner_w - len(rank_display) - 1))
        line1.append(suit_sym, style=f"{suit_color} {style}".strip())
        line1.append(f"{v}", style=border_color)
        lines.append(line1)

        # Row 2: center suit symbol
        center_pad_l = (inner_w - 1) // 2
        center_pad_r = inner_w - 1 - center_pad_l
        line2 = Text(f"{v}", style=border_color)
        line2.append(" " * center_pad_l)
        line2.append(suit_sym, style=f"bold {suit_color} {style}".strip())
        line2.append(" " * center_pad_r)
        line2.append(f"{v}", style=border_color)
        lines.append(line2)

        # Row 3: seal indicator or bottom-right rank
        seal_char = ""
        if card.seal:
            seal_map = {"Red": "R", "Blue": "B", "Gold": "G", "Purple": "P"}
            seal_char = seal_map.get(card.seal, "")

        line3 = Text(f"{v}", style=border_color)
        if seal_char:
            seal_color = BALATRO_PALETTE.get(f"seal_{card.seal.lower()}", suit_color)
            line3.append(seal_char, style=seal_color)
            line3.append(" " * (inner_w - len(rank_display) - 1))
        else:
            line3.append(" " * (inner_w - len(rank_display)))
        line3.append(f"{rank_display:>2}"[-len(rank_display):], style=f"{suit_color} {style}".strip())
        line3.append(f"{v}", style=border_color)
        lines.append(line3)

        # Bottom border
        lines.append(Text(f"{bl}{h * inner_w}{br}", style=border_color))

        result = lines[0]
        for line in lines[1:]:
            result.append("\n")
            result.append(line)
        return result

    def _render_face_down(self) -> Text:
        tl, tr, bl, br = "┌", "┐", "└", "┘"
        h, v = "─", "│"
        bc = BALATRO_PALETTE["card_face_down"]
        inner_w = CARD_WIDTH - 2
        fill = "░" * inner_w

        lines = [Text(f"{tl}{h * inner_w}{tr}", style=bc)]
        for _ in range(CARD_HEIGHT - 2):
            line = Text(f"{v}", style=bc)
            line.append(fill, style=bc)
            line.append(f"{v}", style=bc)
            lines.append(line)
        lines.append(Text(f"{bl}{h * inner_w}{br}", style=bc))

        result = lines[0]
        for line in lines[1:]:
            result.append("\n")
            result.append(line)
        return result

    def on_click(self) -> None:
        self.post_message(self.Clicked(self.index))

    def watch_selected(self, value: bool) -> None:
        self.refresh()

    def watch_highlighted(self, value: bool) -> None:
        self.refresh()
