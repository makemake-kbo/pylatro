"""JokerBar, horizontal row of joker slots."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.text import Text
from textual.binding import Binding
from textual.message import Message
from textual.reactive import reactive
from textual.widget import Widget

from ..theme import BALATRO_PALETTE

if TYPE_CHECKING:
    from pylatro import GameData
    from pylatro.models import JokerInstance


class JokerBar(Widget):
    """Horizontal bar displaying owned jokers."""

    DEFAULT_CSS = """
    JokerBar {
        height: 3;
        width: 100%;
        layout: horizontal;
        padding: 0 1;
    }
    """

    BINDINGS = [
        Binding("left", "cursor_left", "Left", show=False),
        Binding("right", "cursor_right", "Right", show=False),
        Binding("h", "cursor_left", "Left", show=False),
        Binding("l", "cursor_right", "Right", show=False),
        Binding("s", "sell", "Sell", show=False),
    ]

    can_focus = True
    cursor_pos: reactive[int] = reactive(0)

    class SellRequested(Message):
        def __init__(self, index: int) -> None:
            super().__init__()
            self.index = index

    def __init__(self, data: GameData | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._jokers: list[JokerInstance] = []
        self._data = data

    def update_jokers(self, jokers: list[JokerInstance], data: GameData) -> None:
        self._jokers = jokers
        self._data = data
        self.cursor_pos = min(self.cursor_pos, max(0, len(jokers) - 1))
        self.refresh()

    def render(self) -> Text:
        if not self._jokers:
            return Text("  [No Jokers]", style=BALATRO_PALETTE["text_muted"])

        t = Text()
        for i, joker in enumerate(self._jokers):
            name = joker.center_key
            if self._data:
                center = self._data.centers.get(joker.center_key, {})
                name = center.get("name", joker.center_key)

            # Truncate to fit
            display = name[:10]
            if i == self.cursor_pos and self.has_focus:
                t.append(f"[{display}]", style=f"bold {BALATRO_PALETTE['card_selected']}")
            else:
                style = "dim" if joker.debuff else BALATRO_PALETTE["text_primary"]
                t.append(f" {display} ", style=style)
            t.append(" ")
        return t

    def action_cursor_left(self) -> None:
        if self._jokers and self.cursor_pos > 0:
            self.cursor_pos -= 1
            self.refresh()

    def action_cursor_right(self) -> None:
        if self._jokers and self.cursor_pos < len(self._jokers) - 1:
            self.cursor_pos += 1
            self.refresh()

    def action_sell(self) -> None:
        if self._jokers:
            self.post_message(self.SellRequested(self.cursor_pos))
