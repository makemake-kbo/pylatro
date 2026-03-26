"""CardRow — horizontal scrollable container of PlayingCardWidgets."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.binding import Binding
from textual.containers import HorizontalScroll
from textual.message import Message
from textual.reactive import reactive
from textual.widget import Widget

from .card import PlayingCardWidget

if TYPE_CHECKING:
    from pylatro.models import PlayingCard


class CardRow(Widget):
    """Horizontal row of PlayingCardWidgets with cursor navigation and selection."""

    DEFAULT_CSS = """
    CardRow {
        height: auto;
        width: 100%;
    }
    CardRow HorizontalScroll {
        height: auto;
        width: 100%;
    }
    """

    BINDINGS = [
        Binding("h", "cursor_left", "Left", show=False),
        Binding("l", "cursor_right", "Right", show=False),
        Binding("left", "cursor_left", "Left", show=False),
        Binding("right", "cursor_right", "Right", show=False),
        Binding("space", "toggle_select", "Select", show=False),
        Binding("enter", "toggle_select", "Select", show=False),
    ]

    cursor_pos: reactive[int] = reactive(0)
    selected_indices: reactive[frozenset[int]] = reactive(frozenset())

    class SelectionChanged(Message):
        def __init__(self, indices: frozenset[int]) -> None:
            super().__init__()
            self.indices = indices

    can_focus = True

    def __init__(self, cards: list[PlayingCard] | None = None, max_select: int = 5, **kwargs) -> None:
        super().__init__(**kwargs)
        self._cards: list[PlayingCard] = cards or []
        self.max_select = max_select
        self._scroll = HorizontalScroll()

    def compose(self):
        with self._scroll:
            for i, card in enumerate(self._cards):
                yield PlayingCardWidget(card, index=i)

    def update_cards(self, cards: list[PlayingCard]) -> None:
        self._cards = cards
        self.selected_indices = frozenset()
        self.cursor_pos = min(self.cursor_pos, max(0, len(cards) - 1))
        self._scroll.remove_children()
        for i, card in enumerate(cards):
            self._scroll.mount(PlayingCardWidget(card, index=i))
        self._update_highlights()

    def update_selection(self, indices: frozenset[int]) -> None:
        self.selected_indices = indices
        self._update_card_states()

    def clear_selection(self) -> None:
        self.selected_indices = frozenset()
        self._update_card_states()

    def _card_widgets(self) -> list[PlayingCardWidget]:
        return list(self._scroll.query(PlayingCardWidget))

    def _update_highlights(self) -> None:
        for w in self._card_widgets():
            w.highlighted = w.index == self.cursor_pos
        self._update_card_states()

    def _update_card_states(self) -> None:
        for w in self._card_widgets():
            w.selected = w.index in self.selected_indices
            w.highlighted = w.index == self.cursor_pos

    def watch_cursor_pos(self, value: int) -> None:
        self._update_highlights()

    def watch_selected_indices(self, value: frozenset[int]) -> None:
        self._update_card_states()
        self.post_message(self.SelectionChanged(value))

    def action_cursor_left(self) -> None:
        if self._cards and self.cursor_pos > 0:
            self.cursor_pos -= 1

    def action_cursor_right(self) -> None:
        if self._cards and self.cursor_pos < len(self._cards) - 1:
            self.cursor_pos += 1

    def action_toggle_select(self) -> None:
        if not self._cards:
            return
        idx = self.cursor_pos
        if idx in self.selected_indices:
            self.selected_indices = self.selected_indices - {idx}
        elif len(self.selected_indices) < self.max_select:
            self.selected_indices = self.selected_indices | {idx}

    def on_playing_card_widget_clicked(self, event: PlayingCardWidget.Clicked) -> None:
        self.cursor_pos = event.index
        self.action_toggle_select()
