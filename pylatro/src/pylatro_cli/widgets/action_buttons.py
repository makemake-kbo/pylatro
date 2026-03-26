"""ActionButtons — Play Hand, Sort, Discard buttons."""

from __future__ import annotations

from textual.containers import Horizontal
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Button


class ActionButtons(Widget):
    """Play / Sort / Discard button row."""

    DEFAULT_CSS = """
    ActionButtons {
        height: 3;
        width: 100%;
        layout: horizontal;
        align: center middle;
    }
    ActionButtons Button {
        min-width: 14;
        margin: 0 1;
    }
    """

    class PlayPressed(Message):
        pass

    class DiscardPressed(Message):
        pass

    class SortPressed(Message):
        pass

    def compose(self):
        yield Button("Play Hand", id="btn-play", variant="primary")
        yield Button("Sort", id="btn-sort", variant="default")
        yield Button("Discard", id="btn-discard", variant="error")

    def set_states(self, *, can_play: bool, can_discard: bool, has_selection: bool) -> None:
        play_btn = self.query_one("#btn-play", Button)
        disc_btn = self.query_one("#btn-discard", Button)

        play_btn.disabled = not (can_play and has_selection)
        disc_btn.disabled = not (can_discard and has_selection)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        match event.button.id:
            case "btn-play":
                self.post_message(self.PlayPressed())
            case "btn-discard":
                self.post_message(self.DiscardPressed())
            case "btn-sort":
                self.post_message(self.SortPressed())
