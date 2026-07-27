"""Selectable, movable card row for owned jokers."""

from __future__ import annotations

from textwrap import wrap
from typing import TYPE_CHECKING

from rich.text import Text
from textual.binding import Binding
from textual.containers import HorizontalScroll
from textual.message import Message
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static

from ..theme import BALATRO_PALETTE, EDITION_COLORS

if TYPE_CHECKING:
    from pylatro import GameData
    from pylatro.models import JokerInstance


JOKER_WIDTH = 11
JOKER_HEIGHT = 7

SCALING_MULT_JOKERS = frozenset(
    {
        "Ceremonial Dagger",
        "Flash Card",
        "Green Joker",
        "Red Card",
        "Ride the Bus",
        "Spare Trousers",
    }
)


class JokerCardWidget(Widget):
    """A compact joker card with stable dimensions and mouse selection."""

    DEFAULT_CSS = """
    JokerCardWidget {
        width: 11;
        height: 7;
        margin: 0 1;
    }
    """

    selected: reactive[bool] = reactive(False)
    active: reactive[bool] = reactive(False)

    class Clicked(Message):
        def __init__(self, index: int) -> None:
            super().__init__()
            self.index = index

    def __init__(self, joker: JokerInstance, name: str, index: int, rarity: int = 1, **kwargs) -> None:
        super().__init__(**kwargs)
        self.joker = joker
        self.joker_name = name
        self.index = index
        self.rarity = rarity

    def render(self) -> Text:
        inner_width = JOKER_WIDTH - 2
        if self.selected and self.active:
            tl, tr, bl, br, horizontal, vertical = "┏", "┓", "┗", "┛", "━", "┃"
            border_color = BALATRO_PALETTE["card_selected"]
        elif self.selected:
            tl, tr, bl, br, horizontal, vertical = "┌", "┐", "└", "┘", "─", "│"
            border_color = BALATRO_PALETTE["card_highlighted"]
        else:
            tl, tr, bl, br, horizontal, vertical = "┌", "┐", "└", "┘", "─", "│"
            border_color = BALATRO_PALETTE["card_border"]

        if self.joker.edition:
            edition = next((key for key, enabled in self.joker.edition.items() if enabled), "")
            border_color = EDITION_COLORS.get(edition, border_color)

        rarity_colors = {1: "#42a5f5", 2: "#27ae60", 3: "#e74c3c", 4: "#9c27b0"}
        name_color = rarity_colors.get(self.rarity, BALATRO_PALETTE["text_primary"])
        name_lines = wrap(self.joker_name, width=inner_width, break_long_words=True, break_on_hyphens=False)[:2]
        name_lines += [""] * (2 - len(name_lines))

        flags = "".join(
            marker
            for enabled, marker in (
                (self.joker.eternal, "E"),
                (self.joker.perishable, "P"),
                (self.joker.rental, "R"),
                (self.joker.debuff, "!"),
            )
            if enabled
        )
        edition_label = ""
        if self.joker.edition:
            edition_label = next((key[:3].upper() for key, enabled in self.joker.edition.items() if enabled), "")
        status = " ".join(part for part in (edition_label, flags) if part) or f"SLOT {self.index + 1}"

        lines = [Text(f"{tl}{horizontal * inner_width}{tr}", style=border_color)]
        lines.append(self._line(vertical, "JOKER".center(inner_width), border_color, BALATRO_PALETTE["text_muted"]))
        for name_line in name_lines:
            lines.append(self._line(vertical, name_line.center(inner_width), border_color, name_color))
        lines.append(self._line(vertical, status.center(inner_width), border_color, BALATRO_PALETTE["text_muted"]))
        lines.append(
            self._line(
                vertical,
                f"sell ${self.joker.sell_cost}".center(inner_width),
                border_color,
                BALATRO_PALETTE["gold"],
            )
        )
        lines.append(Text(f"{bl}{horizontal * inner_width}{br}", style=border_color))

        result = lines[0]
        for line in lines[1:]:
            result.append("\n")
            result.append(line)
        return result

    @staticmethod
    def _line(border: str, content: str, border_color: str, content_color: str) -> Text:
        line = Text(border, style=border_color)
        line.append(content, style=content_color)
        line.append(border, style=border_color)
        return line

    def on_click(self) -> None:
        self.post_message(self.Clicked(self.index))

    def watch_selected(self, value: bool) -> None:
        self.refresh()

    def watch_active(self, value: bool) -> None:
        self.refresh()


class JokerBar(Widget):
    """Owned jokers with cursor navigation, selling, and position controls."""

    DEFAULT_CSS = """
    JokerBar {
        height: 10;
        width: 100%;
        layout: vertical;
    }
    JokerBar > HorizontalScroll {
        height: 7;
        width: 100%;
        align: center middle;
    }
    JokerBar > .joker-detail {
        height: 1;
        padding: 0 1;
        text-align: center;
        color: #ecf0f1;
        text-overflow: ellipsis;
    }
    JokerBar > .joker-shortcuts {
        height: 1;
        text-align: center;
        color: #95a5a6;
    }
    """

    BINDINGS = [
        Binding("left", "cursor_left", "Left", show=False),
        Binding("right", "cursor_right", "Right", show=False),
        Binding("h", "cursor_left", "Left", show=False),
        Binding("l", "cursor_right", "Right", show=False),
        Binding("shift+left", "move_left", "Move left", show=False),
        Binding("shift+right", "move_right", "Move right", show=False),
        Binding("s", "sell", "Sell", show=False),
    ]

    can_focus = True
    cursor_pos: reactive[int] = reactive(0)

    class SellRequested(Message):
        def __init__(self, index: int) -> None:
            super().__init__()
            self.index = index

    class MoveRequested(Message):
        def __init__(self, index: int, offset: int) -> None:
            super().__init__()
            self.index = index
            self.offset = offset

    def __init__(self, data: GameData | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._jokers: list[JokerInstance] = []
        self._data = data
        self._slots = HorizontalScroll()
        self._detail = Static("", classes="joker-detail")
        self._shortcuts = Static("", classes="joker-shortcuts")

    def compose(self):
        yield self._slots
        yield self._detail
        yield self._shortcuts

    def update_jokers(self, jokers: list[JokerInstance], data: GameData) -> None:
        self._jokers = jokers
        self._data = data
        self.cursor_pos = min(self.cursor_pos, max(0, len(jokers) - 1))
        self._slots.remove_children()
        for index, joker in enumerate(jokers):
            center = data.centers.get(joker.center_key, {})
            self._slots.mount(
                JokerCardWidget(
                    joker,
                    center.get("name", joker.center_key),
                    index,
                    int(center.get("rarity", 1) or 1),
                )
            )
        self._refresh_state()

    def _slot_widgets(self) -> list[JokerCardWidget]:
        return list(self._slots.query(JokerCardWidget))

    def _refresh_state(self) -> None:
        for slot in self._slot_widgets():
            slot.selected = slot.index == self.cursor_pos
            slot.active = self.has_focus

        if not self._jokers:
            self._detail.update(Text("No jokers", style=BALATRO_PALETTE["text_muted"]))
            self._shortcuts.update("")
            return

        joker = self._jokers[self.cursor_pos]
        center = self._data.centers.get(joker.center_key, {}) if self._data else {}
        name = center.get("name", joker.center_key)
        summary = self._joker_summary(joker, center)
        self._detail.update(f"{name}  ·  {summary}  ·  sell ${joker.sell_cost}")
        if self.has_focus:
            self._shortcuts.update("←/→ select   ⇧←/→ move   S sell")
        else:
            self._shortcuts.update("F2/click manage jokers")

    @staticmethod
    def _joker_summary(joker: JokerInstance, center: dict) -> str:
        parts: list[str] = []
        name = str(center.get("name", joker.center_key))
        if joker.mult or name in SCALING_MULT_JOKERS:
            parts.append(f"+{joker.mult} Mult")
        if joker.t_mult:
            parts.append(f"+{joker.t_mult} Mult triggered")
        if joker.t_chips:
            parts.append(f"+{joker.t_chips} chips")
        if isinstance(joker.extra, dict):
            if isinstance(joker.extra.get("chips"), (int, float)):
                parts.append(f"+{joker.extra['chips']:g} chips")
            if isinstance(joker.extra.get("mult"), (int, float)):
                parts.append(f"+{joker.extra['mult']:g} Mult")
            if isinstance(joker.extra.get("Xmult"), (int, float)):
                parts.append(f"x{joker.extra['Xmult']:g} Mult")
        if joker.x_mult != 1:
            parts.append(f"x{joker.x_mult:g} Mult")
        if joker.caino_xmult != 1:
            parts.append(f"x{joker.caino_xmult:g} Mult")
        if joker.h_x_mult:
            parts.append(f"x{joker.h_x_mult:g} held Mult")
        dollars = joker.p_dollars + joker.h_dollars
        if dollars:
            parts.append(f"${dollars:+d}")
        return ", ".join(parts) or str(center.get("effect") or "Owned joker")

    def watch_cursor_pos(self, value: int) -> None:
        if self.is_mounted:
            self._refresh_state()

    def on_focus(self) -> None:
        self.call_after_refresh(self._refresh_state)

    def on_blur(self) -> None:
        self.call_after_refresh(self._refresh_state)

    def action_cursor_left(self) -> None:
        if self._jokers and self.cursor_pos > 0:
            self.cursor_pos -= 1

    def action_cursor_right(self) -> None:
        if self._jokers and self.cursor_pos < len(self._jokers) - 1:
            self.cursor_pos += 1

    def _move(self, offset: int) -> None:
        target = self.cursor_pos + offset
        if self._jokers and 0 <= target < len(self._jokers):
            old_index = self.cursor_pos
            self.cursor_pos = target
            self.post_message(self.MoveRequested(old_index, offset))

    def action_move_left(self) -> None:
        self._move(-1)

    def action_move_right(self) -> None:
        self._move(1)

    def action_sell(self) -> None:
        if self._jokers:
            self.post_message(self.SellRequested(self.cursor_pos))

    def on_joker_card_widget_clicked(self, event: JokerCardWidget.Clicked) -> None:
        self.cursor_pos = event.index
        self.focus()
