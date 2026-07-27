"""ShopScreen, buy cards, vouchers, and booster packs."""

from __future__ import annotations

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.screen import Screen
from textual.widgets import Button, Static

from ..controller import GameController
from ..joker_order import move_owned_joker
from ..theme import BALATRO_PALETTE
from ..widgets.consumable_bar import ConsumableBar
from ..widgets.joker_bar import JokerBar


class ShopScreen(Screen):
    """Shop: buy cards, vouchers, boosters. Sell jokers/consumables."""

    BINDINGS = [
        Binding("h", "cursor_left", "Left", show=False),
        Binding("l", "cursor_right", "Right", show=False),
        Binding("left", "cursor_left", "Left", show=False),
        Binding("right", "cursor_right", "Right", show=False),
        Binding("k", "cursor_up", "Up", show=False),
        Binding("j", "cursor_down", "Down", show=False),
        Binding("up", "cursor_up", "Up", show=False),
        Binding("down", "cursor_down", "Down", show=False),
        Binding("enter", "buy_item", "Buy"),
        Binding("r", "reroll", "Reroll"),
        Binding("n", "next_round", "Next Round"),
        Binding("f2", "focus_jokers", "Manage Jokers", show=False),
        Binding("tab", "focus_next", "Next Zone", show=False),
        Binding("shift+tab", "focus_previous", "Prev Zone", show=False),
    ]

    DEFAULT_CSS = """
    ShopScreen {
        layout: vertical;
    }
    #shop-header {
        height: 3;
        text-align: center;
        padding: 1;
        color: #ffc107;
        text-style: bold;
    }
    #shop-main {
        height: 1fr;
        layout: horizontal;
    }
    #shop-cards-area {
        width: 1fr;
        layout: vertical;
        padding: 1;
    }
    #shop-right {
        width: 24;
        background: #1c1c3a;
        border-left: solid #5c5c8a;
        padding: 1;
    }
    .shop-zone-label {
        height: 1;
        color: #95a5a6;
        margin: 1 0 0 0;
    }
    #shop-items-display {
        height: auto;
        min-height: 8;
    }
    #shop-boosters-display {
        height: auto;
        min-height: 4;
    }
    #shop-vouchers-display {
        height: auto;
        min-height: 4;
    }
    #shop-buttons {
        height: 3;
        layout: horizontal;
        align: center middle;
    }
    #shop-buttons Button {
        margin: 0 1;
    }
    """

    # Flat cursor across all shop items (cards, then boosters, then vouchers)
    cursor: reactive[int] = reactive(0)

    def compose(self) -> ComposeResult:
        ctrl = self._ctrl()
        state = ctrl.state
        dollars = state.dollars if state else 0

        yield Static(f"Shop, ${dollars}", id="shop-header")
        yield JokerBar(id="shop-joker-bar")
        with Vertical(id="shop-cards-area"):
            yield Static("Cards", classes="shop-zone-label")
            yield Static("", id="shop-items-display")
            yield Static("Boosters", classes="shop-zone-label")
            yield Static("", id="shop-boosters-display")
            yield Static("Vouchers", classes="shop-zone-label")
            yield Static("", id="shop-vouchers-display")
        with Horizontal(id="shop-buttons"):
            yield Button("Reroll", id="reroll-btn", variant="warning")
            yield Button("Next Round", id="next-btn", variant="primary")
        yield ConsumableBar(id="shop-consumable-bar")

    def on_mount(self) -> None:
        self._refresh_display()

    def _ctrl(self) -> GameController:
        return self.app.controller

    def _refresh_display(self) -> None:
        ctrl = self._ctrl()
        state = ctrl.state
        if state is None:
            return

        # Header
        self.query_one("#shop-header", Static).update(f"Shop, ${state.dollars}")

        # Cards
        items_text = self._render_shop_items(state.shop.cards, state.dollars, 0)
        self.query_one("#shop-items-display", Static).update(items_text)

        # Boosters
        boosters_text = self._render_shop_items(state.shop.boosters, state.dollars, 1)
        self.query_one("#shop-boosters-display", Static).update(boosters_text)

        # Vouchers
        vouchers_text = self._render_shop_items(state.shop.vouchers, state.dollars, 2)
        self.query_one("#shop-vouchers-display", Static).update(vouchers_text)

        # Reroll button cost
        reroll_cost = state.current_round.reroll_cost
        self.query_one("#reroll-btn", Button).label = f"Reroll ${reroll_cost}"

        # Joker / consumable bars
        self.query_one("#shop-joker-bar", JokerBar).update_jokers(state.jokers, state.data)
        self.query_one("#shop-consumable-bar", ConsumableBar).update_consumables(state.consumables, state.data)

    def _all_items(self) -> list:
        """Flat list of all shop items across zones."""
        state = self._ctrl().state
        if state is None:
            return []
        return list(state.shop.cards) + list(state.shop.boosters) + list(state.shop.vouchers)

    def _zone_and_local_index(self) -> tuple[int, int]:
        """Convert flat cursor to (zone, local_index)."""
        state = self._ctrl().state
        if state is None:
            return 0, 0
        n_cards = len(state.shop.cards)
        n_boosters = len(state.shop.boosters)
        pos = self.cursor
        if pos < n_cards:
            return 0, pos
        pos -= n_cards
        if pos < n_boosters:
            return 1, pos
        pos -= n_boosters
        return 2, pos

    def _render_shop_items(self, items: list, dollars: int, zone_id: int) -> Text:
        if not items:
            return Text("  (empty)", style=BALATRO_PALETTE["text_muted"])

        ctrl = self._ctrl()
        state = ctrl.state
        t = Text()

        current_zone, local_cursor = self._zone_and_local_index()

        for i, item in enumerate(items):
            center = state.data.centers.get(item.center_key, {})
            name = center.get("name", item.center_key)
            affordable = dollars >= item.cost

            is_focused = (zone_id == current_zone) and (i == local_cursor)

            if is_focused:
                t.append(" >> ", style=BALATRO_PALETTE["card_selected"])
            else:
                t.append("    ")

            # Type color
            type_colors = {"Joker": "#ffc107", "Tarot": "#9c27b0", "Planet": "#2196f3", "Spectral": "#b0bec5"}
            tc = type_colors.get(item.card_type, "#95a5a6")
            t.append(f"[{item.card_type}] ", style=tc)
            t.append(f"{name}", style="bold #ecf0f1" if affordable else "dim #636e72")
            t.append(f"  ${item.cost}", style=BALATRO_PALETTE["gold"] if affordable else "#e74c3c")
            t.append("\n")

        return t

    def watch_cursor(self, value: int) -> None:
        if self.is_mounted:
            self._refresh_display()

    def action_cursor_left(self) -> None:
        if self.cursor > 0:
            self.cursor -= 1

    def action_cursor_right(self) -> None:
        all_items = self._all_items()
        if self.cursor < len(all_items) - 1:
            self.cursor += 1

    def action_cursor_up(self) -> None:
        if self.cursor > 0:
            self.cursor -= 1

    def action_cursor_down(self) -> None:
        all_items = self._all_items()
        if self.cursor < len(all_items) - 1:
            self.cursor += 1

    def action_buy_item(self) -> None:
        ctrl = self._ctrl()
        state = ctrl.state
        if state is None:
            return

        all_items = self._all_items()
        if not all_items or self.cursor >= len(all_items):
            return

        item = all_items[self.cursor]
        zone, local_idx = self._zone_and_local_index()

        if state.dollars < item.cost:
            self.notify("Not enough money!", severity="warning")
            return

        try:
            if zone == 2:  # Voucher
                ctrl.buy_voucher(item.center_key)
            elif zone == 1:  # Booster
                pack = ctrl.open_pack(local_idx)
                from .booster_pack import BoosterPackScreen

                self.app.push_screen(BoosterPackScreen())
                return
            else:
                ctrl.buy_card(local_idx)

            self.notify(f"Bought {state.data.centers.get(item.center_key, {}).get('name', item.center_key)}")
        except Exception as e:
            self.notify(f"Cannot buy: {e}", severity="error")

        # Clamp cursor after item removal
        all_items = self._all_items()
        if self.cursor >= len(all_items):
            self.cursor = max(0, len(all_items) - 1)

        self._refresh_display()

    def action_reroll(self) -> None:
        ctrl = self._ctrl()
        state = ctrl.state
        if state is None:
            return

        if state.dollars < state.current_round.reroll_cost:
            self.notify("Not enough money to reroll!", severity="warning")
            return

        ctrl.reroll()
        self._refresh_display()

    def action_next_round(self) -> None:
        ctrl = self._ctrl()
        ctrl.leave_shop()

        from .blind_select import BlindSelectScreen

        self.app.switch_screen(BlindSelectScreen())

    def action_focus_jokers(self) -> None:
        self.query_one("#shop-joker-bar", JokerBar).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        match event.button.id:
            case "reroll-btn":
                self.action_reroll()
            case "next-btn":
                self.action_next_round()

    def on_joker_bar_sell_requested(self, event: JokerBar.SellRequested) -> None:
        self._ctrl().sell_joker(event.index)
        self._refresh_display()

    def on_joker_bar_move_requested(self, event: JokerBar.MoveRequested) -> None:
        state = self._ctrl().state
        if state is not None:
            move_owned_joker(state, event.index, event.offset)
            self._refresh_display()

    def on_consumable_bar_sell_requested(self, event: ConsumableBar.SellRequested) -> None:
        self._ctrl().sell_consumable(event.index)
        self._refresh_display()
