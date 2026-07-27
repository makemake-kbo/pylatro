"""HandPlayScreen, the core gameplay screen."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Button, Static

from ..controller import GameController, GamePhase
from ..joker_order import move_owned_joker
from ..widgets.action_buttons import ActionButtons
from ..widgets.card_row import CardRow
from ..widgets.consumable_bar import ConsumableBar
from ..widgets.joker_bar import JokerBar
from ..widgets.sidebar import InfoSidebar


class HandPlayScreen(Screen):
    """Core gameplay: select cards, play/discard, beat the blind."""

    BINDINGS = [
        Binding("p", "play_hand", "Play Hand", show=True),
        Binding("d", "do_discard", "Discard", show=True),
        Binding("s", "sort_hand", "Sort", show=False),
        Binding("i", "run_info", "Run Info", show=False),
        Binding("f2", "focus_jokers", "Manage Jokers", show=False),
        Binding("tab", "focus_next", "Next Zone", show=False),
        Binding("shift+tab", "focus_previous", "Prev Zone", show=False),
        Binding("1", "use_consumable_1", "Use C1", show=False),
        Binding("2", "use_consumable_2", "Use C2", show=False),
        Binding("3", "use_consumable_3", "Use C3", show=False),
        Binding("4", "use_consumable_4", "Use C4", show=False),
        Binding("5", "use_consumable_5", "Use C5", show=False),
    ]

    DEFAULT_CSS = """
    HandPlayScreen {
        layout: horizontal;
    }
    #center-area {
        width: 1fr;
        layout: vertical;
        align: center middle;
    }
    #hand-label {
        height: 1;
        text-align: center;
        color: #f1c40f;
        text-style: bold;
    }
    #card-count {
        height: 1;
        text-align: center;
        color: #95a5a6;
    }
    #right-sidebar {
        width: 16;
        background: #1c1c3a;
        border-left: solid #5c5c8a;
        padding: 1;
    }
    """

    def compose(self) -> ComposeResult:
        yield InfoSidebar(id="info-sidebar")
        with Vertical(id="center-area"):
            yield JokerBar(id="joker-bar")
            yield Static("", id="hand-label")
            yield CardRow(id="card-row")
            yield Static("", id="card-count")
            yield ActionButtons(id="action-buttons")
        with Vertical(id="right-sidebar"):
            yield Static("", id="right-info")
            yield ConsumableBar(id="consumable-bar")

    def on_mount(self) -> None:
        self._refresh_display()
        # Focus the card row
        self.query_one("#card-row", CardRow).focus()

    def _ctrl(self) -> GameController:
        return self.app.controller

    def _refresh_display(self) -> None:
        ctrl = self._ctrl()
        state = ctrl.state
        if state is None:
            return

        # Re-apply the user's chosen sort so playing/drawing never overrides it.
        self._apply_sort(state)

        # Update card row
        card_row = self.query_one("#card-row", CardRow)
        card_row.update_cards(state.hand_cards)

        # Update sidebar
        blind = state.round_resets.blind
        blind_name = blind.get("name", "???") if blind else "???"

        sidebar = self.query_one("#info-sidebar", InfoSidebar)
        sidebar.update_info(
            blind_name=blind_name,
            blind_target=ctrl.blind_target(),
            round_score=ctrl.round_score,
            hands_left=state.current_round.hands_left,
            discards_left=state.current_round.discards_left,
            dollars=state.dollars,
            ante=state.round_resets.ante,
            round_num=state.round,
        )

        # Update joker bar
        joker_bar = self.query_one("#joker-bar", JokerBar)
        joker_bar.update_jokers(state.jokers, state.data)

        # Update consumable bar
        cons_bar = self.query_one("#consumable-bar", ConsumableBar)
        cons_bar.update_consumables(state.consumables, state.data)

        # Card count
        deck_count = len(state.draw_pile)
        discard_count = len(state.discard_pile)
        self.query_one("#card-count", Static).update(
            f"Deck: {deck_count}  |  Discard: {discard_count}"
        )

        # Right sidebar info
        right = self.query_one("#right-info", Static)
        right.update(
            f"Ante {state.round_resets.ante}\n"
            f"Round {state.round}\n"
            f"Seed: {state.seed}"
        )

        # Update action button states
        card_row = self.query_one("#card-row", CardRow)
        actions = self.query_one("#action-buttons", ActionButtons)
        actions.set_states(
            can_play=state.current_round.hands_left > 0,
            can_discard=state.current_round.discards_left > 0,
            has_selection=len(card_row.selected_indices) > 0,
        )

        # Reflect the active sort mode on the Sort button.
        self.query_one("#btn-sort", Button).label = f"Sort ({self.app.sort_mode})"

    def _update_hand_preview(self) -> None:
        ctrl = self._ctrl()
        card_row = self.query_one("#card-row", CardRow)
        indices = sorted(card_row.selected_indices)

        sidebar = self.query_one("#info-sidebar", InfoSidebar)

        if not indices:
            sidebar.hand_name = ""
            sidebar.hand_level = 0
            sidebar.chips = 0
            sidebar.mult = 0
            sidebar.refresh()

            self.query_one("#hand-label", Static).update("")
            return

        result = ctrl.hand_evaluation(indices)
        if result:
            hand_name, display_name, _scoring = result
            chips, mult = ctrl.hand_chips_mult(hand_name)
            hand_data = ctrl.state.hands.get(hand_name, {}) if ctrl.state else {}
            level = hand_data.get("level", 1)

            sidebar.hand_name = display_name
            sidebar.hand_level = level
            sidebar.chips = chips
            sidebar.mult = mult
            sidebar.refresh()

            self.query_one("#hand-label", Static).update(
                f"{display_name} (Lv.{level}), {chips} × {mult}"
            )

        # Update action buttons
        state = ctrl.state
        actions = self.query_one("#action-buttons", ActionButtons)
        actions.set_states(
            can_play=state.current_round.hands_left > 0 if state else False,
            can_discard=state.current_round.discards_left > 0 if state else False,
            has_selection=len(indices) > 0,
        )

    def on_card_row_selection_changed(self, event: CardRow.SelectionChanged) -> None:
        self._update_hand_preview()

    def action_play_hand(self) -> None:
        ctrl = self._ctrl()
        state = ctrl.state
        if state is None or state.current_round.hands_left <= 0:
            return

        card_row = self.query_one("#card-row", CardRow)
        indices = sorted(card_row.selected_indices)
        if not indices:
            self.notify("Select cards to play", severity="warning")
            return

        result = ctrl.play_selected(indices)
        self.notify(
            f"{result.score.display_name}: {result.score.chips} × {result.score.mult} = {result.score.total:,}",
            title="Hand Played",
        )

        if ctrl.blind_beaten():
            self._transition_to_shop()
        elif ctrl.phase == GamePhase.GAME_OVER:
            self._transition_to_game_over()
        else:
            self._refresh_display()

    def action_do_discard(self) -> None:
        ctrl = self._ctrl()
        state = ctrl.state
        if state is None or state.current_round.discards_left <= 0:
            return

        card_row = self.query_one("#card-row", CardRow)
        indices = sorted(card_row.selected_indices)
        if not indices:
            self.notify("Select cards to discard", severity="warning")
            return

        ctrl.discard_selected(indices)
        self._refresh_display()

    def _apply_sort(self, state) -> None:
        """Order ``state.hand_cards`` by the persisted UI sort mode."""
        rank_order = {"2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8,
                      "9": 9, "10": 10, "J": 11, "Q": 12, "K": 13, "A": 14}
        if self.app.sort_mode == "suit":
            # Group by suit, low rank first within each suit.
            state.hand_cards.sort(key=lambda c: (c.suit, rank_order.get(c.rank, 0)))
        else:
            # High rank first, matching the engine's default ordering.
            state.hand_cards.sort(key=lambda c: rank_order.get(c.rank, 0), reverse=True)

    def action_sort_hand(self) -> None:
        ctrl = self._ctrl()
        state = ctrl.state
        if state is None:
            return
        # Toggle between the two modes; the refresh re-applies the sort.
        self.app.sort_mode = "suit" if self.app.sort_mode == "rank" else "rank"
        self.notify(f"Sorting by {self.app.sort_mode}")
        self._refresh_display()

    def action_run_info(self) -> None:
        from .run_info import RunInfoScreen

        self.app.push_screen(RunInfoScreen())

    def action_focus_jokers(self) -> None:
        self.query_one("#joker-bar", JokerBar).focus()

    def _use_consumable(self, idx: int) -> None:
        ctrl = self._ctrl()
        state = ctrl.state
        if state is None or idx >= len(state.consumables):
            return

        consumable = state.consumables[idx]
        center = state.data.centers.get(consumable.center_key, {})
        card_set = center.get("set", "")

        # Check if targeting is needed
        if card_set == "Tarot":
            from .consumable_target import ConsumableTargetScreen

            self.app.push_screen(ConsumableTargetScreen(consumable_index=idx))
        elif ctrl.can_use(idx):
            result = ctrl.use_consumable_on(idx)
            self.notify(f"Used {center.get('name', consumable.center_key)}")
            self._refresh_display()
        else:
            self.notify("Cannot use this consumable right now", severity="warning")

    def action_use_consumable_1(self) -> None:
        self._use_consumable(0)

    def action_use_consumable_2(self) -> None:
        self._use_consumable(1)

    def action_use_consumable_3(self) -> None:
        self._use_consumable(2)

    def action_use_consumable_4(self) -> None:
        self._use_consumable(3)

    def action_use_consumable_5(self) -> None:
        self._use_consumable(4)

    def on_action_buttons_play_pressed(self, event: ActionButtons.PlayPressed) -> None:
        self.action_play_hand()

    def on_action_buttons_discard_pressed(self, event: ActionButtons.DiscardPressed) -> None:
        self.action_do_discard()

    def on_action_buttons_sort_pressed(self, event: ActionButtons.SortPressed) -> None:
        self.action_sort_hand()

    def on_joker_bar_sell_requested(self, event: JokerBar.SellRequested) -> None:
        ctrl = self._ctrl()
        ctrl.sell_joker(event.index)
        self._refresh_display()

    def on_joker_bar_move_requested(self, event: JokerBar.MoveRequested) -> None:
        state = self._ctrl().state
        if state is not None:
            move_owned_joker(state, event.index, event.offset)
            self._refresh_display()

    def on_consumable_bar_sell_requested(self, event: ConsumableBar.SellRequested) -> None:
        ctrl = self._ctrl()
        ctrl.sell_consumable(event.index)
        self._refresh_display()

    def on_consumable_bar_use_requested(self, event: ConsumableBar.UseRequested) -> None:
        self._use_consumable(event.index)

    def _transition_to_shop(self) -> None:
        ctrl = self._ctrl()
        ctrl.cash_out()
        if ctrl.phase == GamePhase.GAME_WON:
            self._transition_to_game_won()
            return
        ctrl.enter_shop()
        from .shop import ShopScreen

        self.app.switch_screen(ShopScreen())

    def _transition_to_game_over(self) -> None:
        from .game_over import GameOverScreen

        self.app.switch_screen(GameOverScreen(won=False))

    def _transition_to_game_won(self) -> None:
        from .game_over import GameOverScreen

        self.app.switch_screen(GameOverScreen(won=True))
