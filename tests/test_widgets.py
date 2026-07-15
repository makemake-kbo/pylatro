"""Widget tests using Textual pilot."""

import pytest

from pylatro import create_run_state, load_game_data
from pylatro.models import PlayingCard


@pytest.fixture
def data():
    return load_game_data()


class TestPlayingCardWidget:
    def test_render_normal_card(self):
        from pylatro_cli.widgets.card import PlayingCardWidget

        card = PlayingCard(front_key="S_A", suit="Spades", rank="A")
        widget = PlayingCardWidget(card, index=0)
        # Just verify construction works
        assert widget.card.rank == "A"
        assert widget.card.suit == "Spades"
        assert widget.selected is False

        lines = widget.render().plain.splitlines()
        assert len(lines) == 5
        assert all(len(line) == 7 for line in lines)

    def test_render_face_down_card(self):
        from pylatro_cli.widgets.card import PlayingCardWidget

        card = PlayingCard(front_key="S_A", suit="Spades", rank="A", face_down=True)
        widget = PlayingCardWidget(card, index=0)
        assert widget.card.face_down is True

    def test_render_debuffed_card(self):
        from pylatro_cli.widgets.card import PlayingCardWidget

        card = PlayingCard(front_key="H_K", suit="Hearts", rank="K", debuff=True)
        widget = PlayingCardWidget(card, index=0)
        assert widget.card.debuff is True

    def test_render_card_with_seal(self):
        from pylatro_cli.widgets.card import PlayingCardWidget

        card = PlayingCard(front_key="D_Q", suit="Diamonds", rank="Q", seal="Gold")
        widget = PlayingCardWidget(card, index=0)
        assert widget.card.seal == "Gold"

    def test_render_card_with_edition(self):
        from pylatro_cli.widgets.card import PlayingCardWidget

        card = PlayingCard(front_key="C_J", suit="Clubs", rank="J", edition_key="foil")
        widget = PlayingCardWidget(card, index=0)
        assert widget.card.edition_key == "foil"


class TestInfoSidebar:
    def test_update_info(self):
        from pylatro_cli.widgets.sidebar import InfoSidebar

        sidebar = InfoSidebar()
        sidebar.update_info(
            blind_name="Small Blind",
            blind_target=300,
            round_score=150,
            hands_left=3,
            discards_left=2,
            dollars=10,
            ante=1,
            round_num=1,
        )
        assert sidebar.blind_target == 300
        assert sidebar.round_score == 150
        assert sidebar.hands_left == 3


class TestCardRowIntegration:
    @pytest.mark.asyncio
    async def test_card_row_update(self):
        from textual.app import App, ComposeResult

        from pylatro_cli.widgets.card_row import CardRow

        cards = [
            PlayingCard(front_key="S_A", suit="Spades", rank="A"),
            PlayingCard(front_key="H_K", suit="Hearts", rank="K"),
            PlayingCard(front_key="D_Q", suit="Diamonds", rank="Q"),
        ]

        class TestApp(App):
            def compose(self) -> ComposeResult:
                yield CardRow(cards, id="row")

        app = TestApp()
        async with app.run_test(size=(80, 24)):
            row = app.query_one("#row", CardRow)
            assert len(row._cards) == 3

            # Update cards
            new_cards = [PlayingCard(front_key="C_J", suit="Clubs", rank="J")]
            row.update_cards(new_cards)
            assert len(row._cards) == 1


class TestJokerBar:
    def test_joker_card_has_card_proportions(self, data):
        from pylatro import add_joker
        from pylatro_cli.widgets.joker_bar import JokerCardWidget

        state = create_run_state("joker-card", data=data)
        joker = add_joker(state, "j_joker")
        widget = JokerCardWidget(joker, "Joker", 0)

        lines = widget.render().plain.splitlines()
        assert len(lines) == 7
        assert all(len(line) == 11 for line in lines)

    def test_move_owned_joker_keeps_keys_aligned(self, data):
        from pylatro import add_joker
        from pylatro_cli.joker_order import move_owned_joker

        state = create_run_state("joker-order", data=data)
        add_joker(state, "j_joker")
        add_joker(state, "j_blueprint")

        assert move_owned_joker(state, 1, -1) == 0
        assert [joker.center_key for joker in state.jokers] == ["j_blueprint", "j_joker"]
        assert state.joker_keys == ["j_blueprint", "j_joker"]

    def test_scaling_joker_summary_uses_live_chips_and_mult(self, data):
        from pylatro import add_joker
        from pylatro_cli.widgets.joker_bar import JokerBar

        state = create_run_state("joker-scaling", data=data)
        runner = add_joker(state, "j_runner")
        bus = add_joker(state, "j_ride_the_bus")

        runner_center = data.centers[runner.center_key]
        bus_center = data.centers[bus.center_key]
        assert JokerBar._joker_summary(runner, runner_center) == "+0 chips"
        assert JokerBar._joker_summary(bus, bus_center) == "+0 Mult"

        runner.extra["chips"] = 45
        bus.mult = 6
        assert JokerBar._joker_summary(runner, runner_center) == "+45 chips"
        assert JokerBar._joker_summary(bus, bus_center) == "+6 Mult"

    @pytest.mark.asyncio
    async def test_keyboard_selection_move_and_sell_requests(self, data):
        from textual.app import App, ComposeResult
        from textual.widgets import Static

        from pylatro import add_joker
        from pylatro_cli.widgets.joker_bar import JokerBar

        state = create_run_state("joker-controls", data=data)
        add_joker(state, "j_joker")
        add_joker(state, "j_blueprint")

        class TestApp(App):
            moved: tuple[int, int] | None = None
            sold: int | None = None

            def compose(self) -> ComposeResult:
                yield JokerBar(id="jokers")

            def on_mount(self) -> None:
                self.query_one(JokerBar).update_jokers(state.jokers, data)

            def on_joker_bar_move_requested(self, event: JokerBar.MoveRequested) -> None:
                self.moved = (event.index, event.offset)

            def on_joker_bar_sell_requested(self, event: JokerBar.SellRequested) -> None:
                self.sold = event.index

        app = TestApp()
        async with app.run_test(size=(80, 24)) as pilot:
            bar = app.query_one(JokerBar)
            bar.focus()
            await pilot.pause()
            assert "move" in str(app.query_one(".joker-shortcuts", Static).content)
            await pilot.press("right")
            assert bar.cursor_pos == 1

            await pilot.press("shift+left")
            assert bar.cursor_pos == 0
            assert app.moved == (1, -1)

            await pilot.press("s")
            assert app.sold == 0
