"""Widget tests using Textual pilot."""

import asyncio

import pytest

from pylatro import load_game_data, create_run_state
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
        async with app.run_test(size=(80, 24)) as pilot:
            row = app.query_one("#row", CardRow)
            assert len(row._cards) == 3

            # Update cards
            new_cards = [PlayingCard(front_key="C_J", suit="Clubs", rank="J")]
            row.update_cards(new_cards)
            assert len(row._cards) == 1
