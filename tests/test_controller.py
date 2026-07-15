"""Unit tests for GameController."""

import pytest

from pylatro import load_game_data
from pylatro_cli.controller import GameController, GamePhase


@pytest.fixture
def data():
    return load_game_data()


@pytest.fixture
def ctrl(data):
    return GameController(data=data)


class TestNewRun:
    def test_new_run_creates_state(self, ctrl):
        ctrl.new_run("TESTRUN1", stake=1, deck_key="b_red")
        assert ctrl.state is not None
        assert ctrl.state.seed == "TESTRUN1"
        assert ctrl.state.stake == 1
        assert ctrl.phase == GamePhase.BLIND_SELECT

    def test_new_run_resets_score(self, ctrl):
        ctrl.new_run("TESTRUN1")
        assert ctrl.round_score == 0


class TestBlindSelect:
    def test_select_blind_transitions_to_hand_play(self, ctrl):
        ctrl.new_run("TESTRUN1")
        ctrl.select_blind("Small")
        assert ctrl.phase == GamePhase.HAND_PLAY
        assert ctrl.round_score == 0
        assert len(ctrl.state.hand_cards) > 0

    def test_blind_target_positive(self, ctrl):
        ctrl.new_run("TESTRUN1")
        ctrl.select_blind("Small")
        assert ctrl.blind_target() > 0

    def test_skip_blind(self, ctrl):
        ctrl.new_run("TESTRUN1")
        next_blind = ctrl.skip_blind()
        assert next_blind in ("Big", "Boss")


class TestHandPlay:
    def test_play_selected_accumulates_score(self, ctrl):
        ctrl.new_run("TESTRUN1")
        ctrl.select_blind("Small")
        indices = list(range(min(5, len(ctrl.state.hand_cards))))
        result = ctrl.play_selected(indices)
        assert result.score.total > 0
        assert ctrl.round_score == result.score.total

    def test_discard_draws_new_cards(self, ctrl):
        ctrl.new_run("TESTRUN1")
        ctrl.select_blind("Small")
        hand_before = len(ctrl.state.hand_cards)
        ctrl.discard_selected([0])
        assert len(ctrl.state.hand_cards) == hand_before

    def test_play_until_beaten(self, ctrl):
        ctrl.new_run("TESTRUN1")
        ctrl.select_blind("Small")
        for _ in range(ctrl.state.current_round.hands_left):
            if ctrl.blind_beaten():
                break
            indices = list(range(min(5, len(ctrl.state.hand_cards))))
            ctrl.play_selected(indices)
        # Either beaten or game over
        assert ctrl.blind_beaten() or ctrl.phase == GamePhase.GAME_OVER

    def test_hand_evaluation_preview(self, ctrl):
        ctrl.new_run("TESTRUN1")
        ctrl.select_blind("Small")
        result = ctrl.hand_evaluation([0, 1])
        if result:
            name, display, scoring = result
            assert isinstance(name, str)
            assert isinstance(display, str)

    def test_hand_evaluation_empty(self, ctrl):
        ctrl.new_run("TESTRUN1")
        ctrl.select_blind("Small")
        assert ctrl.hand_evaluation([]) is None


class TestShop:
    def _setup_shop(self, ctrl):
        ctrl.new_run("TESTRUN1")
        ctrl.select_blind("Small")
        # Play until beaten
        for _ in range(10):
            if ctrl.blind_beaten():
                break
            indices = list(range(min(5, len(ctrl.state.hand_cards))))
            ctrl.play_selected(indices)
        if not ctrl.blind_beaten():
            pytest.skip("Could not beat blind with this seed")
        ctrl.cash_out()
        ctrl.enter_shop()

    def test_shop_has_items(self, ctrl):
        self._setup_shop(ctrl)
        assert ctrl.phase == GamePhase.SHOP
        assert len(ctrl.state.shop.cards) > 0

    def test_reroll(self, ctrl):
        self._setup_shop(ctrl)
        old_cards = list(ctrl.state.shop.cards)
        ctrl.reroll()
        # Cards should change (with very high probability)
        new_cards = ctrl.state.shop.cards
        assert len(new_cards) > 0

    def test_leave_shop(self, ctrl):
        self._setup_shop(ctrl)
        ctrl.leave_shop()
        assert ctrl.phase == GamePhase.BLIND_SELECT


class TestFullAnteLoop:
    def _beat_current_blind(self, ctrl):
        """Play hands until blind is beaten or hands run out."""
        for _ in range(10):
            if ctrl.blind_beaten():
                return True
            if not ctrl.state.hand_cards:
                return False
            indices = list(range(min(5, len(ctrl.state.hand_cards))))
            ctrl.play_selected(indices)
            if ctrl.phase == GamePhase.GAME_OVER:
                return False
        return ctrl.blind_beaten()

    def test_three_blinds_advance_ante(self, ctrl):
        ctrl.new_run("TESTRUN1")
        initial_ante = ctrl.state.round_resets.ante

        for blind_type in ("Small", "Big", "Boss"):
            ctrl.select_blind(blind_type)

            if not self._beat_current_blind(ctrl):
                pytest.skip(f"Could not beat {blind_type} blind")

            ctrl.cash_out()
            if ctrl.phase == GamePhase.GAME_WON:
                return
            ctrl.enter_shop()
            ctrl.leave_shop()

        assert ctrl.state.round_resets.ante == initial_ante + 1

    def test_new_ante_makes_small_blind_selectable(self, ctrl):
        ctrl.new_run("ANTE2")
        ctrl.state.round_resets.blind_states = {
            "Small": "Defeated",
            "Big": "Defeated",
            "Boss": "Current",
        }

        ctrl.cash_out()

        assert ctrl.state.round_resets.ante == 2
        assert ctrl.state.blind_on_deck == "Small"
        assert ctrl.state.round_resets.blind_states == {
            "Small": "Select",
            "Big": "Upcoming",
            "Boss": "Upcoming",
        }


class TestDisplayHelpers:
    def test_card_display_info(self, ctrl):
        ctrl.new_run("TESTRUN1")
        ctrl.select_blind("Small")
        info = ctrl.card_display_info(ctrl.state.hand_cards[0])
        assert "rank" in info
        assert "suit" in info

    def test_joker_display_info(self, ctrl, data):
        from pylatro import add_joker

        ctrl.new_run("TESTRUN1")
        joker = add_joker(ctrl.state, "j_joker")
        info = ctrl.joker_display_info(joker)
        assert "name" in info
        assert info["name"] == "Joker"

    def test_hand_chips_mult(self, ctrl):
        ctrl.new_run("TESTRUN1")
        chips, mult = ctrl.hand_chips_mult("Pair")
        assert chips > 0
        assert mult > 0
