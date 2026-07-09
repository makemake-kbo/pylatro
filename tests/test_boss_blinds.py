"""Boss blind selection and finisher (showdown) blind behavior."""

from __future__ import annotations

import pytest

from pylatro import load_game_data
from pylatro.pool import get_new_boss
from pylatro_cli.controller import GameController


@pytest.fixture(scope="module")
def data():
    return load_game_data()


def _state_at_ante(data, seed: int, ante: int, win_ante: int):
    controller = GameController(data=data)
    controller.new_run(str(seed))
    state = controller.state
    state.win_ante = win_ante
    state.round_resets.ante = ante
    return state


FINISHERS = {"bl_final_acorn", "bl_final_bell", "bl_final_heart", "bl_final_leaf", "bl_final_vessel"}


def test_no_finishers_before_ante_8_with_lowered_win_ante(data) -> None:
    """A curriculum win_ante (e.g. 5) must not pull showdown bosses to ante 5."""
    for ante, win_ante in [(5, 5), (4, 4), (5, 8), (7, 8)]:
        seen = {get_new_boss(_state_at_ante(data, seed, ante, win_ante)) for seed in range(25)}
        assert not (seen & FINISHERS), f"finisher boss at ante {ante} (win_ante={win_ante}): {seen & FINISHERS}"


def test_only_finishers_at_ante_8_regardless_of_win_ante(data) -> None:
    for win_ante in (5, 8):
        seen = {get_new_boss(_state_at_ante(data, seed, 8, win_ante)) for seed in range(40)}
        assert seen <= FINISHERS, f"non-finisher at ante 8 (win_ante={win_ante}): {seen - FINISHERS}"
        assert seen == FINISHERS, f"finisher pool incomplete at ante 8: missing {FINISHERS - seen}"


def test_finishers_at_ante_16(data) -> None:
    seen = {get_new_boss(_state_at_ante(data, seed, 16, 8)) for seed in range(40)}
    assert seen == FINISHERS


def test_amber_acorn_does_not_debuff_jokers(data) -> None:
    """Upstream flips/shuffles jokers; they keep scoring. Debuffing them is a bug."""
    from pylatro.flow import _reset_for_blind  # type: ignore[attr-defined]

    controller = GameController(data=data)
    controller.new_run("42")
    state = controller.state
    from pylatro.runtime import add_joker

    for key in ("j_joker", "j_greedy_joker", "j_lusty_joker"):
        add_joker(state, key)
    state.round_resets.blind = dict(data.blinds["bl_final_acorn"])

    _reset_for_blind(state, "Boss")
    assert all(not joker.debuff for joker in state.jokers), "Amber Acorn must not debuff jokers"


def test_verdant_leaf_sell_joker_disables_blind(data) -> None:
    """Selling any joker under Verdant Leaf disables the blind and lifts card debuffs."""
    from pylatro.runtime import add_joker, sell_joker

    controller = GameController(data=data)
    controller.new_run("42")
    state = controller.state
    add_joker(state, "j_joker")
    state.round_resets.blind = dict(data.blinds["bl_final_leaf"])
    for card in state.deck_cards:
        card.debuff = True

    sell_joker(state, 0)

    assert state.blind_disabled
    assert all(not card.debuff for card in state.deck_cards)
