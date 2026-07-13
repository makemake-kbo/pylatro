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


def _debuff_state(data, blind_key: str):
    controller = GameController(data=data)
    controller.new_run("42")
    state = controller.state
    state.round_resets.blind = dict(data.blinds[blind_key])
    return state


def test_the_eye_debuffs_repeat_hand_types(data) -> None:
    """Upstream gates The Eye inside `if self.debuff` with an EMPTY debuff
    table — truthy in Lua, falsy once translated to Python — which silently
    disabled the boss. It must zero any hand type already played this round."""
    from pylatro.flow import _debuff_hand  # type: ignore[attr-defined]
    from pylatro.scoring import get_poker_hand_info

    state = _debuff_state(data, "bl_eye")
    state.eye_hands = {}
    pair = [c for c in state.deck_cards if c.rank == "A"][:2]
    hand_name, _, poker_hands, _ = get_poker_hand_info(state, pair)
    assert hand_name == "Pair"

    # check=True probes must not record the hand type
    assert not _debuff_hand(state, pair, hand_name, poker_hands, check=True)
    assert not _debuff_hand(state, pair, hand_name, poker_hands, check=True)
    # first real play of a type is allowed and records it; repeats are zeroed
    assert not _debuff_hand(state, pair, hand_name, poker_hands)
    assert _debuff_hand(state, pair, hand_name, poker_hands)


def test_the_mouth_debuffs_second_hand_type(data) -> None:
    """The Mouth (same empty-debuff gating bug as The Eye) must zero every
    hand type except the first one played this round."""
    from pylatro.flow import _debuff_hand  # type: ignore[attr-defined]
    from pylatro.scoring import get_poker_hand_info

    state = _debuff_state(data, "bl_mouth")
    state.mouth_only_hand = False
    pair = [c for c in state.deck_cards if c.rank == "A"][:2]
    high = [next(c for c in state.deck_cards if c.rank == "9")]
    pair_name, _, pair_hands, _ = get_poker_hand_info(state, pair)
    high_name, _, high_hands, _ = get_poker_hand_info(state, high)
    assert (pair_name, high_name) == ("Pair", "High Card")

    assert not _debuff_hand(state, pair, pair_name, pair_hands)  # locks "Pair"
    assert _debuff_hand(state, high, high_name, high_hands)  # other types zeroed
    assert not _debuff_hand(state, pair, pair_name, pair_hands)  # Pair still fine


def test_hook_blind_triggered_survives_debuff_check(data) -> None:
    """_press_play sets blind_triggered for The Hook before _debuff_hand runs;
    the debuff check must not wipe it (Matador reads it during scoring)."""
    from pylatro.flow import _debuff_hand  # type: ignore[attr-defined]
    from pylatro.scoring import get_poker_hand_info

    state = _debuff_state(data, "bl_hook")
    state.blind_triggered = True  # as set by _press_play this play
    cards = [c for c in state.deck_cards if c.rank == "A"][:2]
    hand_name, _, poker_hands, _ = get_poker_hand_info(state, cards)

    assert not _debuff_hand(state, cards, hand_name, poker_hands)
    assert state.blind_triggered, "empty-debuff boss must keep _press_play's trigger"
