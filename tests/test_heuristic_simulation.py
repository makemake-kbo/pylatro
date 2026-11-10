import pickle
from copy import deepcopy

import pytest

from pylatro import add_joker, create_run_state, select_blind, start_blind
from pylatro.flow import play_cards
from pylatro_agent.heuristic_simulation import copy_for_scoring


@pytest.mark.parametrize("boss", ["bl_hook", "bl_fish", "bl_final_bell", "bl_final_heart", "bl_arm"])
@pytest.mark.parametrize("size", [1, 4, 5])
def test_single_play_copy_matches_full_copy_and_preserves_live_state(boss, size):
    state = create_run_state(f"scoring_copy_{boss}_{size}", deck_key="b_blue")
    state.round_resets.blind_choices["Boss"] = boss
    select_blind(state, "Boss")
    start_blind(state, "Boss")
    for key in ("j_dna", "j_hiker", "j_vampire", "j_mail", "j_trading"):
        add_joker(state, key)
    for card, enhancement in zip(state.hand_cards, ["m_glass", "m_lucky", "m_steel", "m_gold"], strict=False):
        card.center_key = enhancement
        card.seal = "Purple"
    original = pickle.dumps(state)
    full = deepcopy(state, {id(state.data): state.data})
    quick = copy_for_scoring(state)
    full_result = play_cards(full, range(size))
    quick_result = play_cards(quick, range(size))
    assert quick_result.score == full_result.score
    assert pickle.dumps(state) == original
    # Compare all game state, including generated consumables and card effects.
    # New DNA cards have fresh global identities in each independent probe.
    for a, b in zip(quick.deck_cards, full.deck_cards, strict=True):
        if a.reward_uid != b.reward_uid:
            a.reward_uid = b.reward_uid
    assert pickle.dumps(quick) == pickle.dumps(full)


def test_scoring_copy_shares_only_state_a_play_cannot_write():
    state = create_run_state("scoring_copy_sharing", deck_key="b_blue")
    select_blind(state, "Small")
    start_blind(state, "Small")
    add_joker(state, "j_hiker")
    quick = copy_for_scoring(state)
    assert quick is not state and quick.data is state.data
    for name in ("bosses_used", "used_vouchers", "modifiers", "banned_keys"):
        assert getattr(quick, name) is getattr(state, name)
    for name in ("hands", "cards_played", "jokers", "current_round", "pseudorandom", "deck_cards",
                 "draw_pile", "hand_cards", "discard_pile", "consumables", "tags"):
        assert getattr(quick, name) is not getattr(state, name), name
    held = {id(card) for card in state.hand_cards}
    assert all(id(card) not in held for card in quick.hand_cards)
    # The deck list is new but its untouched cards are the live objects.
    shared = [a for a, b in zip(quick.deck_cards, state.deck_cards, strict=True) if a is b]
    assert len(shared) >= len(state.deck_cards) - len(state.hand_cards) - 8
    before = pickle.dumps(state)
    result = play_cards(quick, [0, 1])
    assert result.score.total > 0
    assert pickle.dumps(state) == before

