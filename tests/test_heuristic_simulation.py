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
