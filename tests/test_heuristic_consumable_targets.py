import pickle

import pytest

from pylatro import create_run_state, start_blind
from pylatro.instances import create_consumable_instance
from pylatro_agent.action import decode_action
from pylatro_agent.constants import SubPhase
from pylatro_agent.heuristic import HeuristicAgent
from pylatro_agent.masks import compute_action_mask
from pylatro_agent.subset_actions import consumable_subset_indices


@pytest.mark.parametrize("tarot", [
    "c_magician", "c_empress", "c_heirophant", "c_lovers", "c_chariot", "c_justice", "c_devil", "c_tower",
])
def test_enhancement_tarots_preserve_existing_glass_and_steel(tarot):
    state = create_run_state("preserve_enhancements", deck_key="b_blue")
    start_blind(state, "Small")
    by_key = {c.front_key: c for c in state.deck_cards}
    state.hand_cards = [by_key[k] for k in ("C_K", "D_Q", "H_8", "S_8", "C_2")]
    state.hand_cards[0].center_key = "m_glass"
    state.hand_cards[1].center_key = "m_steel"
    state.consumables = [create_consumable_instance(state, tarot)]
    mask = compute_action_mask(state, SubPhase.CHOOSE_ACTION)
    before = pickle.dumps(state)
    action = HeuristicAgent(shop_policy="search")._atomic_consumable_action(
        state, 0, mask, preferred_indices=(0, 2, 3),
    )
    assert action is not None
    targets = consumable_subset_indices(decode_action(action).detail)
    assert targets and set(targets).isdisjoint({0, 1})
    assert pickle.dumps(state) == before


def test_empress_enhances_the_scoring_pair_before_its_kickers():
    state = create_run_state("scoring_enhancement", deck_key="b_blue")
    start_blind(state, "Small")
    by_key = {c.front_key: c for c in state.deck_cards}
    state.hand_cards = [by_key[k] for k in ("C_K", "D_A", "H_8", "S_8", "C_2")]
    state.consumables = [create_consumable_instance(state, "c_empress")]
    action = HeuristicAgent(shop_policy="search")._atomic_consumable_action(
        state, 0, compute_action_mask(state, SubPhase.CHOOSE_ACTION), preferred_indices=(0, 1, 2, 3, 4),
    )
    assert consumable_subset_indices(decode_action(action).detail) == (2, 3)


def test_enhancement_waits_when_every_held_card_is_already_enhanced():
    state = create_run_state("wait_enhancement", deck_key="b_blue")
    start_blind(state, "Small")
    for card in state.hand_cards:
        card.center_key = "m_glass"
    state.consumables = [create_consumable_instance(state, "c_magician")]
    assert HeuristicAgent(shop_policy="search")._atomic_consumable_action(
        state, 0, compute_action_mask(state, SubPhase.CHOOSE_ACTION),
    ) is None


def test_death_waits_when_its_only_valuable_card_is_leftmost():
    state = create_run_state("death_source_direction", deck_key="b_blue")
    start_blind(state, "Small")
    by_key = {c.front_key: c for c in state.deck_cards}
    state.hand_cards = [by_key[k] for k in ("C_K", "C_Q")]
    state.hand_cards[0].center_key = "m_steel"
    state.hand_cards[0].seal = "Blue"
    state.consumables = [create_consumable_instance(state, "c_death")]
    mask = compute_action_mask(state, SubPhase.CHOOSE_ACTION)
    assert HeuristicAgent(shop_policy="search")._atomic_consumable_action(state, 0, mask) is None
