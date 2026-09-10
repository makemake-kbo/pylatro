"""Counterfactual choices for rare hand-level upgrades in open packs."""

from copy import deepcopy

from pylatro.shop import claim_pack_card, close_pack

from .constants import ActionRange as AR


def select_black_hole_pack(state, mask, agent, search):
    """Compare Black Hole, legal Planets and Red Card's skip reward.

    Ordinary packs keep their established policy. Actual claim effects ensure
    Planet-only triggers such as Constellation are included in the comparison.
    """
    if not state.pack or not any(card.center_key == "c_black_hole" for card in state.pack.cards):
        return None
    candidates = [
        index for index, card in enumerate(state.pack.cards)
        if mask[AR.PACK_CLAIM_START + index]
        and (card.center_key == "c_black_hole" or state.data.centers[card.center_key].get("set") == "Planet")
    ]
    candidates.sort(key=lambda index: state.pack.cards[index].center_key != "c_black_hole")
    if not candidates:
        return None
    best_value, best_action = float("-inf"), None
    for index in candidates:
        trial = deepcopy(state, {id(state.data): state.data})
        claim_pack_card(trial, index)
        _, value = search.purchase_output(trial, agent)
        if value > best_value + 0.001:
            best_value, best_action = value, AR.PACK_CLAIM_START + index
    if mask[AR.PACK_SKIP] and any(j.center_key == "j_red_card" for j in state.jokers):
        trial = deepcopy(state, {id(state.data): state.data})
        close_pack(trial, skipped=True)
        _, value = search.purchase_output(trial, agent)
        if value > best_value + 0.001:
            return AR.PACK_SKIP
    return best_action
