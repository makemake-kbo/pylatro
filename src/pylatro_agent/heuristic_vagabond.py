"""Experimental low-cash Tarot generation while preserving a visible finisher."""

from copy import deepcopy

from pylatro import get_poker_hand_info
from pylatro.runtime import consumable_limit

from .action import ActionType, decode_action, encode_action
from .constants import ActionRange as AR
from .heuristic import HeuristicAgent
from .subset_actions import consumable_subset_index, subset_index


def active_limit(state):
    limits = [j.extra for j in state.jokers if j.center_key == "j_vagabond" and not j.debuff]
    limit = max(limits) if limits else None
    return limit if limit is not None and state.dollars <= limit else None


def defer_cash_mask(state, mask):
    """Keep a money Tarot from disabling generation while spare hands remain."""
    limit = active_limit(state)
    if limit is None or state.current_round.hands_left <= 2:
        return mask
    money = [(i, state.data.centers[c.center_key]) for i, c in enumerate(state.consumables)
             if state.data.centers[c.center_key].get("name") in {"The Hermit", "Temperance"}]
    if len(money) == len(state.consumables) >= consumable_limit(state):
        return mask
    revised = None
    for index, center in money:
        payout = min(state.dollars, center["config"]["extra"]) if center["name"] == "The Hermit" else min(
            sum(j.sell_cost for j in state.jokers), center["config"]["extra"],
        )
        action = encode_action(ActionType.USE_CONSUMABLE_NO_TARGET, index)
        if state.dollars + payout > limit and mask[action]:
            if revised is None:
                revised = mask.copy()
            revised[action] = False
    return mask if revised is None else revised


def farm_tarot(agent, state, mask, remaining):
    if (active_limit(state) is None or state.blind_on_deck == "Boss" or state.round_resets.ante > 6
            or state.current_round.hands_left < 2 or len(state.consumables) >= consumable_limit(state)):
        return None
    keys = {j.center_key for j in state.jokers if not j.debuff}
    if keys & {"j_baron", "j_shoot_the_moon", "j_mime", "j_raised_fist", "j_blackboard",
               "j_loyalty_card", "j_obelisk", "j_ice_cream"}:
        return None
    probe = deepcopy(state, {id(state.data): state.data})
    finisher = tuple(sorted(agent._cached_best_hand(probe, probe.hand_cards)))
    if not finisher or agent._estimate_hand_score(probe, finisher) < remaining * 1.35:
        return None
    _, _, _, scoring = get_poker_hand_info(probe, [probe.hand_cards[i] for i in finisher])
    if "j_ride_the_bus" in keys and any(c.rank in {"J", "Q", "K"} for c in scoring):
        return None
    protected = {c.reward_uid for c in scoring}
    choices = []
    for i, card in enumerate(probe.hand_cards):
        action = AR.PLAY_SUBSET_START + subset_index((i,))
        if (card.reward_uid in protected or card.center_key != "c_base" or card.seal or card.edition_key
                or card.face_down or card.forced_selection or not mask[action]
                or ("j_ride_the_bus" in keys and card.rank in {"J", "Q", "K"})):
            continue
        score = agent._estimate_hand_score(probe, (i,))
        if score < remaining * 0.8:
            choices.append((score, action))
    return min(choices)[1] if choices else None


def clear_blocked_tarot_slot(state, mask):
    """Consume an unused suit Tarot without changing any card's suit.

    Only called after the base policy declines to use a consumable. A full
    inventory otherwise prevents an active Vagabond from generating anything.
    """
    if active_limit(state) is None or len(state.consumables) < consumable_limit(state):
        return None
    suits = {"c_sun": "Hearts", "c_moon": "Clubs", "c_star": "Diamonds", "c_world": "Spades"}
    for slot, consumable in enumerate(state.consumables):
        suit = suits.get(consumable.center_key)
        if suit is None:
            continue
        for index, card in enumerate(state.hand_cards):
            if card.suit != suit or card.face_down:
                continue
            action = encode_action(ActionType.USE_CONSUMABLE_HAND_SUBSET, slot,
                                   consumable_subset_index((index,)))
            if mask[action]:
                return action
    return None


class VagabondAgent(HeuristicAgent):
    def _choose_action(self, state, mask):
        revised = defer_cash_mask(state, mask)
        baseline = super()._choose_action(state, revised)
        if decode_action(baseline).action_type in {ActionType.PLAY_SUBSET, ActionType.DISCARD_SUBSET}:
            action = clear_blocked_tarot_slot(state, revised)
            if action is not None:
                return action
            remaining = self._get_blind_target(state) - self._current_round_score_estimate(state)
            action = farm_tarot(self, state, revised, remaining)
            if action is not None:
                return action
        return baseline
