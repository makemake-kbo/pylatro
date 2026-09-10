"""Use Trading Card when visible retained cards can still finish the blind."""

from collections import Counter
from copy import deepcopy

from pylatro.flow import discard_cards
from pylatro.scoring import RANK_TO_NOMINAL

from .constants import ActionRange as AR
from .subset_actions import subset_index


_AVOID = {
    "j_green_joker", "j_ramen", "j_baron", "j_shoot_the_moon", "j_mime",
    "j_raised_fist", "j_blackboard", "j_delayed_grat",
}


def trading_income(state):
    """Expected income assuming a safe first discard in half of future rounds."""
    keys = {j.center_key for j in state.jokers}
    if (keys & _AVOID or "j_burglar" in keys or state.round_resets.discards <= 0
            or len(state.deck_cards) <= 20):
        return 0.0
    if not any(c.center_key == "c_base" and not c.seal and not c.edition_key and not c.perma_bonus
               for c in state.deck_cards):
        return 0.0
    return 0.5 * sum(j.extra for j in state.jokers if j.center_key == "j_trading")


def trading_discard(agent, state, mask, finisher, remaining, score):
    """Discard one expendable card without consulting its unknown replacement."""
    keys = {j.center_key for j in state.jokers if not j.debuff}
    if ("j_trading" not in keys or keys & _AVOID or state.current_round.discards_used
            or state.current_round.discards_left <= 0 or len(state.deck_cards) <= 20
            or score < remaining * 1.2 or len(state.hand_cards) <= len(finisher) + 1
            or (state.round_resets.ante >= 8 and state.blind_on_deck == "Boss")):
        return None
    protected = {state.hand_cards[i].reward_uid for i in finisher}
    ranks = Counter(c.rank for c in state.deck_cards)
    candidates = [
        i for i, card in enumerate(state.hand_cards)
        if card.reward_uid not in protected and card.center_key == "c_base"
        and not card.seal and not card.edition_key and not card.perma_bonus
        and not card.face_down and not card.forced_selection
        and not (card.rank == "2" and "j_wee" in keys)
        and mask[AR.DISCARD_SUBSET_START + subset_index((i,))]
    ]
    candidates.sort(key=lambda i: (ranks[state.hand_cards[i].rank], RANK_TO_NOMINAL.get(state.hand_cards[i].rank, 0)))
    for index in candidates:
        trial = deepcopy(state, {id(state.data): state.data})
        pool = trial.draw_pile
        trial.draw_pile = []
        discard_cards(trial, [index])
        # The real refill consumes one draw card. Keep its count for Blue
        # Joker but never introduce the unseen card into the forecast hand.
        trial.draw_pile = pool[:-1]
        kept = tuple(i for i, card in enumerate(trial.hand_cards) if card.reward_uid in protected)
        if len(kept) == len(finisher) and agent._estimate_hand_score(trial, kept) >= remaining * 1.1:
            return AR.DISCARD_SUBSET_START + subset_index((index,))
    return None
