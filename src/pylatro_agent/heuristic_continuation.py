"""Estimate the next hand after a play using the public remaining card pool."""

from copy import deepcopy
from random import Random

from pylatro.flow import draw_to_hand, play_cards
from pylatro.rng import PseudorandomState
from pylatro.scoring import RANK_TO_ID

from .constants import ActionRange
from .joker_layout import apply_best_joker_order
from .subset_actions import subset_index


def next_hand_score(agent, state, indices, samples=6):
    if state.current_round.hands_left <= 1:
        return 0.0
    from .heuristic import HeuristicAgent

    pool = sorted(
        state.draw_pile,
        key=lambda c: (c.front_key, c.center_key, c.seal or "", c.edition_key or "", c.perma_bonus, c.reward_uid),
    )
    rng = Random(68423)
    oracle = HeuristicAgent(shop_policy="search")
    total = 0.0
    for sample in range(samples):
        trial = deepcopy(state, {id(state.data): state.data})
        by_uid = {card.reward_uid: card for card in trial.draw_pile}
        trial.draw_pile = [by_uid[card.reward_uid] for card in rng.sample(pool, len(pool))]
        trial.seed = f"CONTINUATION_{sample}"
        trial.pseudorandom = PseudorandomState(trial.seed)
        apply_best_joker_order(trial, indices)
        play_cards(trial, indices)
        draw_to_hand(trial)
        if not trial.hand_cards:
            continue
        best = tuple(sorted(oracle._cached_best_hand(trial, trial.hand_cards)))
        if best:
            total += oracle._estimate_hand_score(trial, best)
    return total / samples


def locked_straight_redraw(agent, state, mask):
    """Use a scoreless play as a redraw when Mouth has locked Straights."""
    if state.mouth_only_hand != "Straight" or state.current_round.hands_left <= 1:
        return None
    hand = state.hand_cards
    forced = {i for i, card in enumerate(hand) if card.forced_selection}
    candidates = set()

    def add(indices):
        indices = tuple(sorted(set(indices) | forced))
        if 1 <= len(indices) <= 5 and mask[ActionRange.PLAY_SUBSET_START + subset_index(indices)]:
            candidates.add(indices)

    add(agent._find_worst_cards(state, hand, max_discard=5))
    for low in range(1, 11):
        held = {}
        for index, card in enumerate(hand):
            if card.center_key == "m_stone" or index in forced:
                continue
            rank = RANK_TO_ID[card.rank]
            if rank == 14 and low == 1:
                rank = 1
            if low <= rank < low + 5 and rank not in held:
                held[rank] = index
        outside = set(range(len(hand))) - set(held.values())
        if len(outside) > 5:
            outside = set(sorted(outside, key=lambda i: (i not in forced, RANK_TO_ID[hand[i].rank]))[:5])
        add(outside)
    if not candidates:
        return None
    return max(sorted(candidates), key=lambda indices: (next_hand_score(agent, state, indices), len(indices)))
