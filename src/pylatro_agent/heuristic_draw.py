"""Small public-card discard search for fragile opening builds."""

import random
from collections import defaultdict
from copy import deepcopy

from pylatro.flow import discard_cards
from pylatro.rng import PseudorandomState
from pylatro.scoring import RANK_TO_ID

from .constants import ActionRange
from .subset_actions import subset_index


def sampled_early_discard(agent, state, mask, remaining, current_score, *, costly=False):
    final_chance = state.current_round.hands_left == 1 and current_score < remaining
    if (
        (state.round_resets.ante > 2 and not costly and not final_chance)
        or state.current_round.discards_left <= 0 or not state.draw_pile
    ):
        return None
    keys = {j.center_key for j in state.jokers if not j.debuff}
    if not costly and not final_chance and keys & {
        "j_baron", "j_shoot_the_moon", "j_mime", "j_blackboard", "j_card_sharp",
    }:
        return None
    hand = state.hand_cards
    protected = {i for i, c in enumerate(hand) if c.forced_selection or (c.seal == "Blue" and not final_chance)}
    candidates = []

    def add(indices):
        indices = tuple(sorted(set(indices) - protected))[:5]
        if indices and mask[ActionRange.DISCARD_SUBSET_START + subset_index(indices)] and indices not in candidates:
            candidates.append(indices)

    proposal = agent._should_discard_for_draw(state, hand, mask)
    if proposal:
        add(proposal)
    add(agent._find_worst_cards(state, hand, max_discard=5))
    if costly or final_chance:
        best_play = set(agent._cached_best_hand(state, hand))
        add(set(range(len(hand))) - best_play)
        for index in agent._find_worst_cards(state, hand, max_discard=2):
            add((index,))
    suits, ranks = defaultdict(set), defaultdict(set)
    for index, card in enumerate(hand):
        if not card.debuff:
            suits[card.suit].add(index)
        ranks[card.rank].add(index)
    for group in suits.values():
        if len(group) >= 3:
            add(set(range(len(hand))) - group)
    for group in ranks.values():
        if len(group) >= 2:
            add(set(range(len(hand))) - group)
    for low in range(1, 11):
        held = {}
        for index, card in enumerate(hand):
            rank = RANK_TO_ID[card.rank]
            if rank == 14 and low == 1:
                rank = 1
            if low <= rank < low + 5 and not card.debuff:
                held.setdefault(rank, index)
        if len(held) >= 4:
            add(set(range(len(hand))) - set(held.values()))
    if not candidates:
        return None
    candidates = candidates[:8]
    pool = sorted(
        state.draw_pile,
        key=lambda c: (c.front_key, c.center_key, c.seal or "", c.edition_key or "", c.perma_bonus, c.reward_uid),
    )
    rng = random.Random(9479)
    orders = [rng.sample(pool, len(pool)) for _ in range(8)]
    from .heuristic import HeuristicAgent

    oracle = HeuristicAgent(shop_policy=agent._shop_policy)
    best = (-1 if final_chance else min(current_score, remaining) * (1.03 if costly else 1.2), None)
    for indices in candidates:
        value = 0
        for sample, order in enumerate(orders):
            trial = deepcopy(state, {id(state.data): state.data})
            cards = {c.reward_uid: c for c in trial.draw_pile}
            trial.draw_pile = [cards[c.reward_uid] for c in order]
            trial.seed = f"DRAW_SAMPLE_{sample}"
            trial.pseudorandom = PseudorandomState(trial.seed)
            discard_cards(trial, indices)
            play = tuple(sorted(oracle._cached_best_hand(trial, trial.hand_cards)))
            score = oracle._estimate_hand_score(trial, play)
            value += min(remaining, score)
            if final_chance and score >= remaining:
                value += remaining * 10
        value /= len(orders)
        if value > best[0]:
            best = (value, indices)
    return best[1]
