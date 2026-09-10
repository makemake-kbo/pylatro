"""Find a deterministic consumable and visible play that avert a final-hand loss."""

from copy import deepcopy
from itertools import combinations

from pylatro.consumables import use_consumable

from .action import ActionType, encode_action
from .constants import ActionRange as AR
from .subset_actions import consumable_subset_index, subset_index


def winning_consumable(agent, state, mask, remaining):
    """Return (consumable action, following play, score), or None.

    No generated cards, random consumable effects or future draws are sampled.
    The caller decides whether the baseline is actually about to lose.
    """
    for slot, consumable in enumerate(state.consumables):
        center = state.data.centers[consumable.center_key]
        config = center.get('config') or {}
        if center.get('set') == 'Planet' or consumable.center_key in {'c_black_hole', 'c_hermit', 'c_temperance'}:
            targets = [()]
        elif (config.get('suit_conv') or config.get('mod_conv')
              or consumable.center_key in {'c_strength', 'c_death', 'c_hanged_man'}):
            minimum = int(config.get('min_highlighted', 1) or 1)
            maximum = int(config.get('max_highlighted', 2))
            targets = (indices for size in range(minimum, maximum + 1)
                       for indices in combinations(range(len(state.hand_cards)), size))
        else:
            continue
        for indices in targets:
            action = (encode_action(ActionType.USE_CONSUMABLE_HAND_SUBSET, slot, consumable_subset_index(indices))
                      if indices else encode_action(ActionType.USE_CONSUMABLE_NO_TARGET, slot))
            if not mask[action]:
                continue
            trial = deepcopy(state, {id(state.data): state.data})
            use_consumable(trial, slot, hand_targets=indices)
            best = tuple(sorted(agent._cached_best_hand(trial, trial.hand_cards)))
            if not best:
                continue
            score = agent._estimate_hand_score(trial, best)
            if score >= remaining:
                return action, AR.PLAY_SUBSET_START + subset_index(best), score
    return None
