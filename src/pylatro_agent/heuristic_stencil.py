"""Evaluate buying Joker Stencil together with the sales that make it useful."""

from copy import deepcopy
from itertools import combinations

from pylatro.instances import add_joker
from pylatro.runtime import joker_limit, sell_joker

from .constants import ActionRange as AR, MAX_JOKER_SLOTS


def acquisition_plan(search, state, mask, agent, baseline, base_value, safe, base_income, boss_reserve):
    offers = [(index, item) for index, item in enumerate(state.shop.cards)
              if item.center_key == "j_stencil"]
    if not offers:
        return 0.03, -1
    removable = [index for index, joker in enumerate(state.jokers[:MAX_JOKER_SLOTS])
                 if not joker.eternal and mask[AR.SHOP_SELL_JOKER_START + index]]
    target = search.target(state, agent)
    boss = state.blind_on_deck == "Boss"
    hands = search.round_hands(state, boss=boss)
    best = (0.03, -1)
    for count in range(1, len(removable) + 1):
        for slots in combinations(removable, count):
            for index, item in offers:
                trial = deepcopy(state, {id(state.data): state.data})
                proceeds = 0
                for slot in reversed(slots):
                    proceeds += trial.jokers[slot].sell_cost
                    sell_joker(trial, slot)
                if trial.dollars - item.cost < boss_reserve:
                    continue
                negative = item.edition and item.edition.get("negative")
                if len(trial.jokers) >= joker_limit(trial) and not negative:
                    continue
                trial.dollars -= item.cost
                add_joker(trial, item.center_key, edition=item.edition)
                score, value = search.purchase_output(trial, agent)
                trial_hands = search.round_hands(trial, boss=boss)
                trial_target = search.target(trial, agent)
                if baseline * hands < target and score * trial_hands / trial_target <= baseline * hands / target * 1.02:
                    continue
                if score / trial_target < baseline / target and score * max(1, trial_hands - 1) < trial_target * 1.1:
                    continue
                income = search.income_value(trial) if safe else 0
                safety_factor = 1.3 if len(state.jokers) < 2 or state.round_resets.ante >= 5 else 1.1
                if income > base_income and score * max(1, trial_hands - 1) < trial_target * safety_factor:
                    income = base_income
                penalty = (0.03 if safe else 0.012) * max(0, item.cost - proceeds)
                if safe and state.round_resets.ante < 7:
                    penalty += 0.004 * max(0, min(state.dollars, 25) - max(0, trial.dollars)) * (8 - state.round_resets.ante)
                gain = value - base_value + income - base_income - penalty
                # The offer stays in the shop after a sale. Re-evaluate the
                # remaining sequence on the next action, before leaving.
                action = AR.SHOP_SELL_JOKER_START + slots[-1]
                if gain > best[0]:
                    best = (gain, action)
    return best
