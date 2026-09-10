"""Experimental Investment Tag skips when the current build can beat the boss."""

from copy import deepcopy

from .constants import ActionRange
from .heuristic import HeuristicAgent
from .heuristic_shop_choice_rollout import trial_blind


def investment_skip(agent, state, mask):
    agent._investment_decision = None
    if (not mask[ActionRange.BLIND_SKIP] or state.round_resets.ante >= 8 or state.dollars >= 25
            or state.round_resets.blind_tags.get(state.blind_on_deck) != 'tag_investment'):
        return None
    probe = deepcopy(state, {id(state.data): state.data})
    probe.blind_on_deck = 'Boss'
    search = agent._get_shop_search()
    score, _ = search.output(probe, agent, boss=True)
    target = search.target(probe, agent)
    hands = search.round_hands(probe, boss=True)
    decision = dict(capacity=score * hands, target=target, sampled_clears=0, samples_run=0, skip=False)
    agent._investment_decision = decision
    if score * max(1, hands - 1) < target * 1.3:
        return None
    for sample in range(6):
        clear, _ = trial_blind(agent, probe, sample)
        decision['samples_run'] += 1
        decision['sampled_clears'] += clear
        if not clear:
            return None
    decision['skip'] = True
    return ActionRange.BLIND_SKIP


class InvestmentAgent(HeuristicAgent):
    def _blind_select(self, state, mask):
        action = investment_skip(self, state, mask)
        return super()._blind_select(state, mask) if action is None else action
