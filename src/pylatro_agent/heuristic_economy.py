"""Experimental Economy Tag skips backed by sampled boss survival."""

from copy import deepcopy

from pylatro.blind import skip_blind

from .constants import ActionRange
from .heuristic import HeuristicAgent
from .heuristic_shop_choice_rollout import trial_blind


def economy_skip(agent, state, mask):
    agent._economy_decision = None
    if (not mask[ActionRange.BLIND_SKIP] or state.round_resets.ante >= 8
            or state.dollars < 25
            or state.round_resets.blind_tags.get(state.blind_on_deck) != 'tag_economy'):
        return None
    probe = deepcopy(state, {id(state.data): state.data})
    skip_blind(probe)
    gain = probe.dollars - state.dollars
    probe.blind_on_deck = 'Boss'
    search = agent._get_shop_search()
    score, _ = search.output(probe, agent, boss=True)
    hands = search.round_hands(probe, boss=True)
    target = search.target(probe, agent)
    decision = dict(cash_gain=gain, cash_after=probe.dollars, capacity=score * hands,
                    target=target, samples_run=0, sampled_clears=0, skip=False)
    agent._economy_decision = decision
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


class EconomyAgent(HeuristicAgent):
    def _blind_select(self, state, mask):
        action = economy_skip(self, state, mask)
        return super()._blind_select(state, mask) if action is None else action
