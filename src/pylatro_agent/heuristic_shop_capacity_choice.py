"""Measure full-round scoring only for a short list of risky shop purchases."""

import math

from pylatro import get_blind_amount

from .constants import ActionRange, SubPhase
from .heuristic_shop_choice_rollout import prepare_trial
from .heuristic_shop_rollout import _CAPACITY_TARGET, _CapacityAgent


def trial_capacity(state, sample):
    runner = prepare_trial(state, sample)
    trial = runner.state
    key = trial.round_resets.blind_choices[trial.blind_on_deck]
    raw_target = get_blind_amount(trial.round_resets.ante, min(trial.stake, 3)) * trial.data.blinds[key].get('mult', 1)
    runner._ctrl.blind_target = lambda: _CAPACITY_TARGET
    runner.step(ActionRange.BLIND_PLAY)
    oracle = _CapacityAgent(raw_target)
    cleared = False
    for _ in range(100):
        if runner.done or runner.sub_phase != SubPhase.CHOOSE_ACTION:
            break
        mask = runner.compute_mask()
        action = oracle.select_action(trial, runner.sub_phase, mask, round_score=runner.round_score)
        if not mask[action]:
            raise RuntimeError('Capacity choice trial selected an illegal action')
        runner.step(action)
        cleared |= runner.round_score >= oracle._get_blind_target(trial)
    if not runner.done:
        raise RuntimeError('Capacity choice trial did not exhaust the round')
    return int(cleared), max(1, runner.round_score)


def choose_capacity(search, agent, best, proposals):
    original = next((p for p in proposals if p[:2] == best), None)
    if original is None:
        return best[1], []
    shortlisted = [original]
    actions = {best[1]}
    for proposal in sorted(proposals, key=lambda p: p[0], reverse=True):
        if proposal[1] not in actions:
            shortlisted.append(proposal)
            actions.add(proposal[1])
        if len(shortlisted) == 3:
            break
    if len(shortlisted) < 2:
        return best[1], []
    results = []
    for gain, action, state in shortlisted:
        outcomes = [trial_capacity(state, sample) for sample in range(3)]
        capacity = math.exp(sum(math.log(score) for _, score in outcomes) / 3)
        short_score, _ = search.output(state, agent, boss=True)
        short_capacity = short_score * search.round_hands(state, boss=True)
        # Preserve the original ordinary/future utility and spending penalties;
        # replace only its boss-capacity component with sequential simulation.
        weight = 1 if state.round_resets.ante >= 8 else 0.25
        value = gain + weight * math.log(capacity / max(1, short_capacity))
        results.append(dict(action=action, gain=gain, value=value, clears=sum(w for w, _ in outcomes),
                            capacity=capacity, short_capacity=short_capacity, outcomes=outcomes))
    winner = max(results, key=lambda r: (r['clears'], r['value']))
    baseline = results[0]
    improve = winner['clears'] > baseline['clears'] or (
        winner['clears'] == baseline['clears'] and winner['value'] > baseline['value'] + 0.03
    )
    return winner['action'] if improve else best[1], results
