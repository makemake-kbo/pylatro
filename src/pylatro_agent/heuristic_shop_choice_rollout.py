"""Rerank a small shop shortlist using paired complete next-blind trials."""

from copy import deepcopy

from pylatro.rng import PseudorandomState

from .constants import ActionRange, SubPhase
from .training.fast_runner import FastRunner


def prepare_trial(state, sample):
    trial = deepcopy(state, {id(state.data): state.data})
    trial.seed = f'SHOP_CHOICE_{sample}'
    trial.pseudorandom = PseudorandomState(trial.seed)
    trial.draw_pile = sorted(trial.deck_cards, key=lambda c: (
        c.front_key, c.center_key, c.seal or '', c.edition_key or '', c.perma_bonus,
    ))
    trial.hand_cards = []
    trial.discard_pile = []
    trial.play_cards = []
    runner = FastRunner(0, state.data, deck_key=state.deck_key, stake=state.stake, raise_errors=True)
    runner._state = runner._ctrl.state = trial
    return runner


def trial_blind(agent, state, sample):
    runner = prepare_trial(state, sample)
    trial = runner.state
    runner.step(ActionRange.BLIND_PLAY)
    oracle = type(agent)(shop_policy=agent._shop_policy, grow_scalers=agent._grow_scalers)
    for _ in range(100):
        if runner.done or runner.sub_phase != SubPhase.CHOOSE_ACTION:
            break
        mask = runner.compute_mask()
        action = oracle.select_action(trial, runner.sub_phase, mask, round_score=runner.round_score)
        if not mask[action]:
            raise RuntimeError('Shop choice trial selected an illegal action')
        runner.step(action)
    if not runner.done and runner.sub_phase != SubPhase.SHOP:
        raise RuntimeError('Shop choice trial did not finish within its action limit')
    return int(runner.won or not runner.done), runner.round_score


def choose_purchase(agent, best, proposals, *, samples=3):
    """Keep the original purchase unless an alternative clears more paired trials.

    Each proposal contains its existing short-evaluator gain, first action,
    and the actual portfolio after its planned sale/purchase sequence.
    """
    if samples < 1:
        raise ValueError('Purchase comparison needs at least one sample')
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
    selected, most_clears = best[1], -1
    for gain, action, state in shortlisted:
        outcomes = [trial_blind(agent, state, sample) for sample in range(samples)]
        clears = sum(outcome[0] for outcome in outcomes)
        results.append(dict(action=action, gain=gain, clears=clears, outcomes=outcomes))
        if clears > most_clears:
            selected, most_clears = action, clears
        if most_clears == samples:
            break
    return selected, results
