"""Compare actions against paired public remaining-blind samples.

Experimental opt-in search. Future draw order and game RNG are replaced;
all simulations finish with the ordinary heuristic, avoiding recursive search.
"""

from copy import deepcopy
from random import Random

from pylatro.rng import PseudorandomState
from pylatro_agent.action import ActionType, decode_action
from pylatro_agent.constants import ActionRange, SubPhase
from pylatro_agent.heuristic import HeuristicAgent
from pylatro_agent.subset_actions import subset_index
from pylatro_agent.training.fast_runner import FastRunner
from pylatro_cli.controller import GamePhase


def candidate_actions(agent, state, mask, baseline):
    candidates = [baseline]

    def add(action):
        if mask[action] and action not in candidates:
            candidates.append(action)

    play = tuple(sorted(agent._cached_best_hand(state, state.hand_cards)))
    if play:
        add(ActionRange.PLAY_SUBSET_START + subset_index(play))
    if state.current_round.discards_left:
        discard = agent._should_discard_for_draw(state, state.hand_cards, mask)
        if discard:
            add(ActionRange.DISCARD_SUBSET_START + subset_index(discard))
        discard = tuple(sorted(agent._find_worst_cards(state, state.hand_cards, max_discard=5)))
        if discard:
            add(ActionRange.DISCARD_SUBSET_START + subset_index(discard))
    options = []
    for name in ("High Card", "Pair", "Two Pair", "Three of a Kind", "Straight", "Flush", "Full House"):
        play = agent._find_type_hand(state, state.hand_cards, name)
        if play:
            play = tuple(sorted(play))
            options.append((agent._estimate_hand_score(state, play), play))
    for _, play in sorted(options, reverse=True):
        add(ActionRange.PLAY_SUBSET_START + subset_index(play))
    # Keep extra type-specific plays: equal immediate scores can leave very
    # different cards behind for the next draw.
    return candidates[:6]


def choose_action(
    agent, state, mask, baseline, round_score, samples=3, *,
    candidate_override=None, sample_seed=468902, seed_prefix="PUBLIC_BLIND",
):
    decoded = decode_action(baseline)
    if decoded.action_type not in {ActionType.PLAY_SUBSET, ActionType.DISCARD_SUBSET}:
        return baseline, []
    if decoded.action_type == ActionType.PLAY_SUBSET:
        from pylatro_agent.subset_actions import subset_indices

        probe = deepcopy(state, {id(state.data): state.data})
        score = agent._estimate_hand_score(probe, subset_indices(decoded.index))
        if score >= agent._get_blind_target(state) - round_score:
            # Keep the existing teacher's exact current-hand clear. Randomized
            # future probes cannot improve it and can obscure its certainty.
            return baseline, []
    # Candidate scoring may populate derived caches; isolate those too.
    probe = deepcopy(state, {id(state.data): state.data})
    candidates = candidate_actions(agent, probe, mask, baseline) if candidate_override is None else candidate_override
    pool = sorted(
        state.draw_pile,
        key=lambda c: (c.front_key, c.center_key, c.seal or "", c.edition_key or "", c.perma_bonus, c.reward_uid),
    )
    rng = Random(sample_seed)
    orders = [rng.sample(pool, len(pool)) for _ in range(samples)]
    results = []
    for action in candidates:
        outcomes = []
        for sample, order in enumerate(orders):
            trial = deepcopy(state, {id(state.data): state.data})
            by_uid = {c.reward_uid: c for c in trial.draw_pile}
            trial.draw_pile = [by_uid[c.reward_uid] for c in order]
            trial.seed = f"{seed_prefix}_{sample}"
            trial.pseudorandom = PseudorandomState(trial.seed)
            runner = FastRunner(0, state.data, deck_key=state.deck_key, stake=state.stake, raise_errors=True)
            runner._state = runner._ctrl.state = trial
            runner._ctrl.phase = GamePhase.HAND_PLAY
            runner._sub_phase = SubPhase.CHOOSE_ACTION
            runner._round_score = runner._ctrl.round_score = round_score
            oracle = type(agent)(shop_policy=agent._shop_policy, grow_scalers=agent._grow_scalers)
            target = oracle._get_blind_target(trial)
            runner.step(action)
            for _ in range(50):
                if runner.done or runner.sub_phase != SubPhase.CHOOSE_ACTION:
                    break
                choice = oracle.select_action(
                    trial, runner.sub_phase, runner.compute_mask(), round_score=runner.round_score
                )
                runner.step(choice)
            clear = runner.won or (not runner.done and runner.sub_phase == SubPhase.SHOP)
            outcomes.append((int(clear), min(1, runner.round_score / max(1, target))))
        value = sum(20 * win + score for win, score in outcomes) / samples
        results.append((value, action, outcomes))
        # A perfect baseline is already maximal under this objective. Preserve
        # it without evaluating alternatives that could only tie.
        if action == baseline and all(win for win, _ in outcomes):
            return baseline, results
    best = max(results, key=lambda item: item[0])
    return (best[1] if best[0] > results[0][0] + 0.03 else baseline), results


class BlindRollout:
    """Search boss actions, optionally recalculating after every play/discard."""

    def __init__(self, *, repeat=False, all_blinds=False, confirm_early=False, min_ante=1):
        if not 1 <= min_ante <= 8:
            raise ValueError("min_ante must be between 1 and 8")
        self.repeat = repeat
        self.all_blinds = all_blinds
        self.confirm_early = confirm_early
        self.min_ante = min_ante
        self._evaluated_rounds = set()
        self.last_decision = None

    def select(self, agent, state, mask, baseline, round_score):
        self.last_decision = None
        if state.round_resets.ante < self.min_ante:
            return baseline
        if state.blind_on_deck != "Boss" and not self.all_blinds:
            return baseline
        if decode_action(baseline).action_type not in {ActionType.PLAY_SUBSET, ActionType.DISCARD_SUBSET}:
            return baseline
        key = (state.seed, state.round_resets.ante, state.blind_on_deck)
        if not self.repeat and key in self._evaluated_rounds:
            return baseline
        self._evaluated_rounds.add(key)
        action, outcomes = choose_action(agent, state, mask, baseline, round_score)
        proposed = action
        confirmation = None
        if self.confirm_early and state.round_resets.ante <= 2 and proposed != baseline:
            action, confirmation = choose_action(
                agent, state, mask, baseline, round_score, samples=6,
                candidate_override=[baseline, proposed], sample_seed=712037, seed_prefix="CONFIRM_BLIND",
            )
        self.last_decision = dict(
            baseline=baseline, proposed=proposed, action=action, outcomes=outcomes, confirmation=confirmation,
        )
        return action
