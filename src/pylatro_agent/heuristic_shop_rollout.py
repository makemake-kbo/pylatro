"""Compare shop portfolios through sampled complete blinds."""

import math
from copy import deepcopy

from pylatro import get_blind_amount
from pylatro.rng import PseudorandomState

from .constants import ActionRange, SubPhase
from .heuristic import HeuristicAgent
from .heuristic_shop_search import ShopSearch
from .training.fast_runner import FastRunner

_CAPACITY_TARGET = 10**30


class _CapacityAgent(HeuristicAgent):
    def __init__(self, target):
        super().__init__(shop_policy="search", grow_scalers=True)
        self.target = target

    def _get_blind_target(self, state):
        divisor = 1
        if state.blind_on_deck == "Boss" and state.blind_disabled:
            divisor = {"bl_wall": 2, "bl_final_vessel": 3}.get(state.round_resets.blind_choices.get("Boss"), 1)
        return self.target / divisor

    def _random_valid(self, mask):
        return next((i for i, valid in enumerate(mask) if valid), 0)


class ShopRollout(ShopSearch):
    def __init__(self):
        super().__init__()
        self.round_cache = {}

    def output(self, state, agent, *, boss=False):
        key = (
            boss, state.round_resets.ante,
            state.round_resets.blind_choices.get("Boss") if boss else "",
            agent._joker_keys_sig(state), agent._hands_sig(state), state.dollars,
            state.starting_params.hand_size, state.round_resets.hands, state.round_resets.discards,
            agent._hashable_extra(state.probabilities),
            tuple((c.front_key, c.center_key, c.seal, c.edition_key, c.perma_bonus, c.played_this_ante)
                  for c in state.deck_cards),
            tuple(c.center_key for c in state.consumables),
        )
        if key in self.round_cache:
            return self.round_cache[key]
        # Retain the short evaluator's long-term option values, but replace
        # its independent-hand scoring forecast with whole-blind capacity.
        short_score, short_value = super().output(state, agent, boss=boss)
        hands = max(1, self.round_hands(state, boss=boss))
        future_value = short_value - math.log(short_score * hands)
        totals = []
        for sample in range(3):
            trial = deepcopy(state, {id(state.data): state.data})
            trial.seed = f"SHOP_ROUND_{sample}"
            trial.pseudorandom = PseudorandomState(trial.seed)
            trial.draw_pile = sorted(
                trial.deck_cards,
                key=lambda c: (c.front_key, c.center_key, c.seal or "", c.edition_key or "", c.perma_bonus),
            )
            trial.hand_cards = []
            trial.discard_pile = []
            trial.play_cards = []
            trial.blind_on_deck = "Boss" if boss else "Small"
            runner = FastRunner(0, state.data, deck_key=state.deck_key, stake=state.stake, raise_errors=True)
            runner._state = runner._ctrl.state = trial
            runner._ctrl.blind_target = lambda: _CAPACITY_TARGET
            runner.step(ActionRange.BLIND_PLAY)
            # Choose discards for the real scoring requirement. Giving the
            # policy an unreachable target burns every discard even when
            # Banner/Green Joker can already win, undervaluing those builds.
            base = get_blind_amount(state.round_resets.ante, min(state.stake, 3))
            if boss:
                real_target = base * state.data.blinds[state.round_resets.blind_choices["Boss"]].get("mult", 2)
            else:
                real_target = base * {"Small": 1, "Big": 1.5, "Boss": 2}.get(state.blind_on_deck, 1)
            oracle = _CapacityAgent(real_target)
            for _ in range(60):
                if runner.done or runner.sub_phase != SubPhase.CHOOSE_ACTION:
                    break
                mask = runner.compute_mask()
                # Selling is useful to disable Leaf, but endless sales cannot
                # rescue an Eye hand with no playable poker types remaining.
                if trial.blind_disabled or trial.round_resets.blind_choices.get("Boss") != "bl_final_leaf":
                    for index in range(ActionRange.SHOP_SELL_JOKER_END - ActionRange.SHOP_SELL_JOKER_START + 1):
                        if index >= len(trial.jokers) or trial.jokers[index].center_key != "j_luchador":
                            mask[ActionRange.SHOP_SELL_JOKER_START + index] = 0
                if not mask.any():
                    break
                action = oracle.select_action(trial, runner.sub_phase, mask, round_score=runner.round_score)
                if not mask[action]:
                    raise RuntimeError("Round forecast selected an illegal action")
                runner.step(action)
            totals.append(max(1, runner.round_score))
        total = math.exp(sum(math.log(score) for score in totals) / len(totals))
        result = (total / hands, math.log(total) + future_value)
        if len(self.round_cache) > 400:
            self.round_cache.clear()
        self.round_cache[key] = result
        return result
