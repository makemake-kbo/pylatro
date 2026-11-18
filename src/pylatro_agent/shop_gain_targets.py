"""The teacher's shop-search gains as prediction targets for the student.

The V68 teacher decides purchases and sales by forecasting what a Joker roster
can score against the coming blind, then comparing that forecast with and
without each item. A behaviour-cloning label says only which action won that
comparison, and a sale is labelled a handful of times per game. The gains
themselves are a graded signal for every Joker and every shop item in every
shop, so the student is trained to predict them from its own observation.

They are targets only. Nothing here is an input: the student never sees a
search result, at training or at play time.
"""

from __future__ import annotations

import math
from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

from pylatro.blind import can_reroll_boss
from pylatro.consumables import use_consumable
from pylatro.instances import add_joker, create_consumable_instance, remove_joker
from pylatro.runtime import joker_limit, sell_joker

from .constants import JOKER_MAX, JOKER_START, MAX_JOKER_SLOTS, MAX_SHOP_ITEMS, SHOP_START, SubPhase, TokenType
from .constants import ActionRange as AR

if TYPE_CHECKING:
    from pylatro.models import RunState

    from .heuristic import HeuristicAgent
    from .heuristic_shop_search import ShopSearch

# One slot per Joker token (the gain of selling it, usually as part of a
# replacement) followed by one per shop token (the gain of buying it).
SHOP_GAIN_SLOTS = JOKER_MAX + MAX_SHOP_ITEMS
# Gains are log-ratio sized (the teacher acts above 0.03); x10 puts typical
# values near 1 and the clamp keeps rare blow-ups from owning the loss.
GAIN_SCALE = 10.0
GAIN_CLAMP = 10.0


def shop_search_gains(search: ShopSearch, state: RunState, mask: np.ndarray, agent: HeuristicAgent) -> dict[int, float]:
    """Gain per legal action for every trial ``ShopSearch.select`` evaluates.

    This repeats the Joker-purchase, Planet-purchase and replacement trials of
    ``select`` without its early exits, so each action keeps the number the
    teacher compared instead of only the winner surviving.
    """
    dollars = state.dollars
    ante = state.round_resets.ante
    owned = {joker.center_key for joker in state.jokers}
    items = list(state.shop.cards) + list(state.shop.vouchers) + list(state.shop.boosters)
    before_boss = state.blind_on_deck == "Boss"
    has_counter = any(
        joker.center_key == "j_chicot" or (joker.center_key == "j_luchador" and not joker.eternal)
        for joker in state.jokers
    )
    boss_risk = False
    if not has_counter and can_reroll_boss(state):
        boss_risk = search.boss_reroll_improves_risk(state, agent)
    baseline, base_value = search.purchase_output(state, agent)
    target = search.target(state, agent)
    forecast_hands = search.round_hands(state, boss=before_boss)
    safety_factor = 1.3 if len(state.jokers) < 2 or ante >= 5 else 1.1
    safe = baseline * max(1, forecast_hands - 1) >= target * safety_factor
    base_income = search.income_value(state) if safe else 0
    boss_reserve = 10 if can_reroll_boss(state) and boss_risk else 0
    gains: dict[int, float] = {}

    def offer(action: int, gain: float) -> None:
        if mask[action] and gain > gains.get(action, -math.inf):
            gains[action] = float(gain)

    if owned & {"j_stencil", "j_campfire"}:
        for slot in range(min(len(state.jokers), MAX_JOKER_SLOTS)):
            action = AR.SHOP_SELL_JOKER_START + slot
            if not mask[action]:
                continue
            trial = deepcopy(state, {id(state.data): state.data})
            sell_joker(trial, slot)
            trial_score, value = search.purchase_output(trial, agent)
            gain = value - base_value
            if safe:
                gain += search.income_value(trial) - base_income
            if before_boss and baseline * forecast_hands < target * 1.15:
                trial_capacity = trial_score * search.round_hands(trial, boss=True) / search.target(trial, agent)
                if trial_capacity >= 1.15:
                    gain = max(gain, math.log(trial_capacity / (baseline * forecast_hands / target)))
            if trial_score >= baseline:
                offer(action, gain)

    for index, item in enumerate(items):
        center = state.data.centers[item.center_key]
        fool_planet = (
            center.get("name") == "The Fool"
            and state.data.centers.get(state.last_tarot_planet, {}).get("set") == "Planet"
        )
        if center.get("set") == "Planet" or fool_planet:
            action = AR.SHOP_BUY_START + index
            if mask[action] and dollars - item.cost >= boss_reserve:
                trial = deepcopy(state, {id(state.data): state.data})
                trial.dollars -= item.cost
                used = use_consumable(trial, create_consumable_instance(trial, item.center_key))
                if fool_planet and used.created_consumables:
                    use_consumable(trial, len(trial.consumables) - 1)
                _, value = search.purchase_output(trial, agent)
                offer(action, value - base_value - (0.04 if safe else 0.012) * item.cost)
            continue
        if center.get("set") != "Joker":
            continue
        negative = bool(item.edition and item.edition.get("negative"))
        if len(state.jokers) < joker_limit(state) or negative:
            replaced_slots: list[int | None] = [None]
        else:
            replaced_slots = [
                slot for slot in range(min(len(state.jokers), MAX_JOKER_SLOTS)) if not state.jokers[slot].eternal
            ]
        for slot in replaced_slots:
            proceeds = 0 if slot is None else state.jokers[slot].sell_cost
            if dollars + proceeds - item.cost < boss_reserve:
                continue
            trial = deepcopy(state, {id(state.data): state.data})
            if slot is not None:
                remove_joker(trial, trial.jokers[slot])
            if len(trial.jokers) >= joker_limit(trial) and not negative:
                continue
            trial.dollars += proceeds - item.cost
            add_joker(trial, item.center_key, edition=item.edition)
            trial_score, value = search.purchase_output(trial, agent)
            trial_hands = search.round_hands(trial, boss=before_boss)
            trial_target = search.target(trial, agent)
            if (
                baseline * forecast_hands < target
                and trial_score * trial_hands / trial_target <= baseline * forecast_hands / target * 1.02
                and item.center_key != "j_riff_raff"
            ):
                continue
            if (
                trial_score / trial_target < baseline / target
                and trial_score * max(1, trial_hands - 1) < trial_target * 1.1
            ):
                continue
            trial_income = search.income_value(trial) if safe else 0
            if trial_income > base_income and trial_score * max(1, trial_hands - 1) < trial_target * safety_factor:
                trial_income = base_income
            price_penalty = (0.03 if safe else 0.012) * max(0, item.cost - proceeds)
            if safe and ante < 7:
                price_penalty += 0.004 * max(0, min(dollars, 25) - max(0, trial.dollars)) * (8 - ante)
            gain = value - base_value + trial_income - base_income - price_penalty
            offer(AR.SHOP_BUY_START + index if slot is None else AR.SHOP_SELL_JOKER_START + slot, gain)
    return gains


def shop_gain_targets(
    agent: HeuristicAgent, state: RunState, sub_phase: SubPhase, mask: np.ndarray
) -> np.ndarray | None:
    """Per-slot gains for a shop decision of a search teacher, NaN where it ran no trial.

    Returns None outside the shop and for teachers without the portfolio search.
    """
    if sub_phase != SubPhase.SHOP or getattr(agent, "_shop_policy", None) != "search":
        return None
    targets = np.full(SHOP_GAIN_SLOTS, np.nan, dtype=np.float32)
    for action, gain in shop_search_gains(agent._get_shop_search(), state, mask, agent).items():
        if AR.SHOP_SELL_JOKER_START <= action < AR.SHOP_SELL_JOKER_START + MAX_JOKER_SLOTS:
            targets[action - AR.SHOP_SELL_JOKER_START] = gain
        elif AR.SHOP_BUY_START <= action < AR.SHOP_BUY_START + MAX_SHOP_ITEMS:
            targets[JOKER_MAX + action - AR.SHOP_BUY_START] = gain
    return targets


def shop_gain_loss(
    prediction: torch.Tensor,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
    token_types: torch.Tensor,
) -> torch.Tensor:
    """Loss for the gain head on the rows that carry search targets.

    ``prediction`` is ``[batch, SHOP_GAIN_SLOTS, 2]``: a logit for "the search
    ran a trial for this item" and the scaled gain of that trial. The first is
    trained on every real Joker and shop token of a labelled shop, so "nothing
    here is worth selling" is learned as well; the second only where a trial
    exists.
    """
    real = torch.cat(
        (
            token_types[:, JOKER_START : JOKER_START + JOKER_MAX] == int(TokenType.JOKER),
            token_types[:, SHOP_START : SHOP_START + MAX_SHOP_ITEMS] == int(TokenType.SHOP),
        ),
        dim=1,
    )
    scope = real & (target_mask > 0).unsqueeze(1)
    tried = scope & ~torch.isnan(targets)
    loss = prediction.new_zeros(())
    if scope.any():
        loss = loss + F.binary_cross_entropy_with_logits(prediction[..., 0][scope], tried[scope].to(prediction.dtype))
    if tried.any():
        scaled = (targets[tried] * GAIN_SCALE).clamp(-GAIN_CLAMP, GAIN_CLAMP)
        loss = loss + F.smooth_l1_loss(prediction[..., 1][tried], scaled.to(prediction.dtype))
    return loss
