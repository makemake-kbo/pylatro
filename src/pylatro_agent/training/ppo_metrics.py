"""PPO rollout diagnostics and TensorBoard metric writers."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np

from ..action import decode_action
from ..constants import (
    POKER_HAND_NAMES,
)
from ..diagnostics import (
    MAX_DIAGNOSTIC_SHOP_JOKERS,
)
from .ppo_observations import (
    _extract_step_info_value,
)
from .ppo_policy import (
    _ACTION_TYPES,
)

_REPORTED_POTENTIAL_COMPONENTS = (
    "economy",
    "tarot_option_value",
    "planet_option_value",
    "seal_value",
    "joker_search_option",
    "standard_pack_search_option",
)

_BLIND_INDEX = {"small": 0, "big": 1, "boss": 2}


def _next_blind_clear_outcome(
    *,
    shop_ante: int,
    shop_blind_index: int,
    final_ante: int,
    won: bool,
    terminal_blind: str,
) -> float:
    """Return whether the blind forecast at shop leave was subsequently cleared."""

    if won or final_ante > shop_ante:
        return 1.0
    if final_ante < shop_ante:
        return 0.0
    terminal_index = _BLIND_INDEX.get(str(terminal_blind).lower(), shop_blind_index)
    return float(terminal_index > shop_blind_index)


def _binary_roc_auc(predictions: list[float], outcomes: list[float]) -> float | None:
    """Return tie-aware ROC AUC, or None when only one outcome class exists."""

    prediction_array = np.asarray(predictions, dtype=np.float64)
    outcome_array = np.asarray(outcomes, dtype=np.float64) > 0.5
    positive_count = int(outcome_array.sum())
    negative_count = int(outcome_array.size - positive_count)
    if positive_count == 0 or negative_count == 0:
        return None

    order = np.argsort(prediction_array, kind="mergesort")
    sorted_predictions = prediction_array[order]
    ranks = np.empty(prediction_array.size, dtype=np.float64)
    start = 0
    while start < prediction_array.size:
        end = start + 1
        while end < prediction_array.size and sorted_predictions[end] == sorted_predictions[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    positive_rank_sum = float(ranks[outcome_array].sum())
    return (positive_rank_sum - positive_count * (positive_count + 1) / 2.0) / (positive_count * negative_count)


@dataclass
class _RolloutMetrics:
    """Per-rollout accumulators populated during the step loop."""

    action_type_counts: Counter = field(default_factory=Counter)
    hand_chosen_counts: Counter = field(default_factory=Counter)
    consumable_use_set_counts: Counter = field(default_factory=Counter)
    consumable_claim_set_counts: Counter = field(default_factory=Counter)
    consumable_buy_set_counts: Counter = field(default_factory=Counter)
    consumable_offered_counts: Counter = field(default_factory=Counter)
    consumable_claimable_counts: Counter = field(default_factory=Counter)
    consumable_eligible_offer_opportunities: Counter = field(default_factory=Counter)
    consumable_inventory_full_blocked_counts: Counter = field(default_factory=Counter)
    consumable_acquired_counts: Counter = field(default_factory=Counter)
    consumable_exact_use_counts: Counter = field(default_factory=Counter)
    consumable_pack_auto_use_counts: Counter = field(default_factory=Counter)
    consumable_sold_counts: Counter = field(default_factory=Counter)
    consumable_overwritten_counts: Counter = field(default_factory=Counter)
    consumable_expired_counts: Counter = field(default_factory=Counter)
    consumable_owned_states: Counter = field(default_factory=Counter)
    consumable_legal_use_opportunities: Counter = field(default_factory=Counter)
    pack_claim_seal_counts: Counter = field(default_factory=Counter)
    pack_offered_seal_counts: Counter = field(default_factory=Counter)
    step_rewards: list[float] = field(default_factory=list)
    progress_flags: list[float] = field(default_factory=list)
    steps_since_progress: list[float] = field(default_factory=list)
    chosen_action_probs: list[float] = field(default_factory=list)
    max_action_type_probs: list[float] = field(default_factory=list)
    done_flags: list[float] = field(default_factory=list)
    completed_episode_rewards: list[float] = field(default_factory=list)
    completed_episode_lengths: list[int] = field(default_factory=list)
    completed_episode_wins: list[float] = field(default_factory=list)
    completed_episode_stalls: list[float] = field(default_factory=list)
    completed_episode_antes: list[int] = field(default_factory=list)
    completed_episode_tarot_uses: list[int] = field(default_factory=list)
    terminal_loss_antes: list[int] = field(default_factory=list)
    terminal_loss_score_ratios: list[float] = field(default_factory=list)
    terminal_loss_dollars: list[float] = field(default_factory=list)
    terminal_loss_cash_ge_10: list[float] = field(default_factory=list)
    terminal_loss_joker_full_weak: list[float] = field(default_factory=list)
    terminal_loss_blind_counts: Counter = field(default_factory=Counter)
    ante1_death_blind_counts: Counter = field(default_factory=Counter)
    terminal_boss_loss_counts: Counter = field(default_factory=Counter)
    terminal_loss_last_play_top1: list[float] = field(default_factory=list)
    terminal_loss_last_play_value_ratios: list[float] = field(default_factory=list)
    ante1_blind_clear_counts: Counter = field(default_factory=Counter)
    ante1_clear_hands_used: list[float] = field(default_factory=list)
    ante1_clear_hands_unused: list[float] = field(default_factory=list)
    ante1_clear_discards_used: list[float] = field(default_factory=list)
    ante1_play_count: int = 0
    ante1_play_hand_counts: Counter = field(default_factory=Counter)
    ante1_play_realized_to_remaining_target: list[float] = field(default_factory=list)
    ante1_conservative_chosen_best_ratios: list[float] = field(default_factory=list)
    ante1_one_hand_clear_proxy_observed: int = 0
    ante1_one_hand_clear_proxy_available: int = 0
    ante1_one_hand_clear_proxy_chosen: int = 0
    ante1_one_hand_clear_proxy_missed: int = 0
    reward_component_values: defaultdict = field(default_factory=lambda: defaultdict(list))
    planet_use_alignment_counts: Counter = field(default_factory=Counter)
    planet_use_hand_counts: Counter = field(default_factory=Counter)
    planet_active_plan_owned: int = 0
    planet_active_plan_legal: int = 0
    planet_active_plan_uses: int = 0
    tarot_use_family_counts: Counter = field(default_factory=Counter)
    tarot_fix_reliability_deltas: list[float] = field(default_factory=list)
    attributable_cash_payouts: list[float] = field(default_factory=list)
    gold_cards_created: int = 0
    held_gold_payout_dollars: int = 0
    shop_offered_joker_counts: Counter = field(default_factory=Counter)
    shop_bought_joker_counts: Counter = field(default_factory=Counter)
    shop_sold_joker_counts: Counter = field(default_factory=Counter)
    clear_probabilities: list[float] = field(default_factory=list)
    immediate_death_probabilities: list[float] = field(default_factory=list)
    shop_leave_flags: list[float] = field(default_factory=list)
    shop_unsafe_leave_flags: list[float] = field(default_factory=list)
    shop_unsafe_can_reroll_flags: list[float] = field(default_factory=list)
    shop_missed_upgrade_flags: list[float] = field(default_factory=list)
    shop_leave_full_weak_flags: list[float] = field(default_factory=list)
    shop_best_upgrade_deltas: list[float] = field(default_factory=list)
    shop_survival_predictions: list[float] = field(default_factory=list)
    shop_survival_outcomes: list[float] = field(default_factory=list)
    shop_survival_briers: list[float] = field(default_factory=list)
    risk_shop_death_predictions: list[float] = field(default_factory=list)
    risk_shop_death_outcomes: list[float] = field(default_factory=list)
    risk_shop_death_briers: list[float] = field(default_factory=list)
    risk_shop_raw_death_predictions: list[float] = field(default_factory=list)
    risk_shop_raw_death_briers: list[float] = field(default_factory=list)
    risk_ante1_false_safe_deaths: list[float] = field(default_factory=list)
    potential_values: defaultdict = field(default_factory=lambda: defaultdict(list))
    hand_plan_type_counts: Counter = field(default_factory=Counter)
    hand_plan_reliability: list[float] = field(default_factory=list)
    hand_plan_readiness: list[float] = field(default_factory=list)
    purple_seal_tarots_generated: int = 0
    blue_seal_planets_generated: int = 0
    purple_seals_activated: int = 0
    blue_seals_activated: int = 0
    joker_churn_count: int = 0
    joker_replacement_events: int = 0
    joker_order_money_flags: list[float] = field(default_factory=list)
    joker_order_dollars_gained: list[float] = field(default_factory=list)
    joker_order_chips_forgone: list[float] = field(default_factory=list)
    joker_order_clear_flags: list[float] = field(default_factory=list)


def _write_means(writer, metrics: dict[str, list], step: int) -> None:
    for tag, values in metrics.items():
        writer.add_scalar(tag, float(np.mean(values)), step)


def _write_rollout_episode_metrics(writer, rm: _RolloutMetrics, step: int) -> None:
    """Log episode outcomes completed in this rollout, never lifetime means."""
    if not rm.completed_episode_rewards:
        return
    metrics = {
        "episode_reward_mean": rm.completed_episode_rewards,
        "episode_length_mean": rm.completed_episode_lengths,
        "win_rate": rm.completed_episode_wins,
        "stall_rate": rm.completed_episode_stalls,
        "final_ante_mean": rm.completed_episode_antes,
        # Retain the clearer alias introduced for ante diagnostics.
        "mean_ante_reached": rm.completed_episode_antes,
    }
    if rm.completed_episode_tarot_uses:
        metrics["tarot_uses_per_completed_episode_mean"] = rm.completed_episode_tarot_uses
    for tag, values in metrics.items():
        writer.add_scalar(f"rollout/{tag}", float(np.mean(values)), step)


def _write_consumable_strategy_metrics(
    writer,
    rm: _RolloutMetrics,
    step: int,
) -> None:
    """Write bounded Planet/Tarot funnel metrics with explicit denominators."""

    steps_per_thousand = max(len(rm.step_rewards) / 1000.0, 1e-9)
    for consumable_set in ("Planet", "Tarot"):
        tag = consumable_set.lower()
        selected = rm.consumable_buy_set_counts[consumable_set] + rm.consumable_claim_set_counts[consumable_set]
        inventory_uses = max(
            rm.consumable_exact_use_counts[consumable_set] - rm.consumable_pack_auto_use_counts[consumable_set],
            0,
        )
        counts = {
            "offered": rm.consumable_offered_counts[consumable_set],
            "claimable": rm.consumable_claimable_counts[consumable_set],
            "acquired": rm.consumable_acquired_counts[consumable_set],
            "used": rm.consumable_exact_use_counts[consumable_set],
            "pack_auto_used": rm.consumable_pack_auto_use_counts[consumable_set],
            "sold": rm.consumable_sold_counts[consumable_set],
            "overwritten": rm.consumable_overwritten_counts[consumable_set],
            "expired": rm.consumable_expired_counts[consumable_set],
            "inventory_full_blocked": rm.consumable_inventory_full_blocked_counts[consumable_set],
        }
        for name, count in counts.items():
            writer.add_scalar(f"strategy/consumables/{tag}/{name}_per_1k_steps", count / steps_per_thousand, step)
        eligible = rm.consumable_eligible_offer_opportunities[consumable_set]
        if eligible:
            writer.add_scalar(f"strategy/consumables/{tag}/claim_rate_given_eligible_offer", selected / eligible, step)
        legal = rm.consumable_legal_use_opportunities[consumable_set]
        if legal:
            writer.add_scalar(f"strategy/consumables/{tag}/use_rate_given_owned_legal", inventory_uses / legal, step)
        offered = rm.consumable_offered_counts[consumable_set]
        if offered:
            writer.add_scalar(
                f"strategy/consumables/{tag}/inventory_full_blocked_offer_rate",
                rm.consumable_inventory_full_blocked_counts[consumable_set] / offered,
                step,
            )
        writer.add_scalar(
            f"strategy/consumables/{tag}/owned_state_fraction",
            rm.consumable_owned_states[consumable_set] / max(len(rm.step_rewards), 1),
            step,
        )

    planet_uses = sum(rm.planet_use_alignment_counts.values())
    if planet_uses:
        writer.add_scalar(
            "strategy/planets/matched_use_rate", rm.planet_use_alignment_counts["matched"] / planet_uses, step
        )
        writer.add_scalar(
            "strategy/planets/unmatched_use_rate", rm.planet_use_alignment_counts["unmatched"] / planet_uses, step
        )
    for hand_name in POKER_HAND_NAMES:
        count = rm.planet_use_hand_counts[hand_name]
        if count:
            hand_tag = hand_name.lower().replace(" ", "_")
            writer.add_scalar(
                f"strategy/planets/uses_by_hand/{hand_tag}_per_1k_steps", count / steps_per_thousand, step
            )
    writer.add_scalar(
        "strategy/planets/active_plan_owned_states_per_1k_steps", rm.planet_active_plan_owned / steps_per_thousand, step
    )
    if rm.planet_active_plan_legal:
        writer.add_scalar(
            "strategy/planets/active_plan_use_rate_given_legal",
            rm.planet_active_plan_uses / rm.planet_active_plan_legal,
            step,
        )

    tarot_uses = sum(rm.tarot_use_family_counts.values())
    for family in (
        "cash",
        "gold",
        "deck_cut",
        "rank_fix",
        "suit_fix",
        "creation",
        "joker",
        "enhancement",
    ):
        if tarot_uses:
            writer.add_scalar(
                f"strategy/tarots/use_family/{family}_share", rm.tarot_use_family_counts[family] / tarot_uses, step
            )
    if rm.tarot_fix_reliability_deltas:
        writer.add_scalar(
            "strategy/tarots/fix_reliability_delta_mean", float(np.mean(rm.tarot_fix_reliability_deltas)), step
        )
    if rm.attributable_cash_payouts:
        writer.add_scalar(
            "strategy/tarots/attributable_cash_per_use_mean", float(np.mean(rm.attributable_cash_payouts)), step
        )
    writer.add_scalar("strategy/gold/cards_created_per_1k_steps", rm.gold_cards_created / steps_per_thousand, step)
    writer.add_scalar(
        "strategy/gold/payout_dollars_per_1k_steps", rm.held_gold_payout_dollars / steps_per_thousand, step
    )
    for seal in ("Blue", "Purple"):
        writer.add_scalar(
            f"strategy/seals/offered/{seal.lower()}_per_1k_steps",
            rm.pack_offered_seal_counts[seal] / steps_per_thousand,
            step,
        )
        writer.add_scalar(
            f"strategy/seals/activated/{seal.lower()}_per_1k_steps",
            (rm.blue_seals_activated if seal == "Blue" else rm.purple_seals_activated) / steps_per_thousand,
            step,
        )


def _write_action_behavior_metrics(writer, rm: _RolloutMetrics, step: int) -> None:
    """Write compact action-family and no-progress-loop diagnostics."""

    action_total = sum(rm.action_type_counts.values())
    if action_total:
        for action_type in _ACTION_TYPES:
            writer.add_scalar(
                f"actions/type/{action_type.value}_fraction",
                rm.action_type_counts.get(action_type.value, 0) / action_total,
                step,
            )
    if rm.joker_order_money_flags:
        # Money mode is only ever taken on a play the harness proved would
        # clear the blind anyway, so chips_forgone is the price of that cash
        # and should stay small relative to the target it still met.
        _write_means(
            writer,
            {
                "joker_order/money_objective_fraction": rm.joker_order_money_flags,
                "joker_order/clearing_play_fraction": rm.joker_order_clear_flags,
                "joker_order/dollars_gained_per_play": rm.joker_order_dollars_gained,
                "joker_order/chips_forgone_per_play": rm.joker_order_chips_forgone,
            },
            step,
        )
    if rm.steps_since_progress:
        writer.add_scalar("rollout/no_progress_streak_p95", float(np.percentile(rm.steps_since_progress, 95)), step)
        writer.add_scalar("rollout/no_progress_streak_max", float(np.max(rm.steps_since_progress)), step)


def _write_terminal_loss_metrics(writer, rm: _RolloutMetrics, step: int, *, win_ante: int) -> None:
    """Write a compact, numeric diagnosis of non-stall episode losses."""
    loss_count = len(rm.terminal_loss_antes)
    completed_count = len(rm.completed_episode_rewards)
    stall_count = int(sum(rm.completed_episode_stalls))
    nonstall_completed_count = max(completed_count - stall_count, 0)
    ante1_death_count = rm.terminal_loss_antes.count(1)

    writer.add_scalar("terminal/completed_episode_count", float(completed_count), step)
    writer.add_scalar("terminal/stall_episode_count", float(stall_count), step)
    writer.add_scalar("terminal/nonstall_completed_episode_count", float(nonstall_completed_count), step)
    writer.add_scalar("terminal/loss_count", float(loss_count), step)
    writer.add_scalar("terminal/loss_ante/denominator_count", float(loss_count), step)
    writer.add_scalar("terminal/loss_ante/1_count", float(ante1_death_count), step)
    writer.add_scalar(
        "terminal/ante1_death_per_nonstall_completed_episode",
        ante1_death_count / nonstall_completed_count if nonstall_completed_count else 0.0,
        step,
    )

    if loss_count:
        writer.add_scalar("terminal/loss_ante_mean", float(np.mean(rm.terminal_loss_antes)), step)
    for ante in range(1, win_ante + 1):
        fraction = rm.terminal_loss_antes.count(ante) / loss_count if loss_count else 0.0
        writer.add_scalar(f"terminal/loss_ante/{ante}_fraction", fraction, step)
    writer.add_scalar("terminal/loss_blind/denominator_count", float(loss_count), step)
    for blind in ("small", "big", "boss"):
        fraction = rm.terminal_loss_blind_counts[blind] / loss_count if loss_count else 0.0
        writer.add_scalar(f"terminal/loss_blind/{blind}_fraction", fraction, step)

    if rm.terminal_loss_score_ratios:
        writer.add_scalar("terminal/loss_score_ratio_mean", float(np.mean(rm.terminal_loss_score_ratios)), step)
        writer.add_scalar("terminal/loss_score_ratio_p50", float(np.median(rm.terminal_loss_score_ratios)), step)
    if rm.terminal_loss_dollars:
        _write_means(
            writer,
            {
                "terminal/loss_cash_mean": rm.terminal_loss_dollars,
                "terminal/loss_cash_ge_10_fraction": rm.terminal_loss_cash_ge_10,
                "terminal/loss_joker_full_weak_fraction": rm.terminal_loss_joker_full_weak,
            },
            step,
        )
    if rm.terminal_loss_last_play_top1:
        legal_top1_fraction = float(np.mean(rm.terminal_loss_last_play_top1))
        writer.add_scalar("terminal/loss_last_play/top1_fraction", legal_top1_fraction, step)
        writer.add_scalar("terminal/loss_last_play/legal_top1_fraction", legal_top1_fraction, step)
    if rm.terminal_loss_last_play_value_ratios:
        legal_value_ratio = float(np.mean(rm.terminal_loss_last_play_value_ratios))
        writer.add_scalar("terminal/loss_last_play/value_ratio_mean", legal_value_ratio, step)
        writer.add_scalar("terminal/loss_last_play/legal_candidate_value_ratio_mean", legal_value_ratio, step)
    for boss_key, count in rm.terminal_boss_loss_counts.items():
        writer.add_scalar(f"terminal/boss_loss/{boss_key}_count", float(count), step)


def _write_ante1_metrics(writer, rm: _RolloutMetrics, step: int) -> None:
    """Write bounded Ante-1 scoring telemetry with explicit denominators."""

    ante1_death_count = rm.terminal_loss_antes.count(1)
    writer.add_scalar("ante1/death/count", float(ante1_death_count), step)
    for blind in ("small", "big", "boss"):
        writer.add_scalar(f"ante1/blind/{blind}/death_count", float(rm.ante1_death_blind_counts.get(blind, 0)), step)
        writer.add_scalar(f"ante1/blind/{blind}/clear_count", float(rm.ante1_blind_clear_counts.get(blind, 0)), step)

    clear_count = sum(rm.ante1_blind_clear_counts.values())
    writer.add_scalar("ante1/clear/count", float(clear_count), step)
    writer.add_scalar(
        "ante1/clear/hands_used_mean",
        float(np.mean(rm.ante1_clear_hands_used)) if rm.ante1_clear_hands_used else 0.0,
        step,
    )
    writer.add_scalar(
        "ante1/clear/hands_unused_mean",
        float(np.mean(rm.ante1_clear_hands_unused)) if rm.ante1_clear_hands_unused else 0.0,
        step,
    )
    writer.add_scalar(
        "ante1/clear/discards_used_mean",
        float(np.mean(rm.ante1_clear_discards_used)) if rm.ante1_clear_discards_used else 0.0,
        step,
    )

    writer.add_scalar("ante1/play/count", float(rm.ante1_play_count), step)
    realized_count = len(rm.ante1_play_realized_to_remaining_target)
    writer.add_scalar("ante1/play/realized_progress_count", float(realized_count), step)
    writer.add_scalar(
        "ante1/play/realized_score_to_remaining_target_mean",
        float(np.mean(rm.ante1_play_realized_to_remaining_target)) if realized_count else 0.0,
        step,
    )
    comparison_count = len(rm.ante1_conservative_chosen_best_ratios)
    writer.add_scalar("ante1/play/conservative_proxy_comparison_count", float(comparison_count), step)
    writer.add_scalar(
        "ante1/play/conservative_chosen_best_ratio_mean",
        float(np.mean(rm.ante1_conservative_chosen_best_ratios)) if comparison_count else 0.0,
        step,
    )

    writer.add_scalar(
        "ante1/one_hand_clear_proxy/opportunity_count", float(rm.ante1_one_hand_clear_proxy_observed), step
    )
    for outcome, count in (
        ("available", rm.ante1_one_hand_clear_proxy_available),
        ("chosen", rm.ante1_one_hand_clear_proxy_chosen),
        ("missed", rm.ante1_one_hand_clear_proxy_missed),
    ):
        writer.add_scalar(f"ante1/one_hand_clear_proxy/{outcome}_count", float(count), step)

    hand_total = sum(rm.ante1_play_hand_counts.values())
    writer.add_scalar("ante1/hand_type/denominator_count", float(hand_total), step)
    for hand_name in POKER_HAND_NAMES:
        tag_name = hand_name.lower().replace(" ", "_")
        writer.add_scalar(
            f"ante1/hand_type/{tag_name}_share",
            rm.ante1_play_hand_counts.get(hand_name, 0) / hand_total if hand_total else 0.0,
            step,
        )


def _write_risk_calibration_metrics(writer, rm: _RolloutMetrics, step: int) -> None:
    """Write compact calibration of shop danger estimates against survival."""

    if rm.risk_shop_death_briers:
        _write_means(
            writer,
            {
                "strategy/risk/shop_death_brier": rm.risk_shop_death_briers,
                "strategy/risk/predicted_death_mean": rm.risk_shop_death_predictions,
                "strategy/risk/actual_death_rate": rm.risk_shop_death_outcomes,
            },
            step,
        )
        auc = _binary_roc_auc(rm.risk_shop_death_predictions, rm.risk_shop_death_outcomes)
        if auc is not None:
            writer.add_scalar("strategy/risk/death_auc", auc, step)
    if rm.risk_shop_raw_death_briers:
        _write_means(
            writer,
            {
                "strategy/risk/raw_shop_death_brier": rm.risk_shop_raw_death_briers,
                "strategy/risk/raw_predicted_death_mean": rm.risk_shop_raw_death_predictions,
            },
            step,
        )
    if rm.risk_ante1_false_safe_deaths:
        writer.add_scalar(
            "strategy/risk/ante1_false_safe_death_fraction", float(np.mean(rm.risk_ante1_false_safe_deaths)), step
        )


def _record_action_diagnostics(rm: _RolloutMetrics, infos: dict, env_idx: int, *, done: bool) -> None:
    """Collect the decision diagnostics consumed by rollout reporting."""

    def get(key, default=None):
        return _extract_step_info_value(infos, key, env_idx, done=done, default=default)

    def count(key):
        return int(get(key, 0) or 0)

    if get("ante1_play_observed", False):
        rm.ante1_play_count += 1
        hand_name = get("ante1_play_hand", "")
        if hand_name:
            rm.ante1_play_hand_counts[str(hand_name)] += 1
        for key, values in (
            ("ante1_play_realized_to_remaining_target", rm.ante1_play_realized_to_remaining_target),
            ("ante1_conservative_chosen_best_ratio", rm.ante1_conservative_chosen_best_ratios),
        ):
            value = get(key)
            if value is not None:
                values.append(float(value))
        if get("ante1_one_hand_clear_proxy_observed", False):
            rm.ante1_one_hand_clear_proxy_observed += 1
            rm.ante1_one_hand_clear_proxy_available += int(bool(get("ante1_one_hand_clear_proxy_available", False)))
            rm.ante1_one_hand_clear_proxy_chosen += int(bool(get("ante1_one_hand_clear_proxy_chosen", False)))
            rm.ante1_one_hand_clear_proxy_missed += int(bool(get("ante1_one_hand_clear_proxy_missed", False)))
    if get("ante1_blind_cleared", False):
        rm.ante1_blind_clear_counts[str(get("ante1_blind_clear_type", "") or "").lower()] += 1
        for name, values in (
            ("hands_used", rm.ante1_clear_hands_used),
            ("hands_unused", rm.ante1_clear_hands_unused),
            ("discards_used", rm.ante1_clear_discards_used),
        ):
            values.append(float(get(f"ante1_blind_clear_{name}", 0)))
    if get("joker_replacement_sequence", False):
        rm.joker_replacement_events += 1
    for key, counter in (
        ("consumable_use_set", rm.consumable_use_set_counts),
        ("pack_claim_set", rm.consumable_claim_set_counts),
        ("shop_bought_consumable_set", rm.consumable_buy_set_counts),
    ):
        consumable_set = get(key, "")
        if consumable_set:
            counter[str(consumable_set)] += 1
    for consumable_set in ("Planet", "Tarot"):
        tag = consumable_set.lower()
        for key, counter in (
            (f"consumable_{tag}_offered_count", rm.consumable_offered_counts),
            (f"consumable_{tag}_claimable_count", rm.consumable_claimable_counts),
            (f"consumable_{tag}_inventory_full_blocked_count", rm.consumable_inventory_full_blocked_counts),
            (f"strategic_{tag}_acquired", rm.consumable_acquired_counts),
            (f"strategic_{tag}_uses", rm.consumable_exact_use_counts),
            (f"strategic_{tag}_pack_auto_uses", rm.consumable_pack_auto_use_counts),
            (f"strategic_{tag}_sold", rm.consumable_sold_counts),
            (f"strategic_{tag}_overwritten", rm.consumable_overwritten_counts),
            (f"strategic_{tag}_expired", rm.consumable_expired_counts),
        ):
            counter[consumable_set] += count(key)
        for name, counter in (
            ("owned_state", rm.consumable_owned_states),
            ("legal_use_opportunity", rm.consumable_legal_use_opportunities),
            ("eligible_offer_opportunity", rm.consumable_eligible_offer_opportunities),
        ):
            counter[consumable_set] += int(bool(get(f"consumable_{tag}_{name}", False)))

    rm.planet_active_plan_owned += int(bool(get("consumable_planet_active_plan_owned", False)))
    rm.planet_active_plan_legal += int(bool(get("consumable_planet_active_plan_legal", False)))
    exact_planet_uses = count("strategic_planet_uses")
    if exact_planet_uses:
        prefix = "planet_use" if get("planet_use_observed", False) else "planet_claim"
        alignment = "matched" if get(f"{prefix}_plan_supported", False) else "unmatched"
        rm.planet_use_alignment_counts[alignment] += exact_planet_uses
        hand_type = str(get(f"{prefix}_hand_type", "") or "")
        if hand_type:
            rm.planet_use_hand_counts[hand_type] += exact_planet_uses
        if get(f"{prefix}_active_plan_match", False):
            rm.planet_active_plan_uses += max(exact_planet_uses - count("strategic_planet_pack_auto_uses"), 0)

    tarot_uses = count("strategic_tarot_uses")
    tarot_family = str(get("strategic_tarot_family", "") or "")
    if tarot_uses and tarot_family:
        rm.tarot_use_family_counts[tarot_family] += tarot_uses
        if tarot_family in {"deck_cut", "rank_fix", "suit_fix"}:
            rm.tarot_fix_reliability_deltas.append(float(get("strategic_tarot_fix_reliability_delta", 0.0)))
        if tarot_family == "cash":
            rm.attributable_cash_payouts.append(float(get("strategic_attributable_cash_payout", 0.0)))

    rm.gold_cards_created += count("strategic_gold_created_tarot") + count("strategic_gold_created_midas")
    rm.held_gold_payout_dollars += count("strategic_held_gold_payout")
    claimed_seal = get("pack_claim_seal", "")
    if claimed_seal:
        rm.pack_claim_seal_counts[str(claimed_seal)] += 1
    for seal in ("Blue", "Purple"):
        rm.pack_offered_seal_counts[seal] += count(f"seal_{seal.lower()}_offered_count")
    rm.blue_seals_activated += count("strategic_blue_seals_activated")
    rm.purple_seals_activated += count("strategic_purple_seals_activated")
    rm.purple_seal_tarots_generated += int(
        get("strategic_purple_tarots_generated", get("purple_seal_tarot_generated_count", 0))
    )
    rm.blue_seal_planets_generated += int(
        get("strategic_blue_planets_generated", get("blue_seal_planet_generated_count", 0))
    )

    if get("shop_leave_observed", False):
        rm.shop_leave_flags.append(1.0)
        for key, values in (
            ("shop_unsafe_leave", rm.shop_unsafe_leave_flags),
            ("shop_unsafe_can_reroll", rm.shop_unsafe_can_reroll_flags),
            ("shop_missed_confident_upgrade", rm.shop_missed_upgrade_flags),
            ("shop_leave_joker_full_weak", rm.shop_leave_full_weak_flags),
        ):
            values.append(float(get(key, False)))
        upgrade_delta = get("shop_best_confident_upgrade_delta")
        if upgrade_delta is not None:
            rm.shop_best_upgrade_deltas.append(float(upgrade_delta))

    if get("hand_play_observed", False) and get("hand_play_in_candidates", False):
        chosen_hand = get("hand_play_chosen_hand", "")
        if chosen_hand:
            rm.hand_chosen_counts[str(chosen_hand)] += 1
    if get("shop_joker_offer_observed", False):
        offered_count = int(get("shop_offered_joker_emitted_count", 0))
        for index in range(min(offered_count, MAX_DIAGNOSTIC_SHOP_JOKERS)):
            center = get(f"shop_offered_joker_{index}_id", "")
            if center:
                rm.shop_offered_joker_counts[str(center)] += 1
    for key, counter in (
        ("shop_bought_joker_id", rm.shop_bought_joker_counts),
        ("shop_sold_joker_id", rm.shop_sold_joker_counts),
    ):
        center = get(key, "")
        if center:
            counter[str(center)] += 1
    if get("joker_roster_changed", False):
        rm.joker_churn_count += int(get("joker_churn_count", 0))
        rm.joker_replacement_events += int(bool(get("joker_replacement_event", False)))

    if get("build_diagnostics_observed", False):
        plan_type = get("hand_plan_post_type", "")
        if plan_type:
            rm.hand_plan_type_counts[str(plan_type)] += 1
        for name, values in (
            ("reliability", rm.hand_plan_reliability),
            ("readiness", rm.hand_plan_readiness),
        ):
            value = get(f"hand_plan_post_{name}")
            if value is not None:
                values.append(float(value))
        for component in _REPORTED_POTENTIAL_COMPONENTS:
            value = get(f"potential_post_{component}")
            if value is not None:
                rm.potential_values[f"post_{component}"].append(float(value))


def _safe_mean(values: list[float]) -> float:
    """Return the mean of a list or NaN when empty."""
    return float(np.mean(values)) if values else float("nan")


def _action_type_name(action_id: int) -> str:
    """Map a flat action id to a stable TensorBoard-friendly action type name."""
    return decode_action(int(action_id)).action_type.value


@dataclass
class EpisodeHistory:
    """Completed-episode telemetry shared by collection and reporting."""

    rewards: list[float] = field(default_factory=list)
    lengths: list[int] = field(default_factory=list)
    wins: list[bool] = field(default_factory=list)
    stalls: list[bool] = field(default_factory=list)
    antes: list[int] = field(default_factory=list)
    tarot_uses: list[int] = field(default_factory=list)
    origin_wins: dict[str, list[float]] = field(default_factory=lambda: {"fresh": [], "archive": []})
    continuation_survival: dict[int, list[float]] = field(default_factory=lambda: {a: [] for a in range(1, 9)})


def write_rollout_metrics(
    writer,
    rm: _RolloutMetrics,
    buffer,
    history: EpisodeHistory,
    *,
    win_ante: int,
    step: int,
) -> None:
    """Report one completed rollout without changing training state."""
    # Explained variance of the composed critic on this rollout:
    # 1 - Var(returns - values) / Var(returns).
    flat_returns = buffer._flat_returns
    explained_variance = float("nan")
    if len(flat_returns) > 1:
        var_returns = float(np.var(flat_returns))
        if var_returns > 1e-9:
            explained_variance = 1.0 - float(np.var(flat_returns - buffer._flat_values)) / var_returns
    writer.add_scalar("ppo/explained_variance", explained_variance, step)
    # Which rollout rows the main-loop outcome NLL can actually train
    # on, and how much shallower they are than the rows it drops.
    for label_key, label_value in buffer.outcome_label_diagnostics().items():
        writer.add_scalar(f"critic/rollout_labels/{label_key}", label_value, step)
    _write_means(
        writer,
        {
            "rollout/step_reward_mean": rm.step_rewards,
            "rollout/reward_mean": rm.step_rewards,
            "rollout/chosen_action_prob_mean": rm.chosen_action_probs,
            "rollout/max_action_type_prob_mean": rm.max_action_type_probs,
            "rollout/done_rate": rm.done_flags,
            "rollout/progress_rate": rm.progress_flags,
        },
        step,
    )
    writer.add_scalar("rollout/completed_episodes", float(np.sum(rm.done_flags)), step)
    _write_rollout_episode_metrics(writer, rm, step)
    _write_action_behavior_metrics(writer, rm, step)
    _write_terminal_loss_metrics(writer, rm, step, win_ante=win_ante)
    _write_ante1_metrics(writer, rm, step)
    if rm.clear_probabilities:
        _write_means(
            writer,
            {
                "strategy/risk/clear_probability_mean": rm.clear_probabilities,
                "strategy/risk/immediate_death_probability_mean": rm.immediate_death_probabilities,
            },
            step,
        )
    if rm.shop_leave_flags:
        _write_means(
            writer,
            {
                "shop/unsafe_leave_fraction": rm.shop_unsafe_leave_flags,
                "shop/unsafe_can_reroll_fraction": rm.shop_unsafe_can_reroll_flags,
                "shop/missed_confident_upgrade_fraction": rm.shop_missed_upgrade_flags,
                "shop/full_weak_leave_fraction": rm.shop_leave_full_weak_flags,
            },
            step,
        )
    if rm.shop_best_upgrade_deltas:
        writer.add_scalar("shop/best_confident_upgrade_delta_mean", float(np.mean(rm.shop_best_upgrade_deltas)), step)
    if rm.shop_survival_briers:
        _write_means(
            writer,
            {
                "critic/shop_survival_brier": rm.shop_survival_briers,
                "critic/shop_survival_prediction_mean": rm.shop_survival_predictions,
                "critic/shop_survival_outcome_mean": rm.shop_survival_outcomes,
            },
            step,
        )
    _write_risk_calibration_metrics(writer, rm, step)
    if history.rewards:
        _write_means(
            writer,
            {
                "recent_100/episode_reward_mean": history.rewards[-100:],
                "recent_100/episode_length_mean": history.lengths[-100:],
                "recent_100/win_rate": history.wins[-100:],
                "recent_100/stall_rate": history.stalls[-100:],
                "recent_100/final_ante_mean": history.antes[-100:],
                "recent_100/tarot_uses_per_completed_episode_mean": history.tarot_uses[-100:],
            },
            step,
        )
    hand_total = sum(rm.hand_chosen_counts.values())
    if hand_total:
        for hand_name, count in rm.hand_chosen_counts.items():
            tag_name = str(hand_name).lower().replace(" ", "_")
            writer.add_scalar(f"rollout/hands_played/{tag_name}", count / hand_total, step)
    steps_per_thousand = max(len(rm.step_rewards) / 1000.0, 1e-9)
    writer.add_scalar(
        "shop/joker_offers_per_1k_steps", sum(rm.shop_offered_joker_counts.values()) / steps_per_thousand, step
    )
    writer.add_scalar(
        "shop/joker_buys_per_1k_steps", sum(rm.shop_bought_joker_counts.values()) / steps_per_thousand, step
    )
    writer.add_scalar(
        "shop/joker_sells_per_1k_steps", sum(rm.shop_sold_joker_counts.values()) / steps_per_thousand, step
    )
    completed_episodes = len(rm.completed_episode_rewards)
    if completed_episodes:
        writer.add_scalar("joker/replacements_per_episode", rm.joker_replacement_events / completed_episodes, step)
        writer.add_scalar("joker/churn_per_episode", rm.joker_churn_count / completed_episodes, step)
    for consumable_set in ("Planet", "Tarot"):
        tag_name = consumable_set.lower()
        writer.add_scalar(
            f"rollout/shop_buys/{tag_name}_per_1k_steps",
            rm.consumable_buy_set_counts[consumable_set] / steps_per_thousand,
            step,
        )
        writer.add_scalar(
            f"rollout/pack_claims/{tag_name}_per_1k_steps",
            rm.consumable_claim_set_counts[consumable_set] / steps_per_thousand,
            step,
        )
        writer.add_scalar(
            f"rollout/uses/{tag_name}_per_1k_steps",
            rm.consumable_use_set_counts[consumable_set] / steps_per_thousand,
            step,
        )
    for seal in ("Blue", "Purple", "Gold", "Red"):
        writer.add_scalar(
            f"strategy/seals/claims/{seal.lower()}_per_1k_steps",
            rm.pack_claim_seal_counts[seal] / steps_per_thousand,
            step,
        )
    writer.add_scalar(
        "strategy/seals/purple_tarots_generated_per_1k_steps",
        rm.purple_seal_tarots_generated / steps_per_thousand,
        step,
    )
    writer.add_scalar(
        "strategy/seals/blue_planets_generated_per_1k_steps", rm.blue_seal_planets_generated / steps_per_thousand, step
    )
    _write_consumable_strategy_metrics(writer, rm, step)
    plan_total = sum(rm.hand_plan_type_counts.values())
    if plan_total:
        for hand_name, count in rm.hand_plan_type_counts.items():
            tag_name = hand_name.lower().replace(" ", "_")
            writer.add_scalar(f"strategy/hand_plan/{tag_name}_share", count / plan_total, step)
    if rm.hand_plan_reliability:
        writer.add_scalar("strategy/hand_plan/reliability_mean", float(np.mean(rm.hand_plan_reliability)), step)
    if rm.hand_plan_readiness:
        writer.add_scalar("strategy/hand_plan/readiness_mean", float(np.mean(rm.hand_plan_readiness)), step)
    for component in _REPORTED_POTENTIAL_COMPONENTS:
        values = rm.potential_values.get(f"post_{component}")
        if values:
            writer.add_scalar(f"strategy/potential/{component}_mean", float(np.mean(values)), step)
    for component_name, values in rm.reward_component_values.items():
        if values:
            writer.add_scalar(f"rollout/reward_components/{component_name}", float(np.mean(values)), step)
