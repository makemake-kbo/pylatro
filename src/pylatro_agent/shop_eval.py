"""Context-aware build and shop evaluation shared by the reward function.

The reward shaping in :mod:`pylatro_agent.reward` operates on serialized ``info``
dicts captured at each environment step (it never sees the live ``RunState``).
This module provides:

* :func:`capture_build_features`, called from the env / fast-runner info
  builders to attach build, economy, slot, deck, and shop-card fields to the
  ``info`` dict. Centralizing it keeps PPO (BalatroEnv) and BC pretraining
  (fast_generate) emitting identical fields so the value head is consistent.
* :func:`evaluate_build`, :func:`evaluate_shop_opportunity`,
  :func:`score_shop_item`, pure functions over the ``info`` dict that score
  the current build and shop, used for delta/gated reward shaping.

Scoring is context-aware: a joker's value depends on the captured deck, hand,
blind, slot, and ordered build state. The pure estimator values modeled score
effects directly. Heuristic classifications remain only for compatibility and
non-scoring economy metadata.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from math import log, log1p
from typing import TYPE_CHECKING, Any

from pylatro import get_blind_amount
from pylatro.runtime import consumable_limit, joker_limit

from .build_value import BuildValueEstimate, JokerMarginal, estimate_build_value
from .heuristic import (
    _ECONOMY_JOKERS,
    _ECONOMY_SCORES,
    _RETRIGGER_JOKER_KEYS,
    _SCALING_JOKER_KEYS,
    _SCALING_XMULT_JOKER_KEYS,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pylatro.models import JokerInstance, RunState

# ───────────────────────────── info capture ─────────────────────────────


def _config_dict(value: Any, *, copy: bool = False) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return deepcopy(value) if copy else value


def _econ_dollars(cfg: dict) -> float:
    """Best-effort per-round dollar yield encoded in a joker config."""
    total = 0.0
    d = cfg.get("dollars")
    if isinstance(d, (int, float)):
        total += float(d)
    extra = cfg.get("extra")
    if isinstance(extra, dict):
        ed = extra.get("dollars")
        if isinstance(ed, (int, float)):
            total += float(ed)
    return total


def _joker_summary(
    center: dict,
    key: str,
    live: JokerInstance | None,
    *,
    edition: dict[str, bool] | None = None,
    eternal: bool = False,
    perishable: bool = False,
    rental: bool = False,
) -> dict[str, Any]:
    cfg = _config_dict(center.get("config"), copy=True)
    extra_cfg = _config_dict(cfg.get("extra"))
    base_x = cfg.get("Xmult")
    base_x = float(base_x) if isinstance(base_x, (int, float)) else 1.0
    extra_x = extra_cfg.get("Xmult") if isinstance(extra_cfg, dict) else None
    if isinstance(extra_x, (int, float)) and float(extra_x) > base_x:
        base_x = float(extra_x)

    summary: dict[str, Any] = {
        "key": key,
        "name": center.get("name", ""),
        "effect": center.get("effect", ""),
        "rarity": int(center.get("rarity", 0) or 0),
        "type": cfg.get("type") or "",
        "config": cfg,
        "base_mult": float(cfg.get("mult") or 0),
        "base_x_mult": base_x,
        "base_t_mult": float(cfg.get("t_mult") or 0),
        "base_t_chips": float(cfg.get("t_chips") or 0),
        "dollars": _econ_dollars(cfg),
        "h_size": float(cfg.get("h_size") or 0),
        "d_size": float(cfg.get("d_size") or 0),
        "is_economy": key in _ECONOMY_JOKERS,
        "is_scaling": key in _SCALING_JOKER_KEYS,
        "is_scaling_xmult": key in _SCALING_XMULT_JOKER_KEYS,
        "is_retrigger": key in _RETRIGGER_JOKER_KEYS,
        # Static center capability used when Blueprint/Brainstorm resolves its
        # target. ``live.blueprint_compat`` is a UI status on the copying joker
        # itself and must not replace this target capability.
        "copy_compatible": bool(center.get("blueprint_compat")),
    }
    if live is not None:
        summary.update(
            {
                "sell_cost": int(live.sell_cost),
                "edition": deepcopy(live.edition) if live.edition else {},
                "eternal": bool(live.eternal),
                "perishable": bool(live.perishable),
                "perish_tally": live.perish_tally,
                "rental": bool(live.rental),
                "debuffed": bool(live.debuff),
                "mult": float(live.mult),
                "h_mult": float(live.h_mult),
                "h_x_mult": float(live.h_x_mult),
                "h_dollars": float(live.h_dollars),
                "p_dollars": float(live.p_dollars),
                "t_mult": float(live.t_mult),
                "t_chips": float(live.t_chips),
                "x_mult": float(live.x_mult),
                "h_size": float(live.h_size),
                "d_size": float(live.d_size),
                "extra": deepcopy(live.extra),
                "extra_value": int(live.extra_value),
                "hands_played_at_create": int(live.hands_played_at_create),
                "invis_rounds": int(live.invis_rounds),
                "caino_xmult": float(live.caino_xmult),
                "yorick_discards": int(live.yorick_discards),
                "loyalty_remaining": int(live.loyalty_remaining),
                "driver_tally": int(live.driver_tally),
                "stone_tally": int(live.stone_tally),
                "steel_tally": int(live.steel_tally),
                "to_do_poker_hand": live.to_do_poker_hand,
                "blueprint_compat": live.blueprint_compat,
                "money": int(live.money),
                "getting_sliced": bool(live.getting_sliced),
                "nine_tally": int(live.nine_tally),
            }
        )
    else:
        summary.update(
            {
                "sell_cost": 0,
                "edition": deepcopy(edition) if edition else {},
                "eternal": bool(eternal),
                "perishable": bool(perishable),
                "perish_tally": None,
                "rental": bool(rental),
                "debuffed": False,
                "mult": summary["base_mult"],
                "h_mult": float(cfg.get("h_mult") or 0),
                "h_x_mult": float(cfg.get("h_x_mult") or 0),
                "h_dollars": float(cfg.get("h_dollars") or 0),
                "p_dollars": float(cfg.get("p_dollars") or 0),
                "t_mult": summary["base_t_mult"],
                "t_chips": summary["base_t_chips"],
                "x_mult": summary["base_x_mult"],
                "extra": deepcopy(cfg.get("extra")),
                "extra_value": 0,
                "hands_played_at_create": 0,
                "invis_rounds": 0,
                "caino_xmult": 1.0,
                "yorick_discards": 0,
                "loyalty_remaining": 0,
                "driver_tally": 0,
                "stone_tally": 0,
                "steel_tally": 0,
                "to_do_poker_hand": None,
                "blueprint_compat": center.get("blueprint_compat"),
                "money": 0,
                "getting_sliced": False,
                "nine_tally": 0,
            }
        )
    return summary


def _consumable_summary(center: dict, key: str, live) -> dict[str, Any]:
    cfg = _config_dict(center.get("config"))
    return {
        "key": key,
        "name": center.get("name", ""),
        "set": center.get("set", ""),
        "sell_cost": int(getattr(live, "sell_cost", 0) or 0),
        "hand_type": (cfg.get("hand_type") or cfg.get("type") or "") if isinstance(cfg, dict) else "",
    }


def _deck_stats(state: RunState) -> dict[str, Any]:
    rank_counts: dict[str, int] = {}
    suit_counts: dict[str, int] = {}
    rank_suit_counts: dict[tuple[str, str], int] = {}
    enhancement_counts: dict[str, int] = {}
    rank_enhancement_counts: dict[tuple[str, str], int] = {}
    suit_enhancement_counts: dict[tuple[str, str], int] = {}
    rank_suit_enhancement_counts: dict[tuple[str, str, str], int] = {}
    card_signature_counts: dict[tuple[str, str, str, str], int] = {}
    seal_counts: dict[str, int] = {}
    edition_counts: dict[str, int] = {}
    cards: list[dict[str, Any]] = []
    centers = state.data.centers
    for card in state.deck_cards:
        center = centers.get(card.center_key, {})
        config = _config_dict(center.get("config"))
        effect = str(center.get("effect", "") or "") if card.center_key != "c_base" else ""
        rank_counts[card.rank] = rank_counts.get(card.rank, 0) + 1
        suit_counts[card.suit] = suit_counts.get(card.suit, 0) + 1
        rank_suit = (card.rank, card.suit)
        rank_suit_counts[rank_suit] = rank_suit_counts.get(rank_suit, 0) + 1
        if effect:
            enhancement_counts[effect] = enhancement_counts.get(effect, 0) + 1
            rank_effect = (card.rank, effect)
            suit_effect = (card.suit, effect)
            rank_enhancement_counts[rank_effect] = rank_enhancement_counts.get(rank_effect, 0) + 1
            suit_enhancement_counts[suit_effect] = suit_enhancement_counts.get(suit_effect, 0) + 1
        rank_suit_effect = (card.rank, card.suit, effect)
        signature = (card.rank, card.suit, effect, card.seal or "")
        rank_suit_enhancement_counts[rank_suit_effect] = rank_suit_enhancement_counts.get(rank_suit_effect, 0) + 1
        card_signature_counts[signature] = card_signature_counts.get(signature, 0) + 1
        if card.seal:
            seal_counts[card.seal] = seal_counts.get(card.seal, 0) + 1
        if card.edition_key:
            edition_counts[card.edition_key] = edition_counts.get(card.edition_key, 0) + 1
        cards.append(
            {
                "rank": card.rank,
                "suit": card.suit,
                "enhancement": effect,
                "center_key": card.center_key,
                "seal": card.seal or "",
                "edition": card.edition_key or "",
                "bonus": float(config.get("bonus", 0) or 0),
                "mult": float(config.get("mult", 0) or 0),
                "x_mult": float(config.get("Xmult", 1) or 1),
                "h_mult": float(config.get("h_mult", 0) or 0),
                "h_x_mult": float(config.get("h_x_mult", 0) or 0),
                "perma_bonus": int(card.perma_bonus),
                "debuffed": bool(card.debuff),
            }
        )
    return {
        "size": len(state.deck_cards),
        "cards": tuple(cards),
        "card_descriptors": tuple(cards),
        "rank_counts": rank_counts,
        "suit_counts": suit_counts,
        "rank_suit_counts": rank_suit_counts,
        "enhancement_counts": enhancement_counts,
        "rank_enhancement_counts": rank_enhancement_counts,
        "suit_enhancement_counts": suit_enhancement_counts,
        "rank_suit_enhancement_counts": rank_suit_enhancement_counts,
        "card_signature_counts": card_signature_counts,
        "seal_counts": seal_counts,
        "edition_counts": edition_counts,
        "stone_count": enhancement_counts.get("Stone Card", 0),
        "steel_count": enhancement_counts.get("Steel Card", 0),
        "glass_count": enhancement_counts.get("Glass Card", 0),
        "gold_count": enhancement_counts.get("Gold Card", 0),
        "wild_count": enhancement_counts.get("Wild Card", 0),
    }


def _shop_card_summary(state: RunState, item, index: int) -> dict[str, Any]:
    center = state.data.centers.get(item.center_key, {})
    cfg = _config_dict(center.get("config"))
    card_set = center.get("set", "")
    edition = deepcopy(getattr(item, "edition", None)) if getattr(item, "edition", None) else {}
    eternal = bool(getattr(item, "eternal", False))
    perishable = bool(getattr(item, "perishable", False))
    rental = bool(getattr(item, "rental", False))
    detail: dict[str, Any] = {
        "index": index,
        "key": item.center_key,
        "name": center.get("name", ""),
        "set": card_set,
        "cost": int(getattr(item, "cost", 0) or 0),
        "rarity": int(center.get("rarity", 0) or 0),
        "edition": edition,
        "eternal": eternal,
        "perishable": perishable,
        "rental": rental,
        "hand_type": cfg.get("hand_type") or cfg.get("type") or "",
    }
    if card_set == "Joker":
        detail["joker"] = _joker_summary(
            center,
            item.center_key,
            None,
            edition=edition,
            eternal=eternal,
            perishable=perishable,
            rental=rental,
        )
    return detail


def _upcoming_blind_target(state: RunState) -> int:
    blind_type = state.blind_on_deck or "Small"
    blind_key = state.round_resets.blind_choices.get(blind_type, "")
    blind = state.data.blinds.get(blind_key, {}) if blind_key else (state.round_resets.blind or {})
    base = get_blind_amount(state.round_resets.ante, min(state.stake, 3))
    return int(float(base) * float(blind.get("mult", 1) or 1))


def capture_build_features(state: RunState) -> dict[str, Any]:
    """Build/economy/slot/deck/shop fields to merge into the step ``info`` dict.

    Called by both BalatroEnv (PPO) and fast_generate (BC) so the reward sees
    identical fields on both code paths.
    """
    jl = joker_limit(state)
    cl = consumable_limit(state)
    n_jokers = len(state.jokers)
    n_cons = len(state.consumables)
    cap = int(state.interest_cap)
    amount = int(state.interest_amount)
    dollars = int(state.dollars)
    centers = state.data.centers

    shop_items = list(state.shop.cards) + list(state.shop.vouchers) + list(state.shop.boosters)

    return {
        "joker_slots_used": n_jokers,
        "joker_slots_limit": jl,
        "joker_slots_left": max(0, jl - n_jokers),
        "consumable_slots_used": n_cons,
        "consumable_slots_limit": cl,
        "consumable_slots_left": max(0, cl - n_cons),
        "interest_cap_cash": cap,
        "interest_amount": amount,
        "interest_tiers": min(dollars // 5, cap // 5),
        "interest_tiers_cap": cap // 5,
        "interest_gap_cash": max(0, cap - dollars),
        "joker_details": tuple(
            _joker_summary(centers.get(j.center_key, {}), j.center_key, j) for j in state.jokers
        ),
        "consumable_details": tuple(
            _consumable_summary(centers.get(c.center_key, {}), c.center_key, c) for c in state.consumables
        ),
        "deck_stats": _deck_stats(state),
        "hand_levels": {name: int(h.get("level", 1) or 1) for name, h in state.hands.items()},
        "hand_play_counts": {name: int(h.get("played", 0) or 0) for name, h in state.hands.items()},
        "hand_details": {
            name: {
                "chips": float(h.get("chips", 0) or 0),
                "mult": float(h.get("mult", 0) or 0),
                "level": int(h.get("level", 1) or 1),
                "played": int(h.get("played", 0) or 0),
                "played_this_round": int(h.get("played_this_round", 0) or 0),
                "visible": bool(h.get("visible", False)),
            }
            for name, h in state.hands.items()
        },
        "idol_card": deepcopy(state.current_round.idol_card),
        "ancient_card": deepcopy(state.current_round.ancient_card),
        "castle_card": deepcopy(state.current_round.castle_card),
        "mail_card": deepcopy(state.current_round.mail_card),
        "dynamic_targets": {
            "idol_card": deepcopy(state.current_round.idol_card),
            "ancient_card": deepcopy(state.current_round.ancient_card),
            "castle_card": deepcopy(state.current_round.castle_card),
            "mail_card": deepcopy(state.current_round.mail_card),
        },
        "hands_available": max(1, state.current_round.hands_left or state.round_resets.hands),
        "hand_size": int(state.current_round.hand_size or state.starting_params.hand_size),
        "blind_target": _upcoming_blind_target(state),
        "used_voucher_keys": tuple(state.used_vouchers.keys()),
        "shop_cards": tuple(_shop_card_summary(state, item, i) for i, item in enumerate(shop_items)),
    }


# ───────────────────────────── evaluation ─────────────────────────────


def interest_tiers(dollars: int, interest_cap_cash: int) -> int:
    return min(max(0, dollars) // 5, max(0, interest_cap_cash) // 5)


@dataclass(frozen=True)
class JokerScoreBreakdown:
    key: str
    total: float
    immediate_score: float = 0.0
    xmult_score: float = 0.0
    scaling_score: float = 0.0
    economy_score: float = 0.0
    retrigger_score: float = 0.0
    hand_type_synergy_score: float = 0.0
    edition_score: float = 0.0
    slot_pressure_penalty: float = 0.0
    duplicate_penalty: float = 0.0
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ConsumableScoreBreakdown:
    key: str
    total: float
    planet_alignment: float = 0.0
    deck_fixing: float = 0.0
    money_value: float = 0.0
    spectral_power: float = 0.0
    risk_penalty: float = 0.0
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class BuildEval:
    total: float
    score_power: float
    xmult_power: float
    scaling_power: float
    economy_power: float
    retrigger_power: float
    hand_alignment: float
    survival_margin: float

    joker_slots_used: int
    joker_slots_limit: int
    joker_slots_left: int

    has_scoring_joker: bool
    has_xmult_joker: bool
    has_scaling_joker: bool
    has_economy_engine: bool
    has_retrigger_engine: bool

    main_hand_type: str | None
    main_hand_confidence: float

    interest_cap_cash: int
    dollars: int
    interest_tiers: int
    interest_tiers_cap: int
    interest_gap_cash: int

    estimated_score: float
    no_joker_baseline_score: float
    required_score_per_hand: float
    readiness_ratio: float
    joker_marginal_score_ratios: tuple[float, ...]
    build_value: BuildValueEstimate


@dataclass(frozen=True)
class ShopOpportunity:
    best_visible_upgrade: float = 0.0
    best_visible_key: str | None = None
    best_visible_kind: str | None = None
    best_visible_cost: int = 0
    best_visible_net_value: float = 0.0
    best_replacement_index: int | None = None
    best_replacement_key: str | None = None
    best_score_delta: float = 0.0

    has_affordable_upgrade: bool = False
    has_critical_upgrade: bool = False

    reroll_desirable: bool = False
    reroll_cost: int = 0
    dollars_after_reroll: int = 0
    lost_interest_tiers_if_reroll: int = 0
    can_reroll_above_interest_cap: bool = False


def _main_hand(info: dict[str, Any]) -> tuple[str | None, float]:
    counts = info.get("hand_play_counts") or {}
    total = sum(counts.values())
    if total <= 0:
        return None, 0.0
    best_type = max(counts, key=lambda k: counts[k])
    return best_type, counts[best_type] / total


def _is_scoring_marginal(marginal: JokerMarginal) -> bool:
    return marginal.score_ratio > 1.01 and marginal.modeled_effect_fraction > 0


def _economy_power(jokers: Sequence[dict[str, Any]], ante: int) -> float:
    total = 0.0
    for joker in jokers:
        if joker.get("debuffed"):
            continue
        dollars = float(joker.get("dollars", 0) or 0)
        total += dollars * 0.08
        if joker.get("is_economy") and dollars <= 0:
            total += min(0.25, _ECONOMY_SCORES.get(joker.get("key", ""), 0.0) * 0.01)
    return total * (1.2 if ante <= 3 else 1.0)


def _candidate_score_value(current: BuildValueEstimate, candidate: BuildValueEstimate) -> float:
    if candidate.representative_score_per_hand <= current.representative_score_per_hand:
        return 0.0
    current_score = max(1.0, current.representative_score_per_hand)
    baseline = max(1.0, current.no_joker_baseline_score)
    delta = candidate.representative_score_per_hand - current_score
    return log(candidate.representative_score_per_hand / current_score) + 0.05 * log1p(delta / baseline)


def _is_negative(joker: dict[str, Any]) -> bool:
    edition = joker.get("edition")
    return isinstance(edition, dict) and bool(edition.get("negative"))


def _candidate_build_estimate(
    info: dict[str, Any],
    candidate: dict[str, Any],
) -> tuple[BuildValueEstimate | None, int | None]:
    owned = tuple(info.get("joker_details") or ())
    slots_left = int(info.get("joker_slots_left", 0) or 0)
    if slots_left > 0 or _is_negative(candidate):
        return estimate_build_value(info, (*owned, candidate)), None

    best: BuildValueEstimate | None = None
    best_index: int | None = None
    for index, joker in enumerate(owned):
        if joker.get("eternal"):
            continue
        trial = owned[:index] + owned[index + 1 :] + (candidate,)
        estimate = estimate_build_value(info, trial)
        if best is None or estimate.representative_score_per_hand > best.representative_score_per_hand:
            best = estimate
            best_index = index
    return best, best_index


def evaluate_build(info: dict[str, Any]) -> BuildEval:
    """Score the current owned build from a captured ``info`` dict."""
    jokers = tuple(info.get("joker_details") or ())
    estimate = estimate_build_value(info, jokers)
    cap = int(info.get("interest_cap_cash", 25) or 25)
    dollars = int(info.get("dollars", 0) or 0)
    ante = int(info.get("ante", 1) or 1)
    main_type, main_conf = _main_hand(info)
    main_type = main_type or estimate.representative_hand_type

    scoring_marginals = tuple(m for m in estimate.joker_marginals if _is_scoring_marginal(m))
    has_scoring = bool(scoring_marginals)
    has_xmult = any(m.channels.x_mult > 1.001 and _is_scoring_marginal(m) for m in estimate.joker_marginals)
    has_scaling = any(j.get("is_scaling") and not j.get("debuffed") for j in jokers)
    has_economy = any(j.get("is_economy") and not j.get("debuffed") for j in jokers)
    has_retrigger = any(m.channels.retriggers > 0 for m in estimate.joker_marginals)

    baseline = max(1.0, estimate.no_joker_baseline_score)
    score_power = max(0.0, estimate.representative_score_per_hand / baseline - 1.0)
    xmult_power = max(0.0, estimate.channels.x_mult - 1.0)
    scaling_power = sum(
        max(0.0, log(max(1.0, marginal.score_ratio)))
        for marginal, joker in zip(estimate.joker_marginals, jokers, strict=True)
        if joker.get("is_scaling")
    )
    economy_power = _economy_power(jokers, ante)
    retrigger_power = max(0.0, estimate.channels.retriggers) * 0.1
    hand_align = sum(
        max(0.0, log(max(1.0, marginal.score_ratio)))
        for marginal, joker in zip(estimate.joker_marginals, jokers, strict=True)
        if joker.get("type") and joker.get("type") == main_type
    )
    survival_margin = estimate.readiness_ratio
    deck = info.get("deck_stats") or {}
    has_full_context = bool(info.get("hand_details")) and bool(deck.get("cards") or deck.get("card_descriptors"))
    if estimate.required_score_per_hand <= 0 or not has_full_context:
        engine = score_power + xmult_power + retrigger_power
        survival_margin = engine / max(1.0, ante * 0.6) if has_scoring else min(engine, 0.5)

    return BuildEval(
        total=score_power + economy_power,
        score_power=score_power,
        xmult_power=xmult_power,
        scaling_power=scaling_power,
        economy_power=economy_power,
        retrigger_power=retrigger_power,
        hand_alignment=hand_align,
        survival_margin=survival_margin,
        joker_slots_used=int(info.get("joker_slots_used", len(jokers)) or 0),
        joker_slots_limit=int(info.get("joker_slots_limit", 0) or 0),
        joker_slots_left=int(info.get("joker_slots_left", 0) or 0),
        has_scoring_joker=has_scoring,
        has_xmult_joker=has_xmult,
        has_scaling_joker=has_scaling,
        has_economy_engine=has_economy,
        has_retrigger_engine=has_retrigger,
        main_hand_type=main_type,
        main_hand_confidence=main_conf,
        interest_cap_cash=cap,
        dollars=dollars,
        interest_tiers=interest_tiers(dollars, cap),
        interest_tiers_cap=cap // 5,
        interest_gap_cash=max(0, cap - dollars),
        estimated_score=estimate.representative_score_per_hand,
        no_joker_baseline_score=estimate.no_joker_baseline_score,
        required_score_per_hand=estimate.required_score_per_hand,
        readiness_ratio=estimate.readiness_ratio,
        joker_marginal_score_ratios=estimate.joker_marginal_score_ratios,
        build_value=estimate,
    )


# Hand-type alignment for planets / hand-specific fixing.
_TAROT_SUIT_FIX = {
    "c_sun": "Hearts",
    "c_moon": "Clubs",
    "c_star": "Diamonds",
    "c_world": "Spades",
}


def score_consumable_item(info: dict[str, Any], summary: dict[str, Any], build: BuildEval) -> ConsumableScoreBreakdown:
    """Value a tarot/planet/spectral by what it fixes or creates for this build."""
    key = summary["key"]
    cset = summary.get("set", "")
    planet_alignment = deck_fixing = money_value = spectral_power = risk = 0.0
    notes: list[str] = []

    if cset == "Planet":
        htype = summary.get("hand_type") or summary.get("type") or ""
        if htype and htype == build.main_hand_type:
            planet_alignment = 0.30 + 0.20 * build.main_hand_confidence
            notes.append("main_hand_planet")
        elif htype and (info.get("hand_play_counts") or {}).get(htype, 0) > 0:
            planet_alignment = 0.10
        else:
            planet_alignment = 0.02
    elif cset == "Tarot":
        deck = info.get("deck_stats") or {}
        suit_counts = deck.get("suit_counts", {})
        # Suit-fixing tarots help flush builds; reward when one suit dominates.
        if key in _TAROT_SUIT_FIX and suit_counts:
            top = max(suit_counts.values()) if suit_counts else 0
            size = max(1, deck.get("size", 1))
            if top / size >= 0.4:
                deck_fixing = 0.18
                notes.append("suit_fix")
            else:
                deck_fixing = 0.06
        else:
            deck_fixing = 0.08
        money_value = 0.04
    elif cset == "Spectral":
        # High variance: modest positive, capped, penalize when build is already safe.
        spectral_power = 0.15
        if build.survival_margin >= 1.5:
            risk = -0.08
            notes.append("spectral_risk_when_safe")

    total = planet_alignment + deck_fixing + money_value + spectral_power + risk
    return ConsumableScoreBreakdown(
        key=key,
        total=total,
        planet_alignment=planet_alignment,
        deck_fixing=deck_fixing,
        money_value=money_value,
        spectral_power=spectral_power,
        risk_penalty=risk,
        notes=tuple(notes),
    )


def _score_joker_candidate(
    info: dict[str, Any],
    joker: dict[str, Any],
    build: BuildEval,
) -> tuple[float, BuildValueEstimate | None, int | None]:
    candidate, replacement = _candidate_build_estimate(info, joker)
    if candidate is None:
        return 0.0, None, None

    value = _candidate_score_value(build.build_value, candidate)
    value += float(joker.get("dollars", 0) or 0) * 0.08
    if joker.get("rental"):
        value -= 0.15
    if joker.get("perishable"):
        value *= 0.9
    if joker.get("eternal") and value < 0.25:
        value -= 0.05
    return value, candidate, replacement


def score_shop_item(info: dict[str, Any], card: dict[str, Any], build: BuildEval | None = None) -> float:
    """Net strategic value of a single shop card for the current build."""
    if build is None:
        build = evaluate_build(info)
    cset = card.get("set", "")
    if cset == "Joker":
        joker = card.get("joker")
        if not isinstance(joker, dict):
            return 0.0
        value, _, _ = _score_joker_candidate(info, joker, build)
        return value
    if cset in ("Tarot", "Planet", "Spectral"):
        summary = {**card, "set": cset, "hand_type": card.get("hand_type") or card.get("type", "")}
        return score_consumable_item(info, summary, build).total
    return 0.0


def evaluate_shop_opportunity(info: dict[str, Any], build: BuildEval | None = None) -> ShopOpportunity:
    """Identify the best visible buy and whether rerolling/leaving is justified."""
    if build is None:
        build = evaluate_build(info)
    cards = info.get("shop_cards") or ()
    dollars = int(info.get("dollars", 0) or 0)
    reroll_cost = int(info.get("reroll_cost", 0) or 0)
    cap = build.interest_cap_cash

    best_val = 0.0
    best_key: str | None = None
    best_kind: str | None = None
    best_cost = 0
    best_net = 0.0
    best_replacement_index: int | None = None
    best_replacement_key: str | None = None
    best_score_delta = 0.0
    has_affordable = False
    has_critical = False

    owned = tuple(info.get("joker_details") or ())
    for card in cards:
        cost = int(card.get("cost", 0) or 0)
        candidate_estimate: BuildValueEstimate | None = None
        replacement_index: int | None = None
        if card.get("set") == "Joker" and isinstance(card.get("joker"), dict):
            value, candidate_estimate, replacement_index = _score_joker_candidate(info, card["joker"], build)
        else:
            value = score_shop_item(info, card, build)
        sale_proceeds = 0
        replacement_key: str | None = None
        if replacement_index is not None:
            replaced = owned[replacement_index]
            sale_proceeds = int(replaced.get("sell_cost", 0) or 0)
            replacement_key = str(replaced.get("key") or "")
        affordable = cost <= dollars + sale_proceeds
        dollars_after_buy = dollars + sale_proceeds - cost
        # Net value charges the lost-interest opportunity cost after any replacement sale.
        lost = interest_tiers(dollars, cap) - interest_tiers(dollars_after_buy, cap)
        emergency = build.survival_margin < 1.0 or not build.has_scoring_joker
        opp_cost = lost * (0.25 if emergency else 1.0) * 0.05
        net = value - opp_cost
        if net > best_net:
            best_val = value
            best_key = card.get("key")
            best_kind = card.get("set")
            best_cost = cost
            best_net = net
            best_replacement_index = replacement_index
            best_replacement_key = replacement_key
            best_score_delta = (
                candidate_estimate.representative_score_per_hand - build.estimated_score
                if candidate_estimate is not None
                else 0.0
            )
        if affordable and net > 0.15:
            has_affordable = True
        # Critical: no scoring engine yet and an affordable scoring upgrade is visible.
        if (
            affordable
            and not build.has_scoring_joker
            and candidate_estimate is not None
            and candidate_estimate.representative_score_per_hand > build.estimated_score * 1.05
            and value > 0.1
        ):
            has_critical = True

    dollars_after_reroll = dollars - reroll_cost
    lost_tiers = interest_tiers(dollars, cap) - interest_tiers(dollars_after_reroll, cap)
    can_reroll_above_cap = dollars_after_reroll >= cap
    # Rerolling is desirable when nothing visible is worth buying yet we still
    # need build power and can afford the roll.
    reroll_desirable = (
        reroll_cost <= dollars
        and not has_affordable
        and (build.joker_slots_left > 0 or not build.has_xmult_joker)
    )

    return ShopOpportunity(
        best_visible_upgrade=best_val,
        best_visible_key=best_key,
        best_visible_kind=best_kind,
        best_visible_cost=best_cost,
        best_visible_net_value=best_net,
        best_replacement_index=best_replacement_index,
        best_replacement_key=best_replacement_key,
        best_score_delta=best_score_delta,
        has_affordable_upgrade=has_affordable,
        has_critical_upgrade=has_critical,
        reroll_desirable=reroll_desirable,
        reroll_cost=reroll_cost,
        dollars_after_reroll=dollars_after_reroll,
        lost_interest_tiers_if_reroll=max(0, lost_tiers),
        can_reroll_above_interest_cap=can_reroll_above_cap,
    )
