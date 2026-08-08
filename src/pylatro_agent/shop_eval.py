"""Capture and evaluate build state for reward shaping and diagnostics.

The reward shaping in :mod:`pylatro_agent.reward` operates on serialized ``info``
dicts captured at each environment step (it never sees the live ``RunState``).
This module provides:

``capture_build_features`` keeps the Gym environment and fast trajectory
generator on the same serialized schema. ``evaluate_build`` exposes the
score-model projection consumed by reward shaping and diagnostics.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pylatro import get_blind_amount
from pylatro.runtime import consumable_limit, joker_limit

from .build_value import BuildValueEstimate, estimate_build_value
from .heuristic import (
    _ECONOMY_JOKERS,
    _RETRIGGER_JOKER_KEYS,
    _SCALING_JOKER_KEYS,
    _SCALING_XMULT_JOKER_KEYS,
)

if TYPE_CHECKING:
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
        "effect": center.get("effect", ""),
        "sell_cost": int(getattr(live, "sell_cost", 0) or 0),
        "hand_type": (cfg.get("hand_type") or cfg.get("type") or "") if isinstance(cfg, dict) else "",
    }


def _pack_card_summary(state: RunState, item, index: int) -> dict[str, Any]:
    """Serialize the actual contents of an open pack, including seals."""
    center = state.data.centers.get(item.center_key, {})
    front = state.data.cards.get(item.front_key, {}) if item.front_key else {}
    cfg = _config_dict(center.get("config"))
    return {
        "index": index,
        "key": item.center_key,
        "name": center.get("name", ""),
        "set": center.get("set", ""),
        "effect": center.get("effect", ""),
        "rank": str(item.front_key[2] if item.front_key else front.get("value") or ""),
        "suit": str(front.get("suit") or ""),
        "enhancement": str(center.get("effect", "") or "") if item.center_key != "c_base" else "",
        "seal": str(getattr(item, "seal", "") or ""),
        "edition": deepcopy(getattr(item, "edition", None)) if getattr(item, "edition", None) else {},
        "hand_type": cfg.get("hand_type") or cfg.get("type") or "",
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
                "identity": card.reward_uid,
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
                "times_played": int(card.times_played),
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
    centers = state.data.centers

    shop_items = list(state.shop.cards) + list(state.shop.vouchers) + list(state.shop.boosters)
    pack_items = state.pack.cards if state.pack is not None else ()
    round_active = any(value == "Current" for value in state.round_resets.blind_states.values())
    hands_available = state.current_round.hands_left if round_active else state.round_resets.hands
    discards_available = state.current_round.discards_left if round_active else state.round_resets.discards
    round_discard_capacity = (
        state.current_round.discards_left + state.current_round.discards_used
        if round_active
        else state.round_resets.discards
    )
    hand_size = state.current_round.hand_size if round_active else state.starting_params.hand_size
    boss_key = state.round_resets.blind_choices.get("Boss", "")
    boss = state.data.blinds.get(boss_key, {}) if boss_key else {}
    boss_debuff = boss.get("debuff", {}) or {}

    return {
        "joker_details": tuple(_joker_summary(centers.get(j.center_key, {}), j.center_key, j) for j in state.jokers),
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
        "hands_available": max(1, hands_available),
        "discards_available": max(0, discards_available),
        "round_discard_capacity": max(0, round_discard_capacity),
        "round_active": round_active,
        "blind_disabled": bool(state.blind_disabled),
        "hand_size": int(hand_size),
        "joker_limit": joker_limit(state),
        "consumable_limit": consumable_limit(state),
        "tarot_usage_total": int(state.consumeable_usage_total.get("tarot", 0) or 0),
        "planet_usage_total": int(state.consumeable_usage_total.get("planet", 0) or 0),
        "boss_debuff_suit": (
            "" if state.blind_disabled else str(boss_debuff.get("suit", "") or "")
        ),
        "blind_target": _upcoming_blind_target(state),
        "shop_cards": tuple(_shop_card_summary(state, item, i) for i, item in enumerate(shop_items)),
        "pack_card_details": tuple(_pack_card_summary(state, item, i) for i, item in enumerate(pack_items)),
    }


# ───────────── evaluation ─────────────


@dataclass(frozen=True)
class BuildEval:
    """Build-value projection used by reward and diagnostics."""

    estimated_score: float
    no_joker_baseline_score: float
    required_score_per_hand: float
    readiness_ratio: float
    build_value: BuildValueEstimate


def evaluate_build(info: dict[str, Any]) -> BuildEval:
    """Evaluate the current build from a captured step-info dictionary."""
    estimate = estimate_build_value(info, tuple(info.get("joker_details") or ()))
    return BuildEval(
        estimated_score=estimate.representative_score_per_hand,
        no_joker_baseline_score=estimate.no_joker_baseline_score,
        required_score_per_hand=estimate.required_score_per_hand,
        readiness_ratio=estimate.readiness_ratio,
        build_value=estimate,
    )
