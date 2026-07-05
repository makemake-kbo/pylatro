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

Scoring is intentionally *context-aware*: a joker's value depends on the current
deck, money, slots, ante, main hand, and existing engine, not a global "good
joker" table. Joker categories reuse the classification constants from
:mod:`pylatro_agent.heuristic` so the heuristic and the reward agree on what
counts as a scoring / xmult / scaling / economy / retrigger joker.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pylatro.runtime import consumable_limit, joker_limit

from .heuristic import (
    _ECONOMY_JOKERS,
    _ECONOMY_SCORES,
    _RETRIGGER_JOKER_KEYS,
    _SCALING_JOKER_KEYS,
    _SCALING_JOKER_SCORES,
    _SCALING_XMULT_JOKER_KEYS,
)

if TYPE_CHECKING:
    from pylatro.models import JokerInstance, RunState

_MID_GAME_ANTE = 3
_COMMITTED_HAND_TYPES = frozenset({"Pair", "High Card", "Two Pair", "Three of a Kind"})


# ───────────────────────────── info capture ─────────────────────────────


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


def _joker_summary(center: dict, key: str, live: JokerInstance | None) -> dict[str, Any]:
    cfg = center.get("config")
    if not isinstance(cfg, dict):
        cfg = {}
    extra = cfg.get("extra") if isinstance(cfg.get("extra"), dict) else {}
    base_x = cfg.get("Xmult")
    base_x = float(base_x) if isinstance(base_x, (int, float)) else 1.0
    extra_x = extra.get("Xmult") if isinstance(extra, dict) else None
    if isinstance(extra_x, (int, float)) and float(extra_x) > base_x:
        base_x = float(extra_x)

    summary: dict[str, Any] = {
        "key": key,
        "name": center.get("name", ""),
        "rarity": int(center.get("rarity", 0) or 0),
        "type": cfg.get("type") or "",
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
    }
    if live is not None:
        summary.update(
            {
                "sell_cost": int(live.sell_cost),
                "edition": live.edition or {},
                "eternal": bool(live.eternal),
                "perishable": bool(live.perishable),
                "rental": bool(live.rental),
                "debuffed": bool(live.debuff),
                "mult": float(live.mult),
                "t_mult": float(live.t_mult),
                "t_chips": float(live.t_chips),
                "x_mult": float(live.x_mult),
            }
        )
    else:
        summary.update(
            {
                "sell_cost": 0,
                "edition": {},
                "eternal": False,
                "perishable": False,
                "rental": False,
                "debuffed": False,
                "mult": summary["base_mult"],
                "t_mult": summary["base_t_mult"],
                "t_chips": summary["base_t_chips"],
                "x_mult": summary["base_x_mult"],
            }
        )
    return summary


def _consumable_summary(center: dict, key: str, live) -> dict[str, Any]:
    cfg = center.get("config") if isinstance(center.get("config"), dict) else {}
    return {
        "key": key,
        "name": center.get("name", ""),
        "set": center.get("set", ""),
        "sell_cost": int(getattr(live, "sell_cost", 0) or 0),
        "type": (cfg.get("type") or "") if isinstance(cfg, dict) else "",
    }


def _deck_stats(state: RunState) -> dict[str, Any]:
    rank_counts: dict[str, int] = {}
    suit_counts: dict[str, int] = {}
    enhancement_counts: dict[str, int] = {}
    seal_counts: dict[str, int] = {}
    edition_counts: dict[str, int] = {}
    centers = state.data.centers
    for card in state.deck_cards:
        rank_counts[card.rank] = rank_counts.get(card.rank, 0) + 1
        suit_counts[card.suit] = suit_counts.get(card.suit, 0) + 1
        if card.center_key and card.center_key != "c_base":
            effect = centers.get(card.center_key, {}).get("effect", card.center_key)
            enhancement_counts[effect] = enhancement_counts.get(effect, 0) + 1
        if card.seal:
            seal_counts[card.seal] = seal_counts.get(card.seal, 0) + 1
        if card.edition_key:
            edition_counts[card.edition_key] = edition_counts.get(card.edition_key, 0) + 1
    return {
        "size": len(state.deck_cards),
        "rank_counts": rank_counts,
        "suit_counts": suit_counts,
        "enhancement_counts": enhancement_counts,
        "seal_counts": seal_counts,
        "edition_counts": edition_counts,
        "stone_count": enhancement_counts.get("Stone Card", 0),
        "steel_count": enhancement_counts.get("Steel Card", 0),
        "glass_count": enhancement_counts.get("Glass Card", 0),
        "gold_count": enhancement_counts.get("Gold Card", 0),
    }


def _shop_card_summary(state: RunState, item, index: int) -> dict[str, Any]:
    center = state.data.centers.get(item.center_key, {})
    card_set = center.get("set", "")
    detail: dict[str, Any] = {
        "index": index,
        "key": item.center_key,
        "name": center.get("name", ""),
        "set": card_set,
        "cost": int(getattr(item, "cost", 0) or 0),
        "rarity": int(center.get("rarity", 0) or 0),
        "edition": getattr(item, "edition", None) or {},
        "eternal": bool(getattr(item, "eternal", False)),
        "perishable": bool(getattr(item, "perishable", False)),
        "rental": bool(getattr(item, "rental", False)),
    }
    if card_set == "Joker":
        detail["joker"] = _joker_summary(center, item.center_key, None)
    return detail


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


@dataclass(frozen=True)
class ShopOpportunity:
    best_visible_upgrade: float = 0.0
    best_visible_key: str | None = None
    best_visible_kind: str | None = None
    best_visible_cost: int = 0
    best_visible_net_value: float = 0.0

    has_affordable_upgrade: bool = False
    has_critical_upgrade: bool = False

    reroll_desirable: bool = False
    reroll_cost: int = 0
    dollars_after_reroll: int = 0
    lost_interest_tiers_if_reroll: int = 0
    can_reroll_above_interest_cap: bool = False


@dataclass
class _BuildContext:
    """Cheap, mutable view of the current build used to score individual items."""

    ante: int
    main_hand_type: str | None
    main_hand_confidence: float
    total_additive: float
    owned_keys: tuple[str, ...]
    joker_slots_left: int
    has_scoring_joker: bool
    has_xmult_joker: bool
    deck_stats: dict[str, Any] = field(default_factory=dict)


# Weights (pre dense-scale). Tuned to keep a single strong joker ≈ O(1).
_W_MULT = 0.020
_W_T_MULT_MATCH = 0.030
_W_T_MULT_GENERIC = 0.012
_W_T_CHIPS = 0.004
_W_XMULT = 0.55
_W_SCALING = 0.030
_W_ECONOMY = 0.040
_W_RETRIGGER = 0.40
_W_HAND_SYNERGY = 0.30
_W_FIRST_SCORING = 0.60


def _main_hand(info: dict[str, Any]) -> tuple[str | None, float]:
    counts = info.get("hand_play_counts") or {}
    total = sum(counts.values())
    if total <= 0:
        return None, 0.0
    best_type = max(counts, key=lambda k: counts[k])
    return best_type, counts[best_type] / total


def _edition_bonus(edition: Any) -> float:
    if not isinstance(edition, dict):
        return 0.0
    if edition.get("negative"):
        return 0.5
    if edition.get("polychrome"):
        return 0.4
    if edition.get("holo"):
        return 0.25
    if edition.get("foil"):
        return 0.1
    return 0.0


def _score_joker_summary(summary: dict[str, Any], ctx: _BuildContext) -> JokerScoreBreakdown:
    """Context-aware value of a joker (owned or in-shop) given the build."""
    key = summary["key"]
    mid_game = ctx.ante >= _MID_GAME_ANTE
    notes: list[str] = []

    immediate = summary["mult"] * _W_MULT
    immediate += summary["t_chips"] * _W_T_CHIPS * (0.5 if mid_game else 1.0)

    htype = summary.get("type") or ""
    hand_match = bool(htype) and htype == ctx.main_hand_type
    t_mult = summary["t_mult"]
    if t_mult:
        if hand_match:
            immediate += t_mult * _W_T_MULT_MATCH
        elif htype in _COMMITTED_HAND_TYPES or not htype:
            immediate += t_mult * _W_T_MULT_GENERIC
        else:
            immediate += t_mult * _W_T_MULT_GENERIC * 0.4

    # Multiplicative mult: only valuable once there is an additive base to scale,
    # and worth more on the main hand / when it has no hand restriction.
    xmult_score = 0.0
    x_mult = summary["x_mult"]
    if x_mult > 1.0:
        base = (x_mult - 1.0) * _W_XMULT
        if hand_match or not htype:
            base *= 1.5
        elif htype and not hand_match:
            base *= 0.5
        if ctx.total_additive >= 10:
            base *= 1.6
        elif ctx.total_additive >= 5:
            base *= 1.3
        if mid_game:
            base *= 1.2
        xmult_score = base

    scaling_score = 0.0
    if summary["is_scaling"]:
        base = _SCALING_JOKER_SCORES.get(key, 10.0) * _W_SCALING
        ante_left_bonus = max(8 - ctx.ante, 1) * 0.04
        scaling_score = base * (1.0 + ante_left_bonus)
        if ctx.joker_slots_left > 0:
            scaling_score *= 1.2

    economy_score = 0.0
    if summary["is_economy"]:
        eco = _ECONOMY_SCORES.get(key, 5.0) * _W_ECONOMY
        if ctx.ante <= 3:
            eco *= 1.4
        economy_score = eco
    economy_score += summary["dollars"] * _W_ECONOMY

    retrigger_score = 0.0
    if summary["is_retrigger"]:
        deck = ctx.deck_stats or {}
        enh = sum(deck.get("enhancement_counts", {}).values()) if deck else 0
        seals = sum(deck.get("seal_counts", {}).values()) if deck else 0
        synergy = 1.0 + min(enh + seals, 12) * 0.08
        if "j_photograph" in ctx.owned_keys:
            synergy += 0.5
        retrigger_score = _W_RETRIGGER * synergy

    hand_synergy = 0.0
    if hand_match and (t_mult or x_mult > 1.0):
        hand_synergy = _W_HAND_SYNERGY * ctx.main_hand_confidence

    edition_score = _edition_bonus(summary.get("edition"))

    duplicate_penalty = 0.0
    if key in ctx.owned_keys and not summary["is_economy"]:
        duplicate_penalty = -0.3

    slot_pressure_penalty = 0.0
    total = (
        immediate
        + xmult_score
        + scaling_score
        + economy_score
        + retrigger_score
        + hand_synergy
        + edition_score
        + duplicate_penalty
        + slot_pressure_penalty
    )

    # First real scoring joker is a survival priority: boost a reasonable buy.
    is_scoring_item = (
        summary["mult"] > 0
        or summary["t_mult"] > 0
        or summary["t_chips"] > 0
        or x_mult > 1.0
        or summary["is_scaling"]
    )
    if not ctx.has_scoring_joker and is_scoring_item and total > 0.05:
        total += _W_FIRST_SCORING
        notes.append("first_scoring_joker")

    return JokerScoreBreakdown(
        key=key,
        total=total,
        immediate_score=immediate,
        xmult_score=xmult_score,
        scaling_score=scaling_score,
        economy_score=economy_score,
        retrigger_score=retrigger_score,
        hand_type_synergy_score=hand_synergy,
        edition_score=edition_score,
        duplicate_penalty=duplicate_penalty,
        notes=tuple(notes),
    )


def _is_scoring_summary(summary: dict[str, Any]) -> bool:
    return (
        summary["mult"] > 0
        or summary["t_mult"] > 0
        or summary["t_chips"] > 0
        or summary["x_mult"] > 1.0
        or summary["is_scaling"]
    )


def _build_context(info: dict[str, Any], build: BuildEval) -> _BuildContext:
    jokers = info.get("joker_details") or ()
    total_additive = sum(j["mult"] + j["t_mult"] for j in jokers)
    return _BuildContext(
        ante=int(info.get("ante", 1) or 1),
        main_hand_type=build.main_hand_type,
        main_hand_confidence=build.main_hand_confidence,
        total_additive=total_additive,
        owned_keys=tuple(j["key"] for j in jokers),
        joker_slots_left=build.joker_slots_left,
        has_scoring_joker=build.has_scoring_joker,
        has_xmult_joker=build.has_xmult_joker,
        deck_stats=info.get("deck_stats") or {},
    )


def evaluate_build(info: dict[str, Any]) -> BuildEval:
    """Score the current owned build from a captured ``info`` dict."""
    jokers = info.get("joker_details") or ()
    cap = int(info.get("interest_cap_cash", 25) or 25)
    dollars = int(info.get("dollars", 0) or 0)
    main_type, main_conf = _main_hand(info)

    has_scoring = any(_is_scoring_summary(j) for j in jokers)
    has_xmult = any(j["x_mult"] > 1.0 or j["is_scaling_xmult"] for j in jokers)
    has_scaling = any(j["is_scaling"] for j in jokers)
    has_economy = any(j["is_economy"] for j in jokers)
    has_retrigger = any(j["is_retrigger"] for j in jokers)

    ctx = _BuildContext(
        ante=int(info.get("ante", 1) or 1),
        main_hand_type=main_type,
        main_hand_confidence=main_conf,
        total_additive=sum(j["mult"] + j["t_mult"] for j in jokers),
        owned_keys=tuple(j["key"] for j in jokers),
        joker_slots_left=int(info.get("joker_slots_left", 0) or 0),
        has_scoring_joker=has_scoring,
        has_xmult_joker=has_xmult,
        deck_stats=info.get("deck_stats") or {},
    )

    score_power = xmult_power = scaling_power = economy_power = retrigger_power = hand_align = 0.0
    for j in jokers:
        if j["debuffed"]:
            continue
        bd = _score_joker_summary(j, ctx)
        score_power += bd.immediate_score
        xmult_power += bd.xmult_score
        scaling_power += bd.scaling_score
        economy_power += bd.economy_score
        retrigger_power += bd.retrigger_score
        hand_align += bd.hand_type_synergy_score

    # Survival margin: how much engine relative to the ante. <1 means likely dying.
    engine = score_power + xmult_power + scaling_power
    survival_margin = engine / max(1.0, ctx.ante * 0.6) if ctx.ante else engine
    if not has_scoring:
        survival_margin = min(survival_margin, 0.5)

    total = score_power + xmult_power + scaling_power + economy_power + retrigger_power + hand_align

    return BuildEval(
        total=total,
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
        # Planet hand type lives in config "type"; align to the main hand.
        htype = summary.get("type") or ""
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


def score_shop_item(info: dict[str, Any], card: dict[str, Any], build: BuildEval | None = None) -> float:
    """Net strategic value of a single shop card for the current build."""
    if build is None:
        build = evaluate_build(info)
    cset = card.get("set", "")
    if cset == "Joker":
        joker = card.get("joker")
        if not joker:
            return 0.0
        ctx = _build_context(info, build)
        bd = _score_joker_summary(joker, ctx)
        if build.joker_slots_left <= 0:
            # Replacing requires a sell; discount the raw value.
            return bd.total * 0.5
        return bd.total
    if cset in ("Tarot", "Planet", "Spectral"):
        return score_consumable_item(info, {**card, "set": cset, "type": card.get("type", "")}, build).total
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
    has_affordable = False
    has_critical = False

    for card in cards:
        cost = int(card.get("cost", 0) or 0)
        affordable = cost <= dollars
        value = score_shop_item(info, card, build)
        # Net value charges the lost-interest opportunity cost of the spend.
        lost = interest_tiers(dollars, cap) - interest_tiers(dollars - cost, cap)
        emergency = build.survival_margin < 1.0 or not build.has_scoring_joker
        opp_cost = lost * (0.25 if emergency else 1.0) * 0.05
        net = value - opp_cost
        if value > best_val:
            best_val = value
            best_key = card.get("key")
            best_kind = card.get("set")
            best_cost = cost
            best_net = net
        if affordable and net > 0.15:
            has_affordable = True
        # Critical: no scoring engine yet and an affordable scoring joker is sitting here.
        if (
            affordable
            and card.get("set") == "Joker"
            and not build.has_scoring_joker
            and card.get("joker")
            and _is_scoring_summary(card["joker"])
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
        has_affordable_upgrade=has_affordable,
        has_critical_upgrade=has_critical,
        reroll_desirable=reroll_desirable,
        reroll_cost=reroll_cost,
        dollars_after_reroll=dollars_after_reroll,
        lost_interest_tiers_if_reroll=max(0, lost_tiers),
        can_reroll_above_interest_cap=can_reroll_above_cap,
    )
