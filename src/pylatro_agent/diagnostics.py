"""Action-quality diagnostics shared by BalatroEnv and fast_generate.

These functions compute per-step diagnostics (hand_play_*, planet_*,
pack_skip_*) that default_reward_components reads as shaping inputs.
Kept here, rather than on BalatroEnv, so fast_generate can produce
the same diagnostic fields without depending on gymnasium.
"""

from __future__ import annotations

from typing import Any

from .action import ActionType
from .hand_candidates import generate_hand_candidates
from .subset_actions import subset_indices


def _center_key(card) -> str:
    return str(getattr(card, "center_key", "") or "")


def _center_set(state, center_key: str) -> str:
    center = state.data.centers.get(center_key, {})
    return str(center.get("set", "") or "")


def _planet_hand_type(state, center_key: str) -> str:
    center = state.data.centers.get(center_key, {})
    config = center.get("config", {}) or {}
    return str(config.get("hand_type", "") or "")


def _main_hand_proxy(state) -> str:
    # rank by (times played, hand level, chips*mult) descending; name breaks final ties
    rows: list[tuple[int, int, float, str]] = []
    for name, hand in state.hands.items():
        rows.append((
            int(hand.get("played", 0) or 0),
            int(hand.get("level", 1) or 1),
            float(hand.get("chips", 0) or 0) * float(hand.get("mult", 1) or 1),
            str(name),
        ))
    rows.sort(reverse=True)
    if not rows or rows[0][0] <= 0:
        return ""
    return rows[0][3]


def _planet_alignment_rank(
    state, center_key: str, main_hand: str, max_played: int
) -> tuple[int, float]:
    """How well a planet fits the run: (matches main hand, hand play share)."""
    hand_type = _planet_hand_type(state, center_key)
    played = (
        int(state.hands.get(hand_type, {}).get("played", 0) or 0) if hand_type else 0
    )
    share = (played / max_played) if max_played > 0 else 0.0
    return (1 if hand_type and hand_type == main_hand else 0, share)


def _is_best_planet_in_pack(state, chosen_index: int) -> bool:
    """True when no other planet in the open pack better fits the run.

    Claiming the best planet a matchless pack offers (even Pluto) is
    legitimate — the pack is already paid for and skipping wastes it — so
    the unmatched-claim penalty exempts these claims. Only planet cards
    compete in the ranking; ties keep the claim exempt.
    """
    main_hand = _main_hand_proxy(state)
    max_played = max(
        (int(hand.get("played", 0) or 0) for hand in state.hands.values()),
        default=0,
    )
    chosen_rank = _planet_alignment_rank(
        state, _center_key(state.pack.cards[chosen_index]), main_hand, max_played
    )
    for index, card in enumerate(state.pack.cards):
        if index == chosen_index:
            continue
        other_key = _center_key(card)
        if _center_set(state, other_key) != "Planet":
            continue
        if _planet_alignment_rank(state, other_key, main_hand, max_played) > chosen_rank:
            return False
    return True


def _planet_diagnostics(state, center_key: str, *, prefix: str) -> dict[str, Any]:
    hand_type = _planet_hand_type(state, center_key)
    main_hand = _main_hand_proxy(state)
    hand_info = state.hands.get(hand_type, {}) if hand_type else {}
    played = int(hand_info.get("played", 0) or 0) if hand_info else 0
    max_played = max(
        (int(hand.get("played", 0) or 0) for hand in state.hands.values()),
        default=0,
    )
    return {
        f"{prefix}_observed": True,
        f"{prefix}_key": center_key,
        f"{prefix}_hand_type": hand_type,
        f"{prefix}_played_hand": played > 0,
        # Play count of the planet's hand relative to the most-played hand this
        # run, 1.0 for the workhorse hand, near 0 for a hand played once.
        f"{prefix}_play_share": (played / max_played) if max_played > 0 else 0.0,
        f"{prefix}_main_hand": main_hand,
        f"{prefix}_main_hand_match": bool(hand_type and hand_type == main_hand),
    }


def action_diagnostics(state, decoded) -> dict[str, Any]:
    """Return policy-quality diagnostics for the pre-action state.

    Mirrors the env's _action_diagnostics, when this returns fields,
    default_reward_components reads them to award candidate / planet /
    pack-skip shaping bonuses. fast_generate calls this so its value-
    head BC targets see the same reward signal as PPO.
    """
    if state is None:
        return {}

    if decoded.action_type == ActionType.PLAY_SUBSET:
        diagnostics: dict[str, Any] = {"hand_play_observed": True}
        indices = tuple(subset_indices(decoded.index))
        if any(index >= len(state.hand_cards) for index in indices):
            diagnostics["hand_play_not_in_candidates"] = True
            return diagnostics
        play_candidates, _discard_candidates = generate_hand_candidates(state)
        if not play_candidates:
            diagnostics["hand_play_not_in_candidates"] = True
            return diagnostics

        chosen = next(
            (candidate for candidate in play_candidates if candidate.indices == indices),
            None,
        )
        if chosen is None:
            diagnostics["hand_play_not_in_candidates"] = True
            diagnostics["hand_play_best_hand"] = play_candidates[0].hand_name
            return diagnostics

        best = play_candidates[0]
        diagnostics.update({
            "hand_play_in_candidates": True,
            "hand_play_top1": chosen.indices == best.indices,
            "hand_play_top3": any(c.indices == chosen.indices for c in play_candidates[:3]),
            "hand_play_candidate_value_ratio": float(
                chosen.estimated_score / max(best.estimated_score, 1e-9)
            ),
            "hand_play_chosen_hand": chosen.hand_name,
            "hand_play_best_hand": best.hand_name,
        })
        return diagnostics

    if decoded.action_type == ActionType.USE_CONSUMABLE_NO_TARGET:
        if decoded.index >= len(state.consumables):
            return {}
        center_key = _center_key(state.consumables[decoded.index])
        if _center_set(state, center_key) != "Planet":
            return {}
        return _planet_diagnostics(state, center_key, prefix="planet_use")

    if decoded.action_type == ActionType.PACK_CLAIM:
        if state.pack is None or decoded.index >= len(state.pack.cards):
            return {}
        center_key = _center_key(state.pack.cards[decoded.index])
        if _center_set(state, center_key) != "Planet":
            return {}
        diagnostics = _planet_diagnostics(state, center_key, prefix="planet_claim")
        diagnostics["planet_claim_best_available"] = _is_best_planet_in_pack(
            state, decoded.index
        )
        return diagnostics

    if decoded.action_type == ActionType.PACK_SKIP and state.pack is not None:
        return {
            "pack_skip_state_name": state.pack.state_name,
            "planet_pack_skip": state.pack.state_name == "PLANET_PACK",
        }

    return {}
