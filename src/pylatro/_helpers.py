from __future__ import annotations

from math import floor
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .models import RunState


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _edition_cost(edition: dict[str, bool] | None) -> int:
    if not edition:
        return 0
    return (
        (3 if edition.get("holo") else 0)
        + (2 if edition.get("foil") else 0)
        + (5 if edition.get("polychrome") else 0)
        + (5 if edition.get("negative") else 0)
    )


def _has_showman(state: RunState) -> bool:
    return state.has_joker("Showman")


def _mark_center_used(state: RunState, center_key: str) -> None:
    state.used_jokers[center_key] = True


def _calculate_cost(
    state: RunState,
    center: dict[str, Any],
    *,
    edition: dict[str, bool] | None = None,
    rental: bool = False,
) -> int:
    base_cost = int(center.get("cost", 1) or 1)
    cost = max(
        1,
        floor((base_cost + state.inflation + _edition_cost(edition) + 0.5) * (100 - state.discount_percent) / 100),
    )
    if center.get("set") == "Booster" and state.modifiers.get("booster_ante_scaling"):
        cost += state.round_resets.ante - 1
    if (
        center.get("set") == "Planet" or (center.get("set") == "Booster" and "Celestial" in center["name"])
    ) and state.has_joker("Astronomer"):
        cost = 0
    if rental:
        cost = 1
    return cost


def _apply_voucher_to_run(state: RunState, center_key: str) -> None:
    center = state.data.centers[center_key]
    extra: Any = _as_dict(center.get("config")).get("extra")
    name = center["name"]

    if name in {"Overstock", "Overstock Plus"}:
        state.shop.joker_max += 1
    elif name in {"Tarot Merchant", "Tarot Tycoon"}:
        state.tarot_rate = 4 * extra
    elif name in {"Planet Merchant", "Planet Tycoon"}:
        state.planet_rate = 4 * extra
    elif name in {"Hone", "Glow Up"}:
        state.edition_rate = extra
    elif name in {"Magic Trick", "Illusion"}:
        state.playing_card_rate = extra
    elif name == "Crystal Ball":
        state.starting_params.consumable_slots += 1
    elif name in {"Clearance Sale", "Liquidation"}:
        state.discount_percent = extra
    elif name in {"Reroll Surplus", "Reroll Glut"}:
        state.starting_params.reroll_cost -= extra
    elif name in {"Seed Money", "Money Tree"}:
        state.interest_cap = extra
    elif name in {"Grabber", "Nacho Tong"}:
        state.starting_params.hands += extra
    elif name in {"Wasteful", "Recyclomancy"}:
        state.starting_params.discards += extra
    elif name == "Antimatter":
        state.starting_params.joker_slots += 1
    elif name in {"Paint Brush", "Palette"}:
        state.starting_params.hand_size += 1
    elif name in {"Hieroglyph", "Petroglyph"}:
        state.round_resets.ante -= extra
        state.round_resets.blind_ante -= extra
        if name == "Hieroglyph":
            state.starting_params.hands -= extra
        else:
            state.starting_params.discards -= extra
