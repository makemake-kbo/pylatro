from __future__ import annotations

from math import floor, fmod, isinf, isnan, log10
from typing import TYPE_CHECKING

from .pool import get_new_boss, get_next_tag_key, get_next_voucher_key
from .runtime import apply_end_of_round

if TYPE_CHECKING:
    from .models import RunState


def get_blind_amount(ante: int, scaling: int | None = None) -> int | float:
    """Base chip requirement for the Small Blind of a given ante.

    Reproduces Balatro's ante scaling. Antes 1-8 use a hand-tuned lookup table
    (one row per stake-difficulty ``scaling`` tier). Beyond ante 8 the value is
    extrapolated with the game's super-exponential growth formula and rounded to
    two significant figures, so endless-mode requirements eventually overflow
    into floats.
    """
    scaling = scaling or 1
    if scaling == 1:
        amounts = [300, 800, 2000, 5000, 11000, 20000, 35000, 50000]
    elif scaling == 2:
        amounts = [300, 900, 2600, 8000, 20000, 36000, 60000, 100000]
    elif scaling == 3:
        amounts = [300, 1000, 3200, 9000, 25000, 60000, 110000, 200000]
    else:
        raise ValueError(f"Unsupported blind scaling {scaling}")

    if ante < 1:
        return 100
    if ante <= 8:
        return amounts[ante - 1]

    # Endless mode (ante > 8): grow super-exponentially from the ante-8 value.
    # The growth factor 0.75 and base 1.6 are the game's tuning constants; the
    # exponent ramps with each ante past 8. The result can overflow to inf/nan,
    # which we surface rather than clamp.
    ante8_requirement = amounts[7]
    antes_past_8 = ante - 8
    growth_exponent = 1 + 0.2 * antes_past_8
    raw_amount = ante8_requirement * (1.6 + (0.75 * antes_past_8) ** growth_exponent) ** antes_past_8
    if isinf(raw_amount) or isnan(raw_amount):
        return float("nan")

    amount = float(floor(raw_amount))
    if isinf(amount) or isnan(amount):
        return float("nan")

    # Round down to 2 significant figures (the game's display rounding).
    magnitude = 10 ** floor(log10(amount) - 1)
    amount -= fmod(amount, magnitude)
    if isnan(amount):
        return float("nan")
    # 2^53 is the largest integer a float represents exactly; past it, keep the
    # float rather than lie about an exact int value.
    if amount.is_integer() and amount <= 9_007_199_254_740_992:
        return int(amount)
    return amount


def select_blind(state: RunState, blind_type: str | None = None) -> None:
    blind_type = blind_type or state.blind_on_deck or "Small"
    blind_key = state.round_resets.blind_choices[blind_type]
    state.round_resets.blind = state.data.blinds[blind_key]
    state.round_resets.blind_states[blind_type] = "Current"
    state.shop.cards = []
    state.shop.vouchers = []
    state.shop.boosters = []
    state.pack = None
    state.current_round.discards_left = max(0, state.round_resets.discards)
    state.current_round.hands_left = max(1, state.round_resets.hands)
    state.current_round.hands_played = 0
    state.current_round.discards_used = 0
    state.current_round.reroll_cost_increase = 0
    state.current_round.used_packs = []
    state.current_round.free_rerolls = sum(
        1 for key in state.joker_keys if state.data.centers[key]["name"] == "Chaos the Clown"
    )
    state.calculate_reroll_cost(skip_increment=True)
    state.current_round.dollars = 0


def skip_blind(state: RunState) -> str:
    skipped = state.blind_on_deck
    skip_to = "Big" if skipped == "Small" else "Boss"
    state.skips += 1
    if tag := state.round_resets.blind_tags.get(skipped):
        state.tags.append(tag)
    state.round_resets.blind_states[skipped] = "Skipped"
    state.round_resets.blind_states[skip_to] = "Select"
    state.blind_on_deck = skip_to
    return skip_to


def reroll_boss(state: RunState, from_tag: bool = False) -> str:
    state.round_resets.boss_rerolled = True
    if not from_tag:
        state.dollars -= 10  # boss reroll costs $10 (free when granted by a tag)
    state.round_resets.blind_choices["Boss"] = get_new_boss(state)
    return state.round_resets.blind_choices["Boss"]


def reset_blinds(state: RunState) -> None:
    if state.round_resets.blind_states["Boss"] == "Defeated":
        state.round_resets.blind_states = {"Small": "Upcoming", "Big": "Upcoming", "Boss": "Upcoming"}
        state.blind_on_deck = "Small"
        state.round_resets.blind_choices["Boss"] = get_new_boss(state)
        state.round_resets.boss_rerolled = False


def cash_out(state: RunState) -> None:
    apply_end_of_round(state)
    state.current_round.jokers_purchased = 0
    state.current_round.discards_left = max(0, state.round_resets.discards)
    state.current_round.hands_left = max(1, state.round_resets.hands)
    state.shop.cards = []
    state.shop.vouchers = []
    state.shop.boosters = []
    state.pack = None
    state.current_round.used_packs = []
    if state.round_resets.blind_states["Boss"] == "Defeated":
        most_played = max(
            ((hand["played"], -hand["order"], name) for name, hand in state.hands.items()),
            default=(0, 0, "High Card"),
        )[2]
        state.current_round.most_played_poker_hand = most_played
        if state.round_resets.ante == state.win_ante:
            state.won = True
        state.round_resets.ante += 1
        state.round_resets.blind_ante = state.round_resets.ante
        for hand in state.hands.values():
            hand["played_this_round"] = 0
        for card in state.deck_cards:
            card.played_this_ante = False
        state.current_voucher = get_next_voucher_key(state)
        state.round_resets.blind_tags["Small"] = get_next_tag_key(state)
        state.round_resets.blind_tags["Big"] = get_next_tag_key(state)
    reset_blinds(state)
