"""Approximate conditional passive income for experimental shop forecasts."""


def interest_voucher_can_repay(state, item):
    """Whether capped extra interest can exceed the price before the last shop.

    This uses the current interest multiplier and assumes enough cash to earn
    the maximum every round. It is an optimistic eligibility check, not a
    purchase recommendation or a prediction of future To the Moon purchases.
    """
    if state.modifiers.get("no_interest"):
        return False
    cap = state.data.centers[item.center_key]["config"]["extra"]
    extra = max(0, cap // 5 - state.interest_cap // 5) * state.interest_amount
    payouts = max(0, (8 - state.round_resets.ante) * 3 + {
        "Small": 2, "Big": 1, "Boss": 0,
    }.get(state.blind_on_deck, 0))
    return extra * payouts > item.cost


def expected_passive_income(state):
    """Estimate dollars per future round, without changing the live state.

    Delayed Gratification assumes no discards in half of future safe rounds.
    Parking assumes two plays, each leaving hand size minus three cards held.
    These are policy estimates, not guaranteed payouts or simulator rules.
    """
    keys = [joker.center_key for joker in state.jokers]
    income = 0.0
    for joker in state.jokers:
        if joker.center_key == "j_delayed_grat":
            if not any(key in keys for key in ("j_mystic_summit", "j_burglar")):
                income += 0.5 * state.round_resets.discards * joker.extra
        elif joker.center_key == "j_reserved_parking" and state.deck_cards:
            pareidolia = "j_pareidolia" in keys
            faces = sum(
                pareidolia or (card.center_key != "m_stone" and card.rank in {"J", "Q", "K"})
                for card in state.deck_cards
            )
            held = max(0, min(state.starting_params.hand_size, len(state.deck_cards)) - 3)
            chance = min(1, state.probabilities.get("normal", 1) / max(1, joker.extra["odds"]))
            retriggers = 1 + keys.count("j_mime")
            income += 2 * held * faces / len(state.deck_cards) * chance * joker.extra["dollars"] * retriggers
    return income
