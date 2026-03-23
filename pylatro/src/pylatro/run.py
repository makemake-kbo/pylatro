from __future__ import annotations

from ._helpers import _apply_voucher_to_run, _as_dict
from .data import GameData, load_game_data
from .models import PlayingCard, RunState
from .pool import get_new_boss, get_next_tag_key, get_next_voucher_key


def _apply_deck(state: RunState) -> None:
    deck = state.data.centers[state.deck_key]
    config = _as_dict(deck.get("config"))

    if state.stake >= 2:
        state.modifiers.setdefault("no_blind_reward", {})["Small"] = True
    if state.stake >= 3:
        state.modifiers["scaling"] = 2
    if state.stake >= 4:
        state.modifiers["enable_eternals_in_shop"] = True
    if state.stake >= 5:
        state.starting_params.discards -= 1
    if state.stake >= 6:
        state.modifiers["scaling"] = 3
    if state.stake >= 7:
        state.modifiers["enable_perishables_in_shop"] = True
    if state.stake >= 8:
        state.modifiers["enable_rentals_in_shop"] = True

    if voucher := config.get("voucher"):
        state.used_vouchers[voucher] = True
        _apply_voucher_to_run(state, voucher)
    for voucher_key in config.get("vouchers", []):
        state.used_vouchers[voucher_key] = True
        _apply_voucher_to_run(state, voucher_key)
    for consumable_key in config.get("consumables", []):
        state.consumable_keys.append(consumable_key)

    if hands := config.get("hands"):
        state.starting_params.hands += hands
    if dollars := config.get("dollars"):
        state.starting_params.dollars += dollars
    if config.get("remove_faces"):
        state.starting_params.no_faces = True
    if spectral_rate := config.get("spectral_rate"):
        state.spectral_rate = spectral_rate
    if discards := config.get("discards"):
        state.starting_params.discards += discards
    if config.get("randomize_rank_suit"):
        state.starting_params.erratic_suits_and_ranks = True
    if joker_slots := config.get("joker_slot"):
        state.starting_params.joker_slots += joker_slots
    if hand_size := config.get("hand_size"):
        state.starting_params.hand_size += hand_size
    if ante_scaling := config.get("ante_scaling"):
        state.starting_params.ante_scaling = ante_scaling
    if consumable_slot := config.get("consumable_slot"):
        state.starting_params.consumable_slots += consumable_slot
    if config.get("no_interest"):
        state.modifiers["no_interest"] = True
    if extra_hand_bonus := config.get("extra_hand_bonus"):
        state.modifiers["money_per_hand"] = extra_hand_bonus
    if extra_discard_bonus := config.get("extra_discard_bonus"):
        state.modifiers["money_per_discard"] = extra_discard_bonus


def _iter_starting_card_controls(state: RunState) -> list[dict[str, str | None]]:
    base_card_keys = list(state.data.cards.keys())
    controls: list[dict[str, str | None]] = []
    iterations = len(base_card_keys)

    for index in range(iterations):
        card_key = base_card_keys[index]
        if state.starting_params.erratic_suits_and_ranks:
            _, card_key = state.pseudorandom.pseudorandom_element(
                state.data.cards,
                state.pseudorandom.pseudoseed("erratic"),
            )
            assert isinstance(card_key, str)
        suit = card_key[0]
        rank = card_key[2]
        if state.starting_params.no_faces and rank in {"J", "Q", "K"}:
            continue
        controls.append({"s": suit, "r": rank, "e": None, "d": None, "g": None})

    controls.sort(key=lambda card: f"{card['s']}{card['r']}{card['e'] or ''}{card['d'] or ''}{card['g'] or ''}")
    return controls


def _build_starting_deck(state: RunState) -> None:
    controls = _iter_starting_card_controls(state)
    state.deck_cards = [
        PlayingCard(
            front_key=f"{card['s']}_{card['r']}",
            suit=state.data.cards[f"{card['s']}_{card['r']}"]["suit"],
            rank=card["r"] or "",
            center_key=card["e"] or "c_base",
            edition_key=card["d"],
            seal=card["g"],
        )
        for card in controls
    ]

    if state.data.centers[state.deck_key]["name"] == "Checkered Deck":
        for card in state.deck_cards:
            if card.suit == "Clubs":
                card.suit = "Spades"
                card.front_key = f"S_{card.rank}"
            elif card.suit == "Diamonds":
                card.suit = "Hearts"
                card.front_key = f"H_{card.rank}"

    shuffled = state.pseudorandom.pseudoshuffle(
        [
            {
                "front_key": card.front_key,
                "suit": card.suit,
                "rank": card.rank,
                "center_key": card.center_key,
                "edition_key": card.edition_key,
                "seal": card.seal,
            }
            for card in state.deck_cards
        ],
        state.pseudorandom.pseudoseed("shuffle"),
    )
    state.deck_cards = [
        PlayingCard(
            front_key=card["front_key"] or "",
            suit=card["suit"] or "",
            rank=card["rank"] or "",
            center_key=card["center_key"] or "c_base",
            edition_key=card["edition_key"],
            seal=card["seal"],
        )
        for card in shuffled
    ]


def create_run_state(seed: str, stake: int = 1, deck_key: str = "b_red", data: GameData | None = None) -> RunState:
    game_data = data or load_game_data()
    state = RunState(data=game_data, seed=seed, stake=stake, deck_key=deck_key)
    _apply_deck(state)
    state.blind_on_deck = "Small"

    state.round_resets.hands = state.starting_params.hands
    state.round_resets.discards = state.starting_params.discards
    state.round_resets.reroll_cost = state.starting_params.reroll_cost
    state.dollars = state.starting_params.dollars
    state.base_reroll_cost = state.starting_params.reroll_cost
    state.current_round.reroll_cost = state.base_reroll_cost

    state.round_resets.blind_choices["Boss"] = get_new_boss(state)
    state.current_voucher = get_next_voucher_key(state)
    state.round_resets.blind_tags["Small"] = get_next_tag_key(state)
    state.round_resets.blind_tags["Big"] = get_next_tag_key(state)

    _build_starting_deck(state)
    state.current_round.discards_left = state.round_resets.discards
    state.current_round.hands_left = state.round_resets.hands
    return state
