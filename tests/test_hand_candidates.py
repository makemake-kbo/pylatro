from pylatro import create_run_state, start_blind
from pylatro_agent.hand_candidates import generate_hand_candidates


def test_generate_hand_candidates_handles_forced_selection() -> None:
    state = create_run_state("AAAAAAAA")
    state.round_resets.blind_choices["Boss"] = "bl_final_bell"
    state.blind_on_deck = "Boss"
    start_blind(state, "Boss")

    forced_slots = {idx for idx, card in enumerate(state.hand_cards) if card.forced_selection}
    play_candidates, discard_candidates = generate_hand_candidates(state)

    assert forced_slots
    assert play_candidates
    assert all(forced_slots.issubset(candidate.indices) for candidate in play_candidates)
    assert all(forced_slots.issubset(candidate.indices) for candidate in discard_candidates)


def test_play_candidates_expose_joker_free_raw_chip_score() -> None:
    state = create_run_state("AAAAAAAA")
    start_blind(state, "Small")
    cards = state.hand_cards
    specs = (
        ("Hearts", "A"),
        ("Hearts", "K"),
        ("Hearts", "Q"),
        ("Hearts", "J"),
        ("Hearts", "9"),
        ("Spades", "A"),
        ("Clubs", "5"),
        ("Diamonds", "2"),
    )
    for card, (suit, rank) in zip(cards, specs, strict=True):
        card.suit = suit
        card.rank = rank

    play_candidates, _discard_candidates = generate_hand_candidates(state)
    flush = next(candidate for candidate in play_candidates if candidate.hand_name == "Flush")
    pair = next(candidate for candidate in play_candidates if candidate.hand_name == "Pair")

    assert flush.raw_score == 340.0
    assert pair.raw_score == 64.0
    assert flush.raw_score > pair.raw_score
