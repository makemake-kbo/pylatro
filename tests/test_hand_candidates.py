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
