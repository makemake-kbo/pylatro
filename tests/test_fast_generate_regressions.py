from __future__ import annotations

from pylatro import create_run_state, load_game_data, select_blind, start_blind
from pylatro_agent.heuristic import HeuristicAgent
from pylatro_agent.tokenizer import Tokenizer
from pylatro_agent.training import fast_generate
from pylatro_agent.vocab import build_vocab


def test_run_game_single_pass_marks_progress() -> None:
    data = load_game_data()
    vocab = build_vocab(data)
    progress_flags: list[bool] = []

    def capture_reward(state, prev_info, curr_info, terminated, won) -> float:
        del state, prev_info, terminated, won
        progress_flags.append(bool(curr_info["progress_made"]))
        return 0.0

    original_reward = fast_generate.default_reward
    fast_generate.default_reward = capture_reward
    try:
        records, _, _ = fast_generate._run_game_single_pass(
            0,
            data,
            Tokenizer(vocab=vocab),
            HeuristicAgent(),
            0.995,
        )
    finally:
        fast_generate.default_reward = original_reward

    assert records
    assert any(progress_flags)


def test_cached_best_hand_updates_after_in_place_mutation() -> None:
    data = load_game_data()
    state = create_run_state("test_seed", 1, "b_red", data=data)
    select_blind(state, "Small")
    start_blind(state, "Small")

    agent = HeuristicAgent()
    hand = state.hand_cards

    initial = agent._cached_best_hand(state, hand)
    candidates = [idx for idx in range(len(hand)) if idx not in initial]
    assert len(candidates) >= 4
    for idx in candidates[:4]:
        hand[idx].rank = "A"

    cached = agent._cached_best_hand(state, hand)
    fresh = agent._find_best_hand(state, hand)

    assert cached == fresh
    assert cached != initial
