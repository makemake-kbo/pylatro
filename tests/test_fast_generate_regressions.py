from __future__ import annotations

from pylatro import add_consumable, create_run_state, load_game_data, select_blind, start_blind
from pylatro_agent.action import ActionType, encode_action
from pylatro_agent.constants import ActionRange
from pylatro_agent.heuristic import HeuristicAgent
from pylatro_agent.subset_actions import consumable_subset_index
from pylatro_agent.tokenizer import Tokenizer
from pylatro_agent.training import fast_generate
from pylatro_agent.training.fast_runner import FastRunner
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


def test_fast_action_diagnostics_match_env_diagnostics() -> None:
    """fast_generate and BalatroEnv must emit the same diagnostic fields so
    the value head sees the same shaped reward during BC and PPO."""
    import numpy as np

    from pylatro_agent.action import ActionType, decode_action
    from pylatro_agent.diagnostics import action_diagnostics
    from pylatro_agent.env import BalatroEnv

    data = load_game_data()
    vocab = build_vocab(data)
    env = BalatroEnv(seed=42, data=data, vocab=vocab, max_steps=200)
    env.reset()

    # Step through BLIND_SELECT into CHOOSE_ACTION so a PLAY_SUBSET is legal.
    for _ in range(20):
        mask = env.action_masks()
        valid = np.flatnonzero(mask)
        play_idx = next(
            (a for a in valid if decode_action(int(a)).action_type == ActionType.PLAY_SUBSET),
            None,
        )
        if play_idx is not None:
            break
        env.step(int(valid[0]))

    assert play_idx is not None, "env should reach CHOOSE_ACTION within 20 steps"

    decoded = decode_action(int(play_idx))
    env_diag = env._action_diagnostics(decoded)
    shared_diag = action_diagnostics(env._controller.state, decoded)
    assert env_diag == shared_diag
    # Sanity check: the diagnostic actually contains hand_play_* fields.
    assert "hand_play_observed" in env_diag


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


def test_fast_runner_unmasks_hand_targeted_consumable() -> None:
    data = load_game_data()
    runner = FastRunner(0, data)
    runner.step(ActionRange.BLIND_PLAY)
    add_consumable(runner.state, "c_lovers")

    mask = runner.compute_mask()
    action = encode_action(
        ActionType.USE_CONSUMABLE_HAND_SUBSET,
        0,
        consumable_subset_index((0,)),
    )

    assert mask[action] == 1
