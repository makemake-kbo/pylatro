from __future__ import annotations

import math

import numpy as np
import pytest

from pylatro import add_consumable, add_joker, create_run_state, load_game_data, select_blind, start_blind
from pylatro.models import PlayingCard
from pylatro_agent.action import ActionType, encode_action
from pylatro_agent.constants import ActionRange, SubPhase
from pylatro_agent.heuristic import HeuristicAgent
from pylatro_agent.masks import compute_action_mask
from pylatro_agent.reward import ANTE1_CHIP_TEMPO_BONUS, default_reward_components
from pylatro_agent.subset_actions import consumable_subset_index, subset_index
from pylatro_agent.tokenizer import Tokenizer
from pylatro_agent.training import fast_generate
from pylatro_agent.training.fast_generate import _capture_info
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


def test_fast_runner_joker_masks_match_main_masks() -> None:
    data = load_game_data()
    runner = FastRunner(42, data)
    runner.step(int(ActionRange.BLIND_PLAY))
    for joker_key in ("j_blueprint", "j_dusk", "j_hack", "j_idol"):
        add_joker(runner.state, joker_key)

    main_play = compute_action_mask(runner.state, SubPhase.CHOOSE_ACTION)
    fast_play = runner.compute_mask().copy()
    move_slice = slice(int(ActionRange.MOVE_JOKER_START), int(ActionRange.MOVE_JOKER_END) + 1)
    np.testing.assert_array_equal(fast_play[move_slice], main_play[move_slice])

    runner._sub_phase = SubPhase.SHOP
    main_shop = compute_action_mask(runner.state, SubPhase.SHOP)
    fast_shop = runner.compute_mask().copy()
    np.testing.assert_array_equal(fast_shop[move_slice], main_shop[move_slice])
    assert not fast_shop[move_slice].any()


def test_fast_runner_blocks_immediate_joker_move_inverse() -> None:
    data = load_game_data()
    runner = FastRunner(42, data)
    runner.step(int(ActionRange.BLIND_PLAY))
    for joker_key in ("j_blueprint", "j_dusk", "j_hack", "j_idol"):
        add_joker(runner.state, joker_key)

    first_step = encode_action(ActionType.MOVE_JOKER, 1, 0)
    assert runner.compute_mask()[first_step] == 1
    runner.step(first_step)

    reverse = encode_action(ActionType.MOVE_JOKER, 0, 1)
    next_step = encode_action(ActionType.MOVE_JOKER, 2, 1)
    mask = runner.compute_mask()
    assert mask[reverse] == 0
    assert mask[next_step] == 1


def test_fast_runner_mr_bones_score_reset_claws_back_ante1_potential() -> None:
    data = load_game_data()
    runner = FastRunner(7, data)
    runner.step(int(ActionRange.BLIND_PLAY))
    add_joker(runner.state, "j_mr_bones")

    blind_target = runner._ctrl.blind_target()
    prior_score = math.ceil(blind_target * 0.25)
    runner._ctrl.round_score = prior_score
    runner._round_score = prior_score
    runner.state.current_round.hands_left = 1
    prev_info = _capture_info(runner)

    action = encode_action(ActionType.PLAY_SUBSET, subset_index((0,)))
    assert runner.compute_mask()[action] == 1
    runner.step(action)
    curr_info = _capture_info(runner)
    curr_info["action_type"] = "play_subset"

    assert runner._ctrl.round_score == 0
    assert runner.round_score == 0
    assert curr_info["round_score"] == 0
    assert "j_mr_bones" not in runner.state.joker_keys
    assert not runner.done

    components = default_reward_components(
        runner.state,
        prev_info,
        curr_info,
        terminated=False,
        won=False,
    )
    assert components["ante1_chip_tempo"] == pytest.approx(-ANTE1_CHIP_TEMPO_BONUS * prior_score / blind_target)


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


def test_high_card_candidate_search_checks_effect_kickers(monkeypatch) -> None:
    data = load_game_data()
    state = create_run_state("high_card_kicker", data=data)
    select_blind(state, "Small")
    start_blind(state, "Small")
    state.hands["High Card"]["level"] = 4
    agent = HeuristicAgent()

    def synthetic_score(_state, indices) -> int:
        indices = tuple(indices)
        if indices == (3, 7):
            return 10_000
        if len(indices) == 1:
            return 100 - indices[0]
        return 1

    monkeypatch.setattr(agent, "_estimate_hand_score", synthetic_score)

    assert agent._cached_best_hand(state, state.hand_cards) == {3, 7}


def test_forced_card_candidate_can_include_a_pair() -> None:
    data = load_game_data()
    state = create_run_state("forced_pair_candidate", data=data)
    state.hands["Pair"]["level"] = 6
    state.hand_cards = [
        PlayingCard(front_key="S_3", suit="Spades", rank="3", forced_selection=True),
        PlayingCard(front_key="H_6", suit="Hearts", rank="6"),
        PlayingCard(front_key="C_6", suit="Clubs", rank="6"),
        PlayingCard(front_key="D_9", suit="Diamonds", rank="9"),
        PlayingCard(front_key="H_Q", suit="Hearts", rank="Q"),
    ]
    agent = HeuristicAgent()

    chosen = agent._cached_best_hand(state, state.hand_cards)

    assert {0, 1, 2}.issubset(chosen)
    assert agent._quick_hand_quality(state, [state.hand_cards[i] for i in chosen]) == "Pair"


def test_cached_best_hand_updates_after_planet_level_change(monkeypatch) -> None:
    data = load_game_data()
    state = create_run_state("best_hand_planet_cache", data=data)
    select_blind(state, "Small")
    start_blind(state, "Small")
    agent = HeuristicAgent()
    calls = 0
    original = agent._find_best_hand

    def counted_find(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(agent, "_find_best_hand", counted_find)
    agent._cached_best_hand(state, state.hand_cards)
    state.hands["Pair"]["level"] += 1
    agent._cached_best_hand(state, state.hand_cards)

    assert calls == 2


def test_hand_score_cache_does_not_leak_between_runs(monkeypatch) -> None:
    data = load_game_data()
    first = create_run_state("score_cache_run_a", data=data)
    second = create_run_state("score_cache_run_b", data=data)
    first.hand_cards = second.hand_cards = [first.deck_cards[0]]
    agent = HeuristicAgent()

    def run_specific_score(state, _indices) -> int:
        return 111 if state.seed == "score_cache_run_a" else 222

    monkeypatch.setattr(agent, "_estimate_hand_score_compute", run_specific_score)

    assert agent._estimate_hand_score(first, (0,)) == 111
    assert agent._estimate_hand_score(second, (0,)) == 222


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


def test_fast_runner_and_env_report_same_one_shot_held_gold_event() -> None:
    from pylatro_agent.env import BalatroEnv

    data = load_game_data()
    runner = FastRunner(73, data)
    runner.step(int(ActionRange.BLIND_PLAY))
    runner.state.hands["High Card"]["chips"] = 1_000
    runner.state.hands["High Card"]["mult"] = 10
    runner.state.hand_cards[0].center_key = "m_gold"
    runner.state.hand_cards[1].center_key = "m_gold"
    fast_action = encode_action(ActionType.PLAY_SUBSET, subset_index((2,)))
    fast_result = runner.step(fast_action)

    vocab = build_vocab(data)
    env = BalatroEnv(seed=73, data=data, vocab=vocab, enable_teacher=False)
    env.reset()
    env.step(int(ActionRange.BLIND_PLAY))
    env.state.hands["High Card"]["chips"] = 1_000
    env.state.hands["High Card"]["mult"] = 10
    env.state.hand_cards[0].center_key = "m_gold"
    env.state.hand_cards[1].center_key = "m_gold"
    env_action = encode_action(ActionType.PLAY_SUBSET, subset_index((2,)))
    _, _, _, _, info = env.step(env_action)

    assert fast_result.held_gold_count == 2
    assert fast_result.held_gold_payout == 6
    assert info["strategic_held_gold_count"] == 2
    assert info["strategic_held_gold_payout"] == 6
    assert info["reward_strategic_held_gold_payout"] == pytest.approx(0.18)


def test_fast_runner_disabled_boss_cashout_exposes_next_suit_boss() -> None:
    from pylatro_agent.tokenizer import strategy_probability_features

    data = load_game_data()
    runner = FastRunner(91, data)
    state = runner.state
    state.blind_on_deck = "Boss"
    state.round_resets.blind_choices["Boss"] = "bl_goad"
    runner.step(int(ActionRange.BLIND_PLAY))
    for card in state.deck_cards:
        card.seal = None
    next(card for card in state.deck_cards if card.suit == "Spades").seal = "Blue"
    state.blind_disabled = True
    runner._ctrl.cash_out()
    state.round_resets.blind_choices["Boss"] = "bl_goad"
    runner._sub_phase = SubPhase.BLIND_SELECT
    runner.step(int(ActionRange.BLIND_SKIP))
    runner.step(int(ActionRange.BLIND_SKIP))

    features = strategy_probability_features(state, SubPhase.BLIND_SELECT)
    assert not state.blind_disabled
    assert state.blind_on_deck == "Boss"
    assert features[0] == 0.0
    assert features[5] == 0.0


def test_heuristic_known_early_seeds_reach_ante_three() -> None:
    from pylatro_agent.env import BalatroEnv

    data = load_game_data()
    vocab = build_vocab(data)
    for seed in (1, 3, 17, 26, 38):
        env = BalatroEnv(seed=seed, data=data, vocab=vocab, max_steps=100)
        obs, _ = env.reset()
        agent = HeuristicAgent()
        done = False
        info = {"ante": 1, "won": False}

        while not done and info["ante"] < 3:
            action = agent.select_action(
                env.state,
                env._sub_phase,
                obs["action_mask"],
                round_score=env._controller.round_score,
            )
            obs, _reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        assert info["won"] or info["ante"] >= 3, f"seed {seed} died before Ante 3"


def test_heuristic_repaired_ante_one_seeds_reach_ante_two() -> None:
    from pylatro_agent.env import BalatroEnv

    data = load_game_data()
    vocab = build_vocab(data)
    for seed in (50, 64, 69, 88, 115, 202, 223, 292):
        env = BalatroEnv(seed=seed, data=data, vocab=vocab, max_steps=100)
        obs, _ = env.reset()
        agent = HeuristicAgent()
        done = False
        info = {"ante": 1, "won": False}

        while not done and info["ante"] < 2:
            action = agent.select_action(
                env.state,
                env._sub_phase,
                obs["action_mask"],
                round_score=env._controller.round_score,
            )
            obs, _reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        assert info["won"] or info["ante"] >= 2, f"seed {seed} died in Ante 1"


def test_baron_seed_preserves_engine_past_ante_three() -> None:
    from pylatro_agent.env import BalatroEnv

    data = load_game_data()
    env = BalatroEnv(seed=18, data=data, vocab=build_vocab(data), max_steps=180)
    obs, _ = env.reset()
    agent = HeuristicAgent()
    done = False
    info = {"ante": 1, "won": False}

    while not done and info["ante"] < 4:
        action = agent.select_action(
            env.state,
            env._sub_phase,
            obs["action_mask"],
            round_score=env._controller.round_score,
        )
        obs, _reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    assert info["won"] or info["ante"] >= 4


def test_mouth_seed_does_not_lock_an_off_plan_straight() -> None:
    from pylatro_agent.env import BalatroEnv

    data = load_game_data()
    env = BalatroEnv(seed=10, data=data, vocab=build_vocab(data), max_steps=180)
    obs, _ = env.reset()
    agent = HeuristicAgent()
    done = False
    info = {"ante": 1, "won": False}

    while not done and info["ante"] < 4:
        action = agent.select_action(
            env.state,
            env._sub_phase,
            obs["action_mask"],
            round_score=env._controller.round_score,
        )
        obs, _reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    assert info["won"] or info["ante"] >= 4
