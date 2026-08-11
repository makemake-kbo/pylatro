from __future__ import annotations

import pickle

import numpy as np
import pytest
import torch

from pylatro import JokerInstance, load_game_data
from pylatro_agent.agent import AgentConfig, BalatroAgent
from pylatro_agent.constants import HISTORY_START, TOKENIZER_SEMANTICS, TOKENIZER_VERSION
from pylatro_agent.env import BalatroEnv
from pylatro_agent.history import (
    BURGLAR_FLAG,
    DNA_FLAG,
    DUSK_PAYOFF_FLAG,
    DUSK_SETUP_FLAG,
    CardSnapshot,
    PendingPlay,
    PlayHistoryTracker,
)
from pylatro_agent.reward import RewardConfig
from pylatro_agent.training.fast_runner import FastRunner
from pylatro_agent.training.model_generate import load_records, save_records
from pylatro_agent.vocab import build_vocab


def _pending(ordinal: int, score_before: int = 0, *, flags: int = 0) -> PendingPlay:
    return PendingPlay(
        cards=(CardSnapshot("A", "Spades", "c_base", None, None, 0, False, False),),
        joker_keys=("j_joker",),
        blind_target=300,
        score_before=score_before,
        play_ordinal=ordinal,
        hands_remaining=max(0, 20 - ordinal),
        mechanic_flags=flags,
    )


def test_tracker_snapshots_cards_joker_order_and_mechanics() -> None:
    env = BalatroEnv(seed=7, enable_teacher=False)
    env.reset()
    env.step(0)  # enter Small Blind
    state = env.state
    assert state is not None
    state.joker_keys[:] = ["j_dna", "j_dusk", "j_burglar"]
    state.jokers[:] = [JokerInstance(center_key=key) for key in state.joker_keys]
    card = state.hand_cards[0]
    state.current_round.hands_played = 0
    state.current_round.hands_left = 3
    state.current_round.discards_left = 0

    pending = env._history.capture(state, [card], blind_target=300, round_score=10)
    original_rank = pending.cards[0].rank
    card.rank = "2"
    state.joker_keys.reverse()
    event = env._history.finalize(pending, hand_type="High Card", score=25)

    assert event.cards[0].rank == original_rank
    assert event.joker_keys == ("j_dna", "j_dusk", "j_burglar")
    assert event.mechanic_flags & DNA_FLAG
    assert event.mechanic_flags & DUSK_SETUP_FLAG
    assert event.mechanic_flags & BURGLAR_FLAG
    assert event.cumulative_round_score == 35

    state.current_round.hands_left = 1
    payoff = env._history.capture(state, [state.hand_cards[1]], blind_target=300, round_score=35)
    assert payoff.mechanic_flags & DUSK_PAYOFF_FLAG


def test_three_round_rotation_reset_and_deterministic_overflow() -> None:
    tracker = PlayHistoryTracker()
    tracker.start_round("one")
    for ordinal in range(1, 15):
        tracker.finalize(
            _pending(ordinal, flags=DNA_FLAG if ordinal == 1 else 0),
            hand_type="Pair" if ordinal % 2 else "High Card",
            score=ordinal * 10,
        )
    current = tracker.rounds[-1]
    assert len(current.events) == 12
    assert current.events == sorted(current.events, key=lambda event: event.play_ordinal)
    assert current.events[0].play_ordinal == 1  # protected mechanic event
    assert current.events[-1].play_ordinal == 14  # latest event
    assert current.omitted.count == 2
    assert current.omitted.total_score == 50
    assert current.omitted.max_score == 30

    tracker.start_round("two")
    tracker.finalize(_pending(1), hand_type="Pair", score=1)
    tracker.start_round("three")
    tracker.finalize(_pending(1), hand_type="Pair", score=1)
    tracker.start_round("four")
    assert [round_.key for round_ in tracker.rounds] == ["two", "three", "four"]
    tracker.reset()
    assert not any(round_.has_content for round_ in tracker.rounds)


def test_env_and_fast_runner_emit_identical_history_for_seeded_trace() -> None:
    data = load_game_data()
    vocab = build_vocab(data)
    env = BalatroEnv(seed=11, data=data, vocab=vocab, enable_teacher=False)
    obs, _ = env.reset()
    runner = FastRunner(11, data, max_steps=100)

    history_keys = tuple(key for key in obs if key.startswith("history_"))
    for _ in range(40):
        fast_arrays = runner.history.encode(vocab).as_dict()
        for key in history_keys:
            np.testing.assert_array_equal(obs[key], fast_arrays[key])
        if runner.done:
            break
        env_mask = env.action_masks()
        fast_mask = runner.compute_mask()
        np.testing.assert_array_equal(env_mask, fast_mask)
        action = int(np.flatnonzero(env_mask)[0])
        obs, _reward, terminated, truncated, _info = env.step(action)
        runner.step(action)
        assert terminated or truncated or not runner.done


def test_skipped_blind_rotates_window_without_a_play_event() -> None:
    data = load_game_data()
    env = BalatroEnv(seed=17, data=data, enable_teacher=False)
    env.reset()
    runner = FastRunner(17, data)
    env.step(1)
    runner.step(1)
    assert env._history.rounds[-1].key == runner.history.rounds[-1].key
    assert env._history.rounds[-1].key == (1, "Small", "skipped")
    assert not env._history.rounds[-1].events


def test_history_padding_is_inert_and_real_history_reaches_shared_heads() -> None:
    data = load_game_data()
    vocab = build_vocab(data)
    model = BalatroAgent(
        AgentConfig(d_model=32, n_layers=1, n_heads=4, d_ff=64, dropout=0.0), vocab
    ).eval()
    env = BalatroEnv(seed=3, data=data, vocab=vocab, enable_teacher=False)
    obs, _ = env.reset()
    env.step(0)
    action = int(np.flatnonzero(env.action_masks())[0])
    obs, *_ = env.step(action)

    def batch(source: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
        result = {}
        for key, value in source.items():
            dtype = torch.float32 if value.dtype == np.float32 or key == "action_mask" else torch.long
            result[key] = torch.as_tensor(value, dtype=dtype).unsqueeze(0)
        return result

    real = batch(obs)
    with torch.no_grad():
        real_dist, real_values = model.action_distribution(**real)
        padded = {key: value.clone() for key, value in real.items()}
        padded["history_round_mask"].zero_()
        padded["history_event_mask"].zero_()
        padded["history_card_mask"].zero_()
        padded["history_joker_mask"].zero_()
        padded["attention_mask"][:, HISTORY_START : HISTORY_START + 3].zero_()
        masked_dist, masked_values = model.action_distribution(**padded)

        random_padding = {key: value.clone() for key, value in padded.items()}
        random_padding["history_events"].random_(0, 5)
        random_padding["history_event_features"].normal_()
        padding_dist, padding_values = model.action_distribution(**random_padding)

    assert not torch.allclose(real_dist.output.macro_logits, masked_dist.output.macro_logits)
    for key in ("expected_score", "win_prob", "ante_survival"):
        assert not torch.allclose(real_values[key], masked_values[key])
        torch.testing.assert_close(masked_values[key], padding_values[key])
    torch.testing.assert_close(masked_dist.output.macro_logits, padding_dist.output.macro_logits)


def test_cached_observation_datasets_are_tokenizer_versioned(tmp_path) -> None:
    path = tmp_path / "records.pkl"
    records = [{"obs": {}, "action": 0}]
    reward_config = RewardConfig(
        gamma=0.99,
        potential_win_ante=4,
        dense_reward_scale=0.25,
    )
    save_records(records, path, reward_config=reward_config)
    assert load_records(path, reward_config=reward_config) == records

    with pytest.raises(ValueError, match="reward fingerprint mismatch"):
        load_records(
            path,
            reward_config=RewardConfig(
                gamma=0.99,
                potential_win_ante=4,
                dense_reward_scale=1.0,
            ),
        )

    legacy_path = tmp_path / "legacy.pkl"
    with legacy_path.open("wb") as handle:
        pickle.dump(records, handle)
    try:
        load_records(legacy_path, reward_config=reward_config)
    except ValueError as exc:
        assert "regenerate" in str(exc)
    else:
        raise AssertionError("legacy observation dataset was accepted")

    metadata_less_path = tmp_path / "metadata_less.pkl"
    with metadata_less_path.open("wb") as handle:
        pickle.dump(
            {
                "tokenizer_version": TOKENIZER_VERSION,
                "tokenizer_semantics": TOKENIZER_SEMANTICS,
                "records": records,
            },
            handle,
        )
    with pytest.raises(ValueError, match="predates reward metadata"):
        load_records(metadata_less_path, reward_config=reward_config)
