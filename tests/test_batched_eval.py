"""Batched evaluation must reproduce serial per-seed outcomes exactly."""

from __future__ import annotations

import json
import math

import pytest
import torch

from pylatro import load_game_data
from pylatro_agent.agent import AgentConfig, BalatroAgent
from pylatro_agent.env import BalatroEnv
from pylatro_agent.training.ppo import run_seed_evaluation
from pylatro_agent.training.ppo_evaluation import _score_critic_forecasts, evaluate_model, evaluation_rank
from pylatro_agent.training.ppo_observations import _single_obs_to_batch
from pylatro_agent.training.ppo_policy import _grammar_distribution
from pylatro_agent.vocab import build_vocab

# A small model keeps the test fast; batching correctness is independent of size.
_AGENT_CONFIG = AgentConfig(d_model=32, n_layers=1, n_heads=2, d_ff=64)


def test_resume_keeps_the_reserved_evaluation_panel():
    from pylatro_agent.training.ppo_config import PPOConfig
    from pylatro_agent.training.ppo_evaluation import validate_resume_evaluation_seeds

    config = PPOConfig(eval_games=2)
    validate_resume_evaluation_seeds(config, {"eval_games": 2, "eval_seeds": [10000, 10001]})
    with pytest.raises(ValueError, match="seeds changed"):
        validate_resume_evaluation_seeds(config, {"eval_games": 2, "eval_seeds": [1, 2]})
    with pytest.raises(ValueError, match="reservation metadata"):
        validate_resume_evaluation_seeds(config, {})


def _serial_outcomes(model, data, vocab, seeds, device, win_ante):
    """The retired serial greedy loop, kept here as the reference implementation."""
    outcomes = []
    model.eval()
    with torch.inference_mode():
        for seed in seeds:
            env = BalatroEnv(
                seed=seed,
                data=data,
                vocab=vocab,
                max_steps=64,
                win_ante=win_ante,
                enable_teacher=False,
            )
            obs, _ = env.reset()
            done = False
            info: dict = {}
            while not done:
                batch = _single_obs_to_batch(obs, device)
                dist, _ = _grammar_distribution(model, batch, temperature=1.0)
                action = dist.mode().item()
                del batch, dist
                obs, _reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            outcomes.append(
                {
                    "seed": seed,
                    "won": bool(info.get("won", False)),
                    "max_ante": int(info.get("ante", 1) or 1),
                }
            )
    return outcomes


def test_batched_greedy_eval_matches_serial_per_seed() -> None:
    data = load_game_data()
    vocab = build_vocab(data)
    torch.manual_seed(0)
    device = torch.device("cpu")
    model = BalatroAgent(_AGENT_CONFIG, vocab).to(device)
    seeds = [4101, 4102, 4103, 4104, 4105]

    serial = _serial_outcomes(model, data, vocab, seeds, device, win_ante=2)
    batched = run_seed_evaluation(
        model,
        data,
        vocab,
        seeds,
        device,
        max_no_progress_steps=64,
        win_ante=2,
        greedy=True,
        batch_size=4,
    )

    assert [row["seed"] for row in batched] == seeds
    for expected, actual in zip(serial, batched, strict=True):
        assert expected["seed"] == actual["seed"]
        assert expected["won"] == actual["won"]
        assert expected["max_ante"] == actual["max_ante"]


def test_batched_eval_is_invariant_to_batch_size() -> None:
    data = load_game_data()
    vocab = build_vocab(data)
    torch.manual_seed(0)
    device = torch.device("cpu")
    model = BalatroAgent(_AGENT_CONFIG, vocab).to(device)
    seeds = [7201, 7202, 7203, 7204, 7205, 7206]

    def outcomes(batch_size: int):
        return [
            (row["seed"], row["won"], row["max_ante"])
            for row in run_seed_evaluation(
                model,
                data,
                vocab,
                seeds,
                device,
                max_no_progress_steps=64,
                win_ante=2,
                greedy=True,
                batch_size=batch_size,
            )
        ]

    # Slot reuse (batch_size < len(seeds)) must not leak state between games.
    assert outcomes(1) == outcomes(2) == outcomes(6)


def test_eval_persists_separate_policies_without_training_or_rng_changes(tmp_path):
    data = load_game_data()
    vocab = build_vocab(data)
    torch.manual_seed(73)
    model = BalatroAgent(_AGENT_CONFIG, vocab)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    path = tmp_path / "eval.jsonl"
    summaries = {}
    class Writer:
        def add_scalar(self, name, value, step):
            summaries[name] = float(value)
    rate = evaluate_model(model, data, vocab, 2, torch.device("cpu"), win_ante=1,
                          max_no_progress_steps=8, eval_batch_size=2,
                          results_path=path, writer=Writer(), sampled_games=2)
    assert 0 <= rate <= 1
    torch.testing.assert_close(torch.get_rng_state(), rng)
    assert model.training
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], atol=0, rtol=0)
    assert all(parameter.grad is None for parameter in model.parameters())
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 4
    assert {row["policy"] for row in rows} == {"fresh_greedy", "fresh_sampled"}
    assert all("action_counts" in row and "cash" in row and "critic_forecasts" in row for row in rows)
    assert "eval/reach_ante1" in summaries and "eval/sampled/reach_ante1" in summaries
    assert "eval/sampled/critic/outcome_nll" in summaries


def test_critic_holdout_censors_stalls_and_scores_complete_labels():
    forecasts = [{"outcome_probabilities": [0.8, *([0.0] * 7), 0.2]}]
    assert _score_critic_forecasts(forecasts, won=False, final_ante=1, censored=True) == {}
    metrics = _score_critic_forecasts(forecasts, won=False, final_ante=1, censored=False)
    assert math.isclose(metrics["outcome_nll"], -math.log(0.8))
    assert math.isclose(metrics["win_brier"], 0.04)


def test_fresh_progress_breaks_zero_win_ties_but_never_overrides_win_rate():
    better_progress = {"clear_ante1": 0.8, "reach_ante6": 0.1}
    assert evaluation_rank(0.0, better_progress) > evaluation_rank(0.0, {})
    assert evaluation_rank(0.01, {}) > evaluation_rank(0.0, better_progress)
