"""Batched evaluation must reproduce serial per-seed outcomes exactly."""

from __future__ import annotations

import torch

from pylatro import load_game_data
from pylatro_agent.agent import AgentConfig, BalatroAgent
from pylatro_agent.env import BalatroEnv
from pylatro_agent.training.ppo import (
    _grammar_distribution,
    _single_obs_to_batch,
    run_seed_evaluation,
)
from pylatro_agent.vocab import build_vocab

# A small model keeps the test fast; batching correctness is independent of size.
_AGENT_CONFIG = AgentConfig(d_model=32, n_layers=1, n_heads=2, d_ff=64)


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
