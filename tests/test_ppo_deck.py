"""PPO deck selection must reach training, greedy eval, and sampled eval."""

import json
import runpy
import sys
from pathlib import Path

import pytest
import torch

from pylatro import load_game_data
from pylatro_agent.agent import AgentConfig, BalatroAgent
from pylatro_agent.training import evaluation_envs, ppo
from pylatro_agent.training.ppo_config import PPOConfig
from pylatro_agent.training.ppo_evaluation import evaluate_model, validate_resume_evaluation_seeds
from pylatro_agent.training.ppo_rollout import _make_vectorized_envs
from pylatro_agent.vocab import build_vocab


@pytest.mark.parametrize("deck,hands,discards", [("b_red", 4, 4), ("b_blue", 5, 3)])
def test_deck_reaches_real_rollouts_and_both_eval_policies(monkeypatch, tmp_path, deck, hands, discards):
    data = load_game_data()
    vocab = build_vocab(data)
    env = _make_vectorized_envs(1, data, vocab, use_async=False, deck_key=deck)
    try:
        env.reset()
        assert env.envs[0].state.round_resets.hands == hands
        assert env.envs[0].state.round_resets.discards == discards
    finally:
        env.close()

    original = evaluation_envs.BalatroEnv
    observed = []

    class CheckingEnv(original):
        def reset(self, **kwargs):
            result = super().reset(**kwargs)
            observed.append((self.state.round_resets.hands, self.state.round_resets.discards))
            return result

    monkeypatch.setattr(evaluation_envs, "BalatroEnv", CheckingEnv)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(0)
        model = BalatroAgent(AgentConfig(d_model=16, n_layers=1, n_heads=2, d_ff=32), vocab)
        results = tmp_path / "outcomes.jsonl"
        evaluate_model(model, data, vocab, 1, torch.device("cpu"), max_no_progress_steps=2,
                       sampled_games=1, deck_key=deck, results_path=results)
    finally:
        torch.set_num_threads(threads)
    assert observed == [(hands, discards), (hands, discards)]
    rows = [json.loads(line) for line in results.read_text().splitlines()]
    assert {row["policy"] for row in rows} == {"fresh_greedy", "fresh_sampled"}
    assert {row["deck_key"] for row in rows} == {deck}


def test_deck_cli_and_resume(monkeypatch):
    captured = []
    monkeypatch.setattr(ppo, "train_ppo", lambda config, **kwargs: captured.append(config))
    monkeypatch.setattr(sys, "argv", ["train.py", "ppo", "--device", "cpu", "--deck", "b_blue"])
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "train.py"), run_name="__main__")
    config = captured[0]
    assert config.deck_key == "b_blue"
    validate_resume_evaluation_seeds(config, {"deck_key": "b_blue", "eval_games": 50})
    with pytest.raises(ValueError, match="Deck changed"):
        validate_resume_evaluation_seeds(config, {"eval_games": 50})
    validate_resume_evaluation_seeds(PPOConfig(), {"eval_games": 50})
