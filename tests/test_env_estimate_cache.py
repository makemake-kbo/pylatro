"""Repeated captures reuse estimates without hiding engine or caller mutations."""

from unittest.mock import patch

import pylatro_agent.env as env_module
from pylatro_agent.env import BalatroEnv


def test_capture_estimate_cache_tracks_fresh_features():
    env = BalatroEnv(seed=1701, enable_teacher=False)
    try:
        env.reset()
        with patch.object(env_module, "estimate_clear_risk", wraps=env_module.estimate_clear_risk) as risk:
            first = env._capture_state_info()
            assert risk.call_count == 0
            assert env._capture_state_info() == first
            first["hand_details"]["High Card"]["chips"] = -100
            assert env._capture_state_info()["hand_details"]["High Card"]["chips"] != -100
            assert risk.call_count == 0
            env.state.hands["High Card"]["chips"] += 100
            changed = env._capture_state_info()
            assert risk.call_count == 1
            assert env._capture_state_info() == changed
            assert risk.call_count == 1
            del env._state_estimate_cache
            assert env._capture_state_info() == changed
            assert risk.call_count == 2
    finally:
        env.close()


def test_cached_evaluation_matches_uncached(monkeypatch):
    import torch

    from pylatro import load_game_data
    from pylatro_agent.agent import AgentConfig, BalatroAgent
    from pylatro_agent.training.ppo_evaluation import run_seed_evaluation
    from pylatro_agent.vocab import build_vocab

    data = load_game_data()
    vocab = build_vocab(data)
    torch.manual_seed(0)
    model = BalatroAgent(AgentConfig(d_model=32, n_layers=1, n_heads=2, d_ff=64), vocab)

    def evaluate():
        rows = run_seed_evaluation(
            model,
            data,
            vocab,
            [10000, 10001, 10002],
            torch.device("cpu"),
            max_no_progress_steps=16,
            win_ante=2,
            batch_size=3,
            record_critic=True,
        )
        return [{k: v for k, v in row.items() if not k.endswith("_seconds")} for row in rows]

    cached = evaluate()
    capture = BalatroEnv._capture_state_info

    def uncached(self):
        self._state_estimate_cache = None
        return capture(self)

    monkeypatch.setattr(BalatroEnv, "_capture_state_info", uncached)
    assert evaluate() == cached
