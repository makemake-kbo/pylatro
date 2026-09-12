"""Regressions for pretraining CLI settings, update rates, and seeded runs."""

from __future__ import annotations

import random
import runpy
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from pylatro import load_game_data
from pylatro_agent.agent import AgentConfig
from pylatro_agent.constants import TOKENIZER_VERSION
from pylatro_agent.tokenizer import Tokenizer
from pylatro_agent.training import supervised
from pylatro_agent.training.fast_generate import _build_obs
from pylatro_agent.training.fast_runner import FastRunner
from pylatro_agent.vocab import build_vocab


@pytest.mark.parametrize("phase", ["supervised", "pretrain_from_checkpoint", "ppo"])
@pytest.mark.parametrize("override", [False, True])
def test_cli_preserves_phase_defaults_and_honors_overrides(monkeypatch, tmp_path, phase, override):
    from pylatro_agent.training import model_generate, ppo

    captured = []

    def capture(config, **kwargs):
        captured.append(config)

    monkeypatch.setattr(supervised, "train_supervised", capture)
    monkeypatch.setattr(ppo, "train_ppo", capture)
    monkeypatch.setattr(model_generate, "load_records", lambda *args, **kwargs: [])
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    argv = ["train.py", phase, "--device", "cpu", "--seed", "123", "--data-path", str(dataset)]
    if phase == "pretrain_from_checkpoint":
        argv += ["--inference-checkpoint", "unused.pt"]
    if override:
        argv += ["--lr", "0.0007", "--value-loss-coeff", "0.7"]
    monkeypatch.setattr(sys, "argv", argv)
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "train.py"), run_name="__main__")

    assert len(captured) == 1
    config = captured[0]
    assert config.lr == (0.0007 if override else (5e-5 if phase == "ppo" else 3e-4))
    assert config.value_loss_coeff == (0.7 if override else (0.25 if phase == "ppo" else 0.5))
    assert config.seed == 123


@pytest.fixture
def training_fixture():
    # Tiny real-model updates exercise dropout, shuffled labels, optimizer, and TB.
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    data = load_game_data()
    obs = _build_obs(FastRunner(0, data), Tokenizer(vocab=build_vocab(data)))
    action = int(np.flatnonzero(obs["action_mask"])[0])
    records = [
        dict(obs=obs, action=action, seed=100 + i, won=True, max_ante=8,
             return_target=float(i), terminal_outcome_target=8, terminal_outcome_mask=1.0,
             tokenizer_version=TOKENIZER_VERSION)
        for i in range(4)
    ]
    yield data, records, AgentConfig(d_model=16, n_layers=1, n_heads=2, d_ff=32, dropout=0.1)
    torch.set_num_threads(threads)


@pytest.mark.parametrize("warmup,expected", [
    (0, [1.0, 0.8535533905932737, 0.5, 0.14644660940672627]),
    (2, [0.5, 1.0, 1.0, 0.5]),
    (1000, [0.001, 0.002, 0.003, 0.004]),
])
def test_actual_update_rates_match_warmup_decay_and_tensorboard(
    training_fixture, tmp_path, monkeypatch, warmup, expected,
):
    data, records, agent_config = training_fixture
    update_rates = []
    optimizers = []

    class RecordingAdamW(torch.optim.AdamW):
        def step(self, closure=None):
            update_rates.append(self.param_groups[0]["lr"])
            return super().step(closure)

    def optimizer(*args, **kwargs):
        result = RecordingAdamW(*args, **kwargs)
        optimizers.append(result)
        return result

    monkeypatch.setattr(supervised, "AdamW", optimizer)
    config = supervised.SupervisedConfig(
        batch_size=2, max_epochs=2, warmup_steps=warmup, log_interval=1,
        save_dir=str(tmp_path / "checkpoints"), log_dir=str(tmp_path / "events"),
    )
    supervised.train_supervised(config, agent_config, data=data, records=records)
    assert update_rates == pytest.approx([config.lr * factor for factor in expected])
    if warmup < len(update_rates):
        assert optimizers[0].param_groups[0]["lr"] == pytest.approx(0.0)
    events = EventAccumulator(config.log_dir).Reload()
    assert [event.value for event in events.Scalars("train/lr")] == pytest.approx(update_rates)
    assert all(np.isfinite(event.value) for event in events.Scalars("train/loss"))


def test_same_seed_reproduces_trained_weights_and_rngs(training_fixture, tmp_path):
    data, records, agent_config = training_fixture
    results = []
    for run, seed in enumerate((42, 42, 43)):
        # Deliberately perturb the incoming global RNG states between runs.
        random.seed(900 + run)
        np.random.seed(900 + run)
        torch.manual_seed(900 + run)
        config = supervised.SupervisedConfig(
            seed=seed, batch_size=2, max_epochs=1,
            save_dir=str(tmp_path / str(run) / "checkpoints"),
            log_dir=str(tmp_path / str(run) / "events"),
        )
        model = supervised.train_supervised(config, replace(agent_config), data=data, records=records)
        results.append((model.state_dict(), random.random(), np.random.random(), torch.rand(1)))
        checkpoint = torch.load(Path(config.save_dir) / "supervised_epoch1.pt", weights_only=True)
        assert checkpoint["supervised_config"]["seed"] == seed
        assert checkpoint["supervised_config"]["lr"] == config.lr

    assert all(torch.equal(results[0][0][name], tensor) for name, tensor in results[1][0].items())
    assert results[0][1:3] == results[1][1:3]
    assert torch.equal(results[0][3], results[1][3])
    assert any(not torch.equal(results[0][0][name], tensor) for name, tensor in results[2][0].items())
