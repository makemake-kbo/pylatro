"""Strict v8 checkpoint and resume coverage."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from pylatro_agent import checkpoint as ckpt
from pylatro_agent.constants import (
    TOKENIZER_SEMANTICS,
    TOKENIZER_VERSION,
)
from pylatro_agent.reward import RewardConfig
from pylatro_agent.training.ppo import (
    PPOConfig,
    _apply_lr_override,
    _load_v8_checkpoint_strict,
    _make_policy_optimizer,
    _optimizer_to,
    _restore_policy_optimizer_state,
)


def _tiny_model() -> torch.nn.Module:
    torch.manual_seed(0)
    return torch.nn.Linear(4, 2)


def _save_full(path, *, reward_config: RewardConfig | None = None, win_ante: int = 8):
    model = _tiny_model()
    optimizer = _make_policy_optimizer(model.parameters(), lr=3e-4)
    model(torch.randn(2, 4)).sum().backward()
    optimizer.step()
    ckpt.save_ppo_checkpoint(
        model,
        path,
        optimizer=optimizer,
        update_count=42,
        total_steps=4096,
        planned_updates=100,
        entropy_coeff=0.01,
        entropy_signal_ema=0.13,
        lr=3e-4,
        reward_config=reward_config or RewardConfig(),
        ppo_config_fields={"win_ante": win_ante, "rollout_temperature": 1.0},
    )
    return model, optimizer


def test_raw_state_dict_checkpoint_is_rejected(tmp_path) -> None:
    path = tmp_path / "raw.pt"
    torch.save(_tiny_model().state_dict(), path)
    with pytest.raises(RuntimeError, match="Raw state dicts are unsupported"):
        ckpt.load_checkpoint_payload(path, "cpu")


@pytest.mark.parametrize("old_version", [1, 5, 7])
def test_every_pre_v8_checkpoint_requires_fresh_training(tmp_path, old_version: int) -> None:
    path = tmp_path / f"v{old_version}.pt"
    torch.save(
        {"tokenizer_version": old_version, "state_dict": _tiny_model().state_dict()},
        path,
    )

    with pytest.raises(RuntimeError, match="fresh supervised training"):
        ckpt.load_checkpoint_payload(path, "cpu")
    with pytest.raises(RuntimeError, match="fresh supervised training"):
        _load_v8_checkpoint_strict(_tiny_model(), str(path), torch.device("cpu"))


def test_v8_weights_checkpoint_round_trips_strictly(tmp_path) -> None:
    source = _tiny_model()
    path = tmp_path / "weights.pt"
    ckpt.save_checkpoint(source, path)
    target = _tiny_model()
    with torch.no_grad():
        target.weight.zero_()
        target.bias.zero_()

    _load_v8_checkpoint_strict(target, str(path), torch.device("cpu"))

    for actual, expected in zip(target.parameters(), source.parameters(), strict=True):
        torch.testing.assert_close(actual, expected)


def test_v8_checkpoint_without_semantics_requires_fresh_training(tmp_path) -> None:
    path = tmp_path / "missing-semantics.pt"
    torch.save(
        {"tokenizer_version": TOKENIZER_VERSION, "state_dict": _tiny_model().state_dict()},
        path,
    )

    with pytest.raises(RuntimeError, match="fresh supervised training"):
        ckpt.load_checkpoint_payload(path, "cpu")


def test_v8_architecture_mismatch_is_not_partially_loaded(tmp_path) -> None:
    path = tmp_path / "bad-shape.pt"
    ckpt.save_checkpoint(torch.nn.Linear(4, 3), path)
    with pytest.raises(RuntimeError, match="architecture-incompatible"):
        _load_v8_checkpoint_strict(_tiny_model(), str(path), torch.device("cpu"))


def test_full_v8_checkpoint_round_trips_resume_state(tmp_path) -> None:
    path = tmp_path / "full.pt"
    model, optimizer = _save_full(path)

    payload = ckpt.load_ppo_resume_payload(
        path,
        "cpu",
        active_reward_config=RewardConfig(),
        active_win_ante=8,
    )
    assert ckpt.is_ppo_full_checkpoint(path, "cpu")
    assert payload["tokenizer_version"] == TOKENIZER_VERSION
    assert payload["tokenizer_semantics"] == TOKENIZER_SEMANTICS
    assert payload["checkpoint_format"] == ckpt.PPO_CHECKPOINT_FORMAT
    assert payload["update_count"] == 42
    assert payload["total_steps"] == 4096
    assert payload["planned_updates"] == 100
    assert "optimizer_state_dict" in payload
    assert set(payload["rng_states"]) >= {"python", "numpy", "torch_cpu"}
    assert "return_rms" not in payload

    restored_model = _tiny_model()
    restored_model.load_state_dict(payload["state_dict"], strict=True)
    restored_optimizer = _make_policy_optimizer(restored_model.parameters(), lr=1e-4)
    restored_lr = _restore_policy_optimizer_state(
        restored_optimizer,
        payload,
        PPOConfig(lr=3e-4),
        torch.device("cpu"),
    )
    assert restored_lr == pytest.approx(3e-4)
    assert restored_optimizer.state_dict()["state"].keys() == optimizer.state_dict()["state"].keys()
    for actual, expected in zip(restored_model.parameters(), model.parameters(), strict=True):
        torch.testing.assert_close(actual, expected)


def test_full_resume_rejects_reward_and_target_mismatch(tmp_path) -> None:
    path = tmp_path / "full.pt"
    _save_full(path, win_ante=5)

    with pytest.raises(RuntimeError, match="win_ante mismatch"):
        ckpt.load_ppo_resume_payload(
            path,
            "cpu",
            active_reward_config=RewardConfig(),
            active_win_ante=8,
        )
    changed_reward = RewardConfig(dense_reward_scale=0.5)
    with pytest.raises(RuntimeError, match="reward fingerprint mismatch"):
        ckpt.load_ppo_resume_payload(
            path,
            "cpu",
            active_reward_config=changed_reward,
            active_win_ante=5,
        )


def test_weights_only_checkpoint_is_rejected_for_resume(tmp_path) -> None:
    path = tmp_path / "weights.pt"
    ckpt.save_checkpoint(_tiny_model(), path)
    with pytest.raises(RuntimeError, match="weights-only checkpoint"):
        ckpt.load_ppo_resume_payload(
            path,
            "cpu",
            active_reward_config=RewardConfig(),
        )


def test_rng_state_capture_and_restore_is_exact() -> None:
    random.seed(10)
    np.random.seed(10)
    torch.manual_seed(10)
    state = ckpt.capture_rng_states()
    expected = (random.random(), np.random.random(), torch.rand(()).item())
    ckpt.restore_rng_states(state)
    actual = (random.random(), np.random.random(), torch.rand(()).item())
    assert actual == pytest.approx(expected)


def test_optimizer_to_moves_state_tensors_to_cpu() -> None:
    model = _tiny_model()
    optimizer = _make_policy_optimizer(model.parameters(), lr=1e-3)
    model(torch.randn(2, 4)).sum().backward()
    optimizer.step()
    _optimizer_to(optimizer, torch.device("cpu"))
    assert all(
        not torch.is_tensor(value) or value.device.type == "cpu"
        for state in optimizer.state.values()
        for value in state.values()
    )


def test_apply_lr_override_updates_every_group() -> None:
    optimizer = _make_policy_optimizer(_tiny_model().parameters(), lr=1e-3)
    _apply_lr_override(optimizer, new_lr=5e-5, checkpoint_lr=1e-3)
    assert [group["lr"] for group in optimizer.param_groups] == [pytest.approx(5e-5)]
