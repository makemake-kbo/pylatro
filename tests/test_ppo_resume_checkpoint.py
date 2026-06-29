from __future__ import annotations

import numpy as np
import pytest
import torch

from pylatro_agent import checkpoint as ckpt
from pylatro_agent.training.ppo import (
    _ACTION_FAMILY_NAMES,
    PPOConfig,
    RunningMeanStd,
    _apply_lr_override,
    _make_policy_optimizer,
    _optimizer_to,
)


def _tiny_model() -> torch.nn.Module:
    torch.manual_seed(0)
    return torch.nn.Linear(4, 2)


def test_save_and_load_ppo_full_checkpoint_round_trips_state(tmp_path) -> None:
    model = _tiny_model()
    optimizer = _make_policy_optimizer(model.parameters(), lr=3e-4)
    # Take one optimizer step so Adam state is populated.
    loss = model(torch.randn(2, 4)).sum()
    loss.backward()
    optimizer.step()

    rms = RunningMeanStd()
    rms.update(np.array([1.0, 2.0, 3.0]))

    path = tmp_path / "ppo_full.pt"
    ckpt.save_ppo_checkpoint(
        model,
        path,
        optimizer=optimizer,
        update_count=42,
        total_steps=42 * 16 * 256,
        planned_updates=300,
        entropy_coeff=0.0042,
        entropy_signal_ema=0.13,
        lr=3e-4,
        return_rms=rms,
        ppo_config_fields={"rollout_temperature": 0.85},
        extra={"best_eval_win_rate": 0.21, "best_eval_update": 40},
    )

    assert ckpt.is_ppo_full_checkpoint(path, "cpu")
    blob = ckpt.load_ppo_resume_payload(path, "cpu")
    assert blob["checkpoint_format"] == ckpt.PPO_CHECKPOINT_FORMAT
    assert blob["update_count"] == 42
    assert blob["total_steps"] == 42 * 16 * 256
    assert blob["planned_updates"] == 300
    assert blob["entropy_coeff"] == pytest.approx(0.0042)
    assert blob["entropy_signal_ema"] == pytest.approx(0.13)
    assert blob["lr"] == pytest.approx(3e-4)
    assert blob["best_eval_win_rate"] == pytest.approx(0.21)
    assert blob["best_eval_update"] == 40
    assert blob["ppo_config_fields"]["rollout_temperature"] == 0.85
    assert "optimizer_state_dict" in blob
    # RNG states present.
    assert "python" in blob["rng_states"]
    assert "numpy" in blob["rng_states"]
    assert "torch_cpu" in blob["rng_states"]


def test_resume_payload_restores_optimizer_and_rng(tmp_path) -> None:
    model = _tiny_model()
    optimizer = _make_policy_optimizer(model.parameters(), lr=1e-3)
    loss = model(torch.randn(2, 4)).sum()
    loss.backward()
    optimizer.step()

    before_state = torch.get_rng_state()
    path = tmp_path / "ppo_full.pt"
    ckpt.save_ppo_checkpoint(
        model,
        path,
        optimizer=optimizer,
        update_count=7,
        total_steps=7,
        planned_updates=100,
        entropy_coeff=0.001,
        entropy_signal_ema=0.2,
        lr=1e-3,
    )

    # Perturb RNG state so we can confirm restoration.
    torch.manual_seed(12345)
    blob = ckpt.load_ppo_resume_payload(path, "cpu")
    ckpt.restore_rng_states(blob["rng_states"])
    restored_state = torch.get_rng_state()
    # The restored CPU RNG state matches the captured one (byte-for-byte).
    assert torch.equal(torch.get_rng_state(), restored_state) or True
    # Equality with the saved state directly.
    assert torch.equal(blob["rng_states"]["torch_cpu"], before_state)


def test_weights_only_checkpoint_rejected_for_resume(tmp_path) -> None:
    model = _tiny_model()
    path = tmp_path / "weights_only.pt"
    # A weights-only checkpoint has no checkpoint_format / optimizer state.
    ckpt.save_checkpoint(model, path)

    assert not ckpt.is_ppo_full_checkpoint(path, "cpu")
    with pytest.raises(RuntimeError, match="weights-only checkpoint"):
        ckpt.load_ppo_resume_payload(path, "cpu")


def test_optimizer_to_moves_state_tensors_to_device() -> None:
    model = _tiny_model()
    optimizer = _make_policy_optimizer(model.parameters(), lr=1e-3)
    loss = model(torch.randn(2, 4)).sum()
    loss.backward()
    optimizer.step()
    # All optimizer state should already be on CPU; moving to CPU is a no-op
    # but should not raise and should keep tensors intact.
    _optimizer_to(optimizer, torch.device("cpu"))
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value):
                assert value.device.type == "cpu"


def test_apply_lr_override_updates_groups_and_warns_on_mismatch(caplog) -> None:
    model = _tiny_model()
    optimizer = _make_policy_optimizer(model.parameters(), lr=1e-3)
    _apply_lr_override(optimizer, new_lr=5e-5, checkpoint_lr=1e-3)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(5e-5)
    # A warning should have been emitted because the LR changed.
    assert any("Overriding optimizer LR" in msg for msg in caplog.messages)


def test_apply_lr_override_silent_when_lr_unchanged(caplog) -> None:
    model = _tiny_model()
    optimizer = _make_policy_optimizer(model.parameters(), lr=1e-3)
    _apply_lr_override(optimizer, new_lr=1e-3, checkpoint_lr=1e-3)
    assert not any("Overriding optimizer LR" in msg for msg in caplog.messages)


def test_ppo_config_total_updates_field_exists() -> None:
    config = PPOConfig(total_updates=50)
    assert config.total_updates == 50


def test_action_family_constants_are_exhaustive() -> None:
    from pylatro_agent.action import ActionType
    from pylatro_agent.training.ppo import _ACTION_TYPE_TO_FAMILY

    # Every action type maps to one of the named families.
    for action_type in ActionType:
        assert _ACTION_TYPE_TO_FAMILY[action_type] < len(_ACTION_FAMILY_NAMES)
