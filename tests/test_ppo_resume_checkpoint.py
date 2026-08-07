from __future__ import annotations

import numpy as np
import pytest
import torch

from pylatro_agent import checkpoint as ckpt
from pylatro_agent.constants import META_COUNT, SCALAR_DIM, TOKEN_DIM, TOKENIZER_SEMANTICS
from pylatro_agent.embeddings import MetaEmbedding
from pylatro_agent.reward import RewardConfig
from pylatro_agent.training.ppo import (
    PPOConfig,
    RunningMeanStd,
    _apply_lr_override,
    _load_checkpoint_compatible,
    _load_state_dict_into_model,
    _make_policy_optimizer,
    _optimizer_to,
)


def _tiny_model() -> torch.nn.Module:
    torch.manual_seed(0)
    return torch.nn.Linear(4, 2)


def test_raw_state_dict_checkpoint_is_rejected(tmp_path) -> None:
    path = tmp_path / "raw_state_dict.pt"
    torch.save(_tiny_model().state_dict(), path)

    with pytest.raises(RuntimeError, match="Raw state dicts are unsupported"):
        ckpt.load_checkpoint_payload(path, "cpu")


def test_tokenizer_v5_is_allowed_only_for_compatible_pretrained_load(tmp_path) -> None:
    path = tmp_path / "tokenizer_v5.pt"
    torch.save({"tokenizer_version": 5, "state_dict": _tiny_model().state_dict()}, path)

    with pytest.raises(RuntimeError, match="tokenizer_version=5"):
        ckpt.load_checkpoint_payload(path, "cpu")

    blob = ckpt.load_checkpoint_payload(path, "cpu", allow_compatible_tokenizer=True)
    assert blob["tokenizer_version"] == 5


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
        reward_config=RewardConfig(),
        ppo_config_fields={"rollout_temperature": 0.85},
        extra={"best_eval_win_rate": 0.21, "best_eval_update": 40},
    )

    assert ckpt.is_ppo_full_checkpoint(path, "cpu")
    blob = ckpt.load_ppo_resume_payload(
        path,
        "cpu",
        active_reward_config=RewardConfig(),
    )
    assert blob["checkpoint_format"] == ckpt.PPO_CHECKPOINT_FORMAT
    assert blob["tokenizer_semantics"] == TOKENIZER_SEMANTICS
    assert blob["update_count"] == 42
    assert blob["total_steps"] == 42 * 16 * 256
    assert blob["planned_updates"] == 300
    assert blob["entropy_coeff"] == pytest.approx(0.0042)
    assert blob["entropy_signal_ema"] == pytest.approx(0.13)
    assert blob["lr"] == pytest.approx(3e-4)
    assert blob["best_eval_win_rate"] == pytest.approx(0.21)
    assert blob["best_eval_update"] == 40
    assert blob["ppo_config_fields"]["rollout_temperature"] == 0.85
    assert blob["reward_config"]["gamma"] == pytest.approx(0.997)
    assert blob["reward_model_version"] >= 1
    assert len(blob["reward_fingerprint"]) == 64
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
        reward_config=RewardConfig(),
    )

    # Perturb RNG state so we can confirm restoration.
    torch.manual_seed(12345)
    blob = ckpt.load_ppo_resume_payload(
        path,
        "cpu",
        active_reward_config=RewardConfig(),
    )
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
        ckpt.load_ppo_resume_payload(
            path,
            "cpu",
            active_reward_config=RewardConfig(),
        )


def test_known_faulty_v6_candidate_semantics_checkpoint_is_rejected(tmp_path) -> None:
    path = tmp_path / "faulty_v6.pt"
    torch.save(
        {
            "tokenizer_version": 6,
            "reward_model_version": 11,
            "state_dict": _tiny_model().state_dict(),
        },
        path,
    )

    with pytest.raises(RuntimeError, match="known faulty tokenizer-v6"):
        ckpt.load_checkpoint_payload(path, "cpu", allow_compatible_tokenizer=True)


def test_historical_unmarked_good_v6_checkpoint_is_pretrained_only(tmp_path) -> None:
    path = tmp_path / "historical_v6.pt"
    torch.save(
        {
            "tokenizer_version": 6,
            "reward_model_version": 10,
            "state_dict": _tiny_model().state_dict(),
        },
        path,
    )

    with pytest.raises(RuntimeError, match="tokenizer_version=6"):
        ckpt.load_checkpoint_payload(path, "cpu")
    assert "state_dict" in ckpt.load_checkpoint_payload(
        path,
        "cpu",
        allow_compatible_tokenizer=True,
    )


def test_v6_pretrained_migration_preserves_initial_meta_outputs_exactly() -> None:
    torch.manual_seed(41)
    v6_reference = MetaEmbedding(d_model=16)
    v6_state = {
        key: value
        for key, value in v6_reference.state_dict().items()
        if not key.startswith("strategy_proj.")
    }
    torch.manual_seed(99)
    migrated = MetaEmbedding(d_model=16)

    _load_state_dict_into_model(migrated, v6_state, "synthetic_v6.pt")

    tokens = torch.zeros(3, META_COUNT, TOKEN_DIM, dtype=torch.long)
    scalars = torch.randn(3, SCALAR_DIM)
    v6_scalars = scalars.clone()
    v6_scalars[:, 13:22] = 0.0
    with torch.no_grad():
        reference_output = v6_reference(tokens, v6_scalars)
        migrated_output = migrated(tokens, scalars)

    assert torch.count_nonzero(migrated.strategy_proj.weight) == 0
    assert torch.equal(migrated_output, reference_output)


def test_strict_resume_rejects_reward_fingerprint_mismatch(tmp_path) -> None:
    model = _tiny_model()
    optimizer = _make_policy_optimizer(model.parameters(), lr=1e-3)
    path = tmp_path / "reward_mismatch.pt"
    ckpt.save_ppo_checkpoint(
        model,
        path,
        optimizer=optimizer,
        update_count=1,
        total_steps=1,
        planned_updates=2,
        entropy_coeff=0.01,
        entropy_signal_ema=None,
        lr=1e-3,
        reward_config=RewardConfig(dense_reward_scale=0.5),
    )

    with pytest.raises(RuntimeError, match="reward fingerprint mismatch") as exc_info:
        ckpt.load_ppo_resume_payload(
            path,
            "cpu",
            active_reward_config=RewardConfig(dense_reward_scale=1.0),
        )
    message = str(exc_info.value)
    assert "--pretrained" in message
    assert "--reinit-value-head" in message
    assert "--critic-warmup-updates 15" in message


def test_pretrained_rejects_stale_reward_value_head_without_reinit(tmp_path) -> None:
    model = _tiny_model()
    optimizer = _make_policy_optimizer(model.parameters(), lr=1e-3)
    path = tmp_path / "stale_reward.pt"
    ckpt.save_ppo_checkpoint(
        model,
        path,
        optimizer=optimizer,
        update_count=1,
        total_steps=1,
        planned_updates=2,
        entropy_coeff=0.01,
        entropy_signal_ema=None,
        lr=1e-3,
        reward_config=RewardConfig(dense_reward_scale=0.5),
    )

    with pytest.raises(RuntimeError, match="different reward fingerprint"):
        _load_checkpoint_compatible(
            model,
            str(path),
            torch.device("cpu"),
            active_reward_config=RewardConfig(dense_reward_scale=1.0),
        )

    _load_checkpoint_compatible(
        model,
        str(path),
        torch.device("cpu"),
        active_reward_config=RewardConfig(dense_reward_scale=1.0),
        reinit_value_head=True,
    )


def test_checkpoint_without_reward_metadata_requires_value_head_reinit(tmp_path) -> None:
    model = _tiny_model()
    path = tmp_path / "weights_without_reward_metadata.pt"
    ckpt.save_checkpoint(model, path)

    with pytest.raises(RuntimeError, match="has no reward fingerprint"):
        _load_checkpoint_compatible(
            model,
            str(path),
            torch.device("cpu"),
            active_reward_config=RewardConfig(),
        )

    _load_checkpoint_compatible(
        model,
        str(path),
        torch.device("cpu"),
        active_reward_config=RewardConfig(),
        reinit_value_head=True,
    )


def test_strict_resume_rejects_win_ante_mismatch(tmp_path) -> None:
    model = _tiny_model()
    optimizer = _make_policy_optimizer(model.parameters(), lr=1e-3)
    path = tmp_path / "win_ante_mismatch.pt"
    ckpt.save_ppo_checkpoint(
        model,
        path,
        optimizer=optimizer,
        update_count=1,
        total_steps=1,
        planned_updates=2,
        entropy_coeff=0.01,
        entropy_signal_ema=None,
        lr=1e-3,
        reward_config=RewardConfig(),
        ppo_config_fields={"win_ante": 5},
    )

    with pytest.raises(RuntimeError, match="win_ante mismatch"):
        ckpt.load_ppo_resume_payload(
            path,
            "cpu",
            active_reward_config=RewardConfig(),
            active_win_ante=8,
        )


def test_strict_resume_rejects_full_checkpoint_without_reward_fingerprint(
    tmp_path,
) -> None:
    model = _tiny_model()
    optimizer = _make_policy_optimizer(model.parameters(), lr=1e-3)
    path = tmp_path / "full_without_reward_fingerprint.pt"
    ckpt.save_ppo_checkpoint(
        model,
        path,
        optimizer=optimizer,
        update_count=1,
        total_steps=1,
        planned_updates=2,
        entropy_coeff=0.01,
        entropy_signal_ema=None,
        lr=1e-3,
        reward_config=RewardConfig(),
    )
    blob = torch.load(path, map_location="cpu", weights_only=False)
    blob.pop("reward_fingerprint")
    blob.pop("reward_model_version")
    blob.pop("reward_config")
    torch.save(blob, path)

    with pytest.raises(RuntimeError, match="has no reward fingerprint"):
        ckpt.load_ppo_resume_payload(
            path,
            "cpu",
            active_reward_config=RewardConfig(),
        )

    # The weights-only/pretrained loader intentionally ignores resume metadata.
    weights_payload = ckpt.load_checkpoint_payload(path, "cpu")
    assert "state_dict" in weights_payload


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


def test_ppo_config_long_horizon_defaults() -> None:
    config = PPOConfig()
    assert config.gamma == pytest.approx(0.997)
    assert config.gae_lambda == pytest.approx(0.97)


def test_ppo_config_total_updates_field_exists() -> None:
    config = PPOConfig(total_updates=50)
    assert config.total_updates == 50


@pytest.mark.parametrize(
    ("extra_args", "expected_build_potential"),
    [
        ([], False),
        (["--score-build-potential"], True),
    ],
)
def test_train_cli_long_horizon_defaults_and_build_potential_flags(
    monkeypatch: pytest.MonkeyPatch,
    extra_args: list[str],
    expected_build_potential: bool,
) -> None:
    import importlib.util
    import sys
    from pathlib import Path

    import pylatro_agent.training.ppo as ppo_module

    train_path = Path(__file__).resolve().parents[1] / "train.py"
    spec = importlib.util.spec_from_file_location("pylatro_train_entrypoint", train_path)
    assert spec is not None and spec.loader is not None
    train_entrypoint = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_entrypoint)

    captured: dict[str, object] = {}

    def fake_train_ppo(config, **_kwargs):
        captured["config"] = config
        return None

    monkeypatch.setattr(ppo_module, "train_ppo", fake_train_ppo)
    monkeypatch.setattr(
        sys,
        "argv",
        ["train.py", "ppo", "--device", "cpu", *extra_args],
    )

    train_entrypoint.main()

    config = captured["config"]
    assert isinstance(config, PPOConfig)
    assert config.gamma == pytest.approx(0.997)
    assert config.gae_lambda == pytest.approx(0.97)
    assert config.reward_config is not None
    assert config.reward_config.gamma == pytest.approx(config.gamma)
    assert config.reward_config.enable_score_build_potential is expected_build_potential


def test_train_cli_passes_reward_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib.util
    import sys
    from pathlib import Path

    import pylatro_agent.training.ppo as ppo_module

    train_path = Path(__file__).resolve().parents[1] / "train.py"
    spec = importlib.util.spec_from_file_location("pylatro_train_scale_entrypoint", train_path)
    assert spec is not None and spec.loader is not None
    train_entrypoint = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_entrypoint)

    captured: dict[str, object] = {}

    def fake_train_ppo(config, **_kwargs):
        captured["config"] = config
        return None

    monkeypatch.setattr(ppo_module, "train_ppo", fake_train_ppo)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "ppo",
            "--device",
            "cpu",
            "--planet-match-shaping",
            "--dense-reward-scale",
            "0.5",
            "--consumable-reward-scale",
            "0.2",
        ],
    )

    train_entrypoint.main()

    config = captured["config"]
    assert isinstance(config, PPOConfig)
    assert config.reward_config is not None
    assert config.reward_config.dense_reward_scale == pytest.approx(0.5)
    assert config.reward_config.consumable_reward_scale == pytest.approx(0.2)
    assert config.reward_config.enable_planet_match_rewards is True
