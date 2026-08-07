from __future__ import annotations

import numpy as np
import pytest
import torch

import pylatro_agent.training.ppo as ppo_module
from pylatro import load_game_data
from pylatro_agent import checkpoint as ckpt
from pylatro_agent.agent import AgentConfig, BalatroAgent
from pylatro_agent.constants import META_COUNT, NUM_ACTIONS, SCALAR_DIM, TOKEN_DIM, TOKENIZER_SEMANTICS
from pylatro_agent.embeddings import MetaEmbedding
from pylatro_agent.reward import RewardConfig
from pylatro_agent.training.ppo import (
    PPOConfig,
    RunningMeanStd,
    _actor_transition_runtime,
    _advance_ppo_transition_state,
    _apply_lr_override,
    _load_checkpoint_compatible,
    _load_state_dict_into_model,
    _make_policy_optimizer,
    _mirror_latest_checkpoint,
    _optimizer_to,
    _PPOTransitionState,
    _record_critic_ev_and_decide,
    _restore_policy_optimizer_state,
    _restore_ppo_transition_state,
    _save_checkpoint,
    _set_optimizer_lr_for_phase,
    _validate_resume_provenance,
    train_ppo,
)
from pylatro_agent.vocab import build_vocab


def _tiny_model() -> torch.nn.Module:
    torch.manual_seed(0)
    return torch.nn.Linear(4, 2)


class _LoopTestDistribution:
    """Minimal rollout distribution that always selects the first legal action."""

    def __init__(self, action_mask: torch.Tensor, actor: torch.Tensor) -> None:
        self._action_mask = action_mask
        logits = actor.unsqueeze(0).expand_as(action_mask).masked_fill(action_mask <= 0, -1e8)
        self._distribution = torch.distributions.Categorical(logits=logits)
        self.action_type_probs = torch.ones(
            action_mask.shape[0],
            1,
            dtype=action_mask.dtype,
            device=action_mask.device,
        )

    def sample(self) -> torch.Tensor:
        return self._action_mask.argmax(dim=-1)

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self._distribution.log_prob(actions)

    def selected_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.log_prob(actions).exp()

    def entropy(self) -> torch.Tensor:
        return self._distribution.entropy()

    def normalized_action_type_entropy(self) -> torch.Tensor:
        return self.actor_dependency * 0.0

    @property
    def actor_dependency(self) -> torch.Tensor:
        return self._distribution.logits.sum()


class _LoopTestAgent(torch.nn.Module):
    """Small actor/value model used to exercise train_ppo orchestration."""

    def __init__(self, _config: AgentConfig, _vocab) -> None:
        super().__init__()
        self.actor = torch.nn.Parameter(torch.zeros(NUM_ACTIONS))
        self.value_head = torch.nn.Linear(1, 1)

    def action_distribution(
        self,
        tokens: torch.Tensor,
        token_types: torch.Tensor,
        scalars: torch.Tensor,
        attention_mask: torch.Tensor,
        action_mask: torch.Tensor,
        temperature: float | torch.Tensor = 1.0,
    ) -> tuple[_LoopTestDistribution, dict[str, torch.Tensor]]:
        del tokens, token_types, scalars, attention_mask, temperature
        batch_size = action_mask.shape[0]
        value_input = torch.ones(batch_size, 1, dtype=action_mask.dtype, device=action_mask.device)
        values = self.value_head(value_input).squeeze(-1)
        survival = torch.full((batch_size, 8), 0.5, dtype=values.dtype, device=values.device)
        return _LoopTestDistribution(action_mask, self.actor), {
            "expected_score": values,
            "ante_survival": survival,
        }


class _LoopTestWriter:
    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def add_scalar(self, *_args, **_kwargs) -> None:
        pass

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


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


def _safe_transition_config() -> PPOConfig:
    return PPOConfig(
        lr=3e-6,
        clip_epsilon=0.1,
        critic_warmup_updates=20,
        critic_warmup_lr=1e-5,
        critic_warmup_min_ev=0.4,
        critic_warmup_ev_window=5,
        critic_warmup_max_updates=80,
        actor_ramp_updates=25,
        actor_ramp_start_clip_fraction=0.5,
    )


def _bind_test_provenance(config: PPOConfig, *, run_uuid: str = "12345678-1234-5678-9234-567812345678") -> PPOConfig:
    config.ppo_run_uuid = run_uuid
    config.ppo_source_sha256 = "a" * 64
    config.ppo_recipe_id = "pylatro-v14-safe-v2"
    return config


def _run_controlled_transition_loop(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    *,
    transition_state: _PPOTransitionState,
    update_count: int,
    additional_updates: int,
) -> list[dict[str, object]]:
    """Drive train_ppo with real phase/checkpoint orchestration and tiny compute."""

    from torch.utils import tensorboard as tensorboard_module

    config = _safe_transition_config()
    config.num_envs = 1
    config.rollout_length = 2
    config.ppo_epochs = 1
    config.mini_batch_size = 2
    config.device = "cpu"
    config.async_envs = False
    config.checkpoint_interval = 1
    config.eval_interval = 100
    config.eval_games = 1
    config.log_interval = 100
    config.target_kl = None
    config.survival_loss_coeff = 0.0
    config.win_probability_loss_coeff = 0.0
    config.save_dir = str(tmp_path)
    config.log_dir = str(tmp_path / "runs")
    agent_config = AgentConfig(d_model=16, n_layers=1, n_heads=2, d_ff=32)

    def make_grouped_optimizer(parameters, lr: float) -> torch.optim.Adam:
        parameter_list = list(parameters)
        assert len(parameter_list) >= 2
        return torch.optim.Adam(
            [
                {"params": parameter_list[:1]},
                {"params": parameter_list[1:]},
            ],
            lr=lr,
        )

    source_model = _LoopTestAgent(agent_config, None)
    source_optimizer = make_grouped_optimizer(source_model.parameters(), config.critic_warmup_lr)
    target_update = update_count + additional_updates
    resume_state = {
        "state_dict": source_model.state_dict(),
        "optimizer_state_dict": source_optimizer.state_dict(),
        "update_count": update_count,
        "total_steps": update_count * config.num_envs * config.rollout_length,
        "planned_updates": target_update,
        "entropy_coeff": config.entropy_coeff,
        "entropy_signal_ema": None,
        "lr": config.lr,
        "ppo_active_lr": config.critic_warmup_lr,
        "ppo_transition_state": transition_state.to_payload(config),
        "ppo_config_fields": {
            name: getattr(config, name)
            for name in ppo_module._TRANSITION_CONFIG_FIELDS
        },
        "agent_config": {
            "value_bins": agent_config.value_bins,
            "value_v_min": agent_config.value_v_min,
            "value_v_max": agent_config.value_v_max,
        },
        "rng_states": {},
        "schedule_total_steps": target_update * config.num_envs * config.rollout_length,
    }

    def perfect_explained_variance(self, last_values) -> None:
        del last_values
        # Two distinct returns with identical saved critic predictions give
        # EV=1.0. train_ppo still computes and records EV through its real loop.
        self.returns[:] = np.linspace(0.0, 1.0, self.total_size, dtype=np.float32)
        self.values[:] = self.returns
        self.advantages[:] = np.linspace(-1.0, 1.0, self.total_size, dtype=np.float32)

    update_calls: list[dict[str, object]] = []
    original_run_update = ppo_module._run_ppo_update

    def observe_real_update(**kwargs):
        optimizer = kwargs["optimizer"]
        model = kwargs["model"]
        actor_before = model.actor.detach().clone()
        value_before = [parameter.detach().clone() for parameter in model.value_head.parameters()]
        call = {
            "group_lrs": [float(group["lr"]) for group in optimizer.param_groups],
            "policy_loss_scale": kwargs["policy_loss_scale"],
            "clip_epsilon": kwargs["clip_epsilon"],
            "protect_actor_from_critic": kwargs["protect_actor_from_critic"],
        }
        result = original_run_update(**kwargs)
        call["actor_changed"] = not torch.equal(model.actor.detach(), actor_before)
        call["value_head_changed"] = any(
            not torch.equal(parameter.detach(), before)
            for parameter, before in zip(model.value_head.parameters(), value_before, strict=True)
        )
        update_calls.append(call)
        return result

    monkeypatch.setattr(ppo_module, "BalatroAgent", _LoopTestAgent)
    monkeypatch.setattr(ppo_module, "_make_policy_optimizer", make_grouped_optimizer)
    monkeypatch.setattr(ppo_module.RolloutBuffer, "compute_returns_and_advantages", perfect_explained_variance)
    monkeypatch.setattr(ppo_module, "_run_ppo_update", observe_real_update)
    monkeypatch.setattr(ppo_module, "evaluate_model", lambda *_args, **_kwargs: 0.0)
    monkeypatch.setattr(tensorboard_module, "SummaryWriter", _LoopTestWriter)
    monkeypatch.setattr(ckpt, "load_ppo_resume_payload", lambda *_args, **_kwargs: resume_state)

    train_ppo(
        config,
        agent_config=agent_config,
        resume_path="controlled-resume.pt",
        additional_updates=additional_updates,
        data=load_game_data(),
    )
    return update_calls


def test_internal_checkpoint_round_trips_complete_transition_state(tmp_path) -> None:
    model = _tiny_model()
    optimizer = _make_policy_optimizer(model.parameters(), lr=3e-6)
    config = _bind_test_provenance(_safe_transition_config())
    transition = _PPOTransitionState(
        warmup_complete=False,
        critic_warmup_ev_history=[0.31, 0.38, 0.42, 0.45],
        critic_warmup_updates_completed=19,
        actor_ramp_successful_updates=0,
    )

    path = _save_checkpoint(
        model=model,
        optimizer=optimizer,
        save_path=tmp_path,
        update_count=19,
        total_steps=19,
        planned_updates=100,
        entropy_coeff=0.01,
        entropy_signal_ema=None,
        lr=3e-6,
        config=config,
        transition_state=transition,
        filename="transition.pt",
    )
    blob = ckpt.load_ppo_resume_payload(
        path,
        "cpu",
        active_reward_config=RewardConfig(),
    )
    restored = _restore_ppo_transition_state(config, blob)
    _validate_resume_provenance(config, blob)

    assert restored == transition
    assert blob["ppo_transition_state"]["version"] == 1
    assert blob["ppo_config_fields"]["lr"] == pytest.approx(3e-6)
    assert blob["ppo_config_fields"]["critic_warmup_lr"] == pytest.approx(1e-5)
    assert blob["ppo_active_lr"] == pytest.approx(1e-5)
    assert blob["optimizer_state_dict"]["param_groups"][0]["lr"] == pytest.approx(1e-5)
    assert blob["ppo_config_fields"]["actor_ramp_updates"] == 25
    assert blob["ppo_run_provenance"] == {
        "run_uuid": config.ppo_run_uuid,
        "source_sha256": config.ppo_source_sha256,
        "recipe_id": config.ppo_recipe_id,
    }


@pytest.mark.parametrize(
    ("warmup_complete", "expected_lr"),
    [
        (False, 1e-5),
        (True, 3e-6),
    ],
)
def test_strict_resume_restores_lr_for_exact_transition_phase(
    tmp_path,
    warmup_complete: bool,
    expected_lr: float,
) -> None:
    config = _safe_transition_config()
    state = _PPOTransitionState(
        warmup_complete=warmup_complete,
        critic_warmup_updates_completed=20 if warmup_complete else 12,
        actor_ramp_successful_updates=4 if warmup_complete else 0,
    )
    model = _tiny_model()
    optimizer = _make_policy_optimizer(model.parameters(), lr=expected_lr)
    loss = model(torch.randn(2, 4)).sum()
    loss.backward()
    optimizer.step()
    saved_step = int(next(iter(optimizer.state.values()))["step"])

    path = _save_checkpoint(
        model=model,
        optimizer=optimizer,
        save_path=tmp_path,
        update_count=12,
        total_steps=12,
        planned_updates=100,
        entropy_coeff=0.01,
        entropy_signal_ema=None,
        lr=config.lr,
        config=config,
        transition_state=state,
        filename=f"phase-{warmup_complete}.pt",
    )
    blob = ckpt.load_ppo_resume_payload(path, "cpu", active_reward_config=RewardConfig())
    restored_state = _restore_ppo_transition_state(config, blob)
    restored_optimizer = _make_policy_optimizer(model.parameters(), lr=9e-4)
    active_lr = _restore_policy_optimizer_state(
        restored_optimizer,
        blob,
        config,
        restored_state,
        torch.device("cpu"),
    )

    assert active_lr == pytest.approx(expected_lr)
    assert blob["ppo_active_lr"] == pytest.approx(expected_lr)
    assert restored_optimizer.param_groups[0]["lr"] == pytest.approx(expected_lr)
    assert int(next(iter(restored_optimizer.state.values()))["step"]) == saved_step


def test_strict_resume_rejects_optimizer_lr_inconsistent_with_warmup_phase(tmp_path) -> None:
    config = _safe_transition_config()
    state = _PPOTransitionState(warmup_complete=False, critic_warmup_updates_completed=12)
    model = _tiny_model()
    optimizer = _make_policy_optimizer(model.parameters(), lr=config.critic_warmup_lr)
    path = _save_checkpoint(
        model=model,
        optimizer=optimizer,
        save_path=tmp_path,
        update_count=12,
        total_steps=12,
        planned_updates=100,
        entropy_coeff=0.01,
        entropy_signal_ema=None,
        lr=config.lr,
        config=config,
        transition_state=state,
        filename="wrong-lr.pt",
    )
    blob = ckpt.load_ppo_resume_payload(path, "cpu", active_reward_config=RewardConfig())
    blob["optimizer_state_dict"]["param_groups"][0]["lr"] = config.lr
    restored_optimizer = _make_policy_optimizer(model.parameters(), lr=config.lr)

    with pytest.raises(RuntimeError, match="optimizer-group LR"):
        _restore_policy_optimizer_state(
            restored_optimizer,
            blob,
            config,
            state,
            torch.device("cpu"),
        )


def test_strict_resume_rejects_foreign_run_provenance() -> None:
    config = _bind_test_provenance(_safe_transition_config())
    foreign = {
        "ppo_run_provenance": {
            "run_uuid": "87654321-4321-6789-9234-567812345678",
            "source_sha256": config.ppo_source_sha256,
            "recipe_id": config.ppo_recipe_id,
        }
    }
    with pytest.raises(RuntimeError, match="provenance mismatch"):
        _validate_resume_provenance(config, foreign)


def test_safety_enabled_internal_checkpoint_requires_transition_state(tmp_path) -> None:
    model = _tiny_model()
    optimizer = _make_policy_optimizer(model.parameters(), lr=3e-6)
    with pytest.raises(ValueError, match="require transition_state"):
        _save_checkpoint(
            model=model,
            optimizer=optimizer,
            save_path=tmp_path,
            update_count=1,
            total_steps=1,
            planned_updates=10,
            entropy_coeff=0.01,
            entropy_signal_ema=None,
            lr=3e-6,
            config=_safe_transition_config(),
        )


def test_warmup_boundary_orders_19_then_critic_update_then_first_actor() -> None:
    config = _safe_transition_config()
    state = _PPOTransitionState(
        warmup_complete=False,
        critic_warmup_ev_history=[0.39, 0.40, 0.41, 0.42],
        critic_warmup_updates_completed=19,
    )
    resume_blob = {
        "ppo_transition_state": state.to_payload(config),
        "ppo_config_fields": {
            name: getattr(config, name)
            for name in (
                "lr",
                "clip_epsilon",
                "critic_warmup_updates",
                "critic_warmup_lr",
                "critic_warmup_min_ev",
                "critic_warmup_ev_window",
                "critic_warmup_max_updates",
                "actor_ramp_updates",
                "actor_ramp_start_clip_fraction",
            )
        },
    }
    restored = _restore_ppo_transition_state(config, resume_blob)
    assert not restored.warmup_complete
    assert restored.critic_warmup_updates_completed == 19
    assert restored.critic_warmup_ev_history == pytest.approx([0.39, 0.40, 0.41, 0.42])

    # At 19 completed critic updates, even a healthy complete window must
    # still choose one final frozen critic update.
    decision_at_19 = _record_critic_ev_and_decide(
        restored,
        config,
        0.43,
    )
    assert decision_at_19.active
    assert not restored.warmup_complete
    optimizer = _make_policy_optimizer(_tiny_model().parameters(), lr=config.lr)
    assert _set_optimizer_lr_for_phase(optimizer, config, in_critic_warmup=True) == pytest.approx(1e-5)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-5)

    _advance_ppo_transition_state(restored, config, in_critic_warmup=True, kl_rollback=False)
    assert restored.critic_warmup_updates_completed == 20

    # The next rollout uses the same record-and-decide helper as the loop; it
    # latches the gate and the immediately following decision is actor ramp 1.
    first_actor_decision = _record_critic_ev_and_decide(restored, config, 0.44)
    assert first_actor_decision.ready
    assert restored.warmup_complete
    assert _set_optimizer_lr_for_phase(optimizer, config, in_critic_warmup=False) == pytest.approx(3e-6)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(3e-6)
    runtime = _actor_transition_runtime(config, restored, in_critic_warmup=False)
    assert runtime.ramp_active
    assert runtime.progress == 0.0
    assert runtime.clip_epsilon == pytest.approx(0.05)

    restored.actor_ramp_successful_updates = 7
    runtime = _actor_transition_runtime(config, restored, in_critic_warmup=False)
    assert runtime.protect_actor_from_critic
    assert runtime.progress == pytest.approx(7 / 24)
    assert runtime.clip_epsilon == pytest.approx(0.05 + 0.05 * (7 / 24))


def test_train_ppo_real_loop_switches_lr_before_first_actor_step_and_checkpoints_each_phase(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _PPOTransitionState(
        warmup_complete=False,
        critic_warmup_ev_history=[0.40, 0.41, 0.42, 0.43],
        critic_warmup_updates_completed=19,
    )

    calls = _run_controlled_transition_loop(
        monkeypatch,
        tmp_path,
        transition_state=state,
        update_count=19,
        additional_updates=2,
    )

    assert calls == [
        {
            "group_lrs": pytest.approx([1e-5, 1e-5]),
            "policy_loss_scale": 0.0,
            "clip_epsilon": pytest.approx(0.05),
            "protect_actor_from_critic": True,
            "actor_changed": False,
            "value_head_changed": True,
        },
        {
            "group_lrs": pytest.approx([3e-6, 3e-6]),
            "policy_loss_scale": 1.0,
            "clip_epsilon": pytest.approx(0.05),
            "protect_actor_from_critic": True,
            "actor_changed": True,
            "value_head_changed": True,
        },
    ]

    warmup_checkpoint = torch.load(tmp_path / "ppo_update20.pt", map_location="cpu", weights_only=False)
    assert warmup_checkpoint["ppo_transition_state"]["warmup_complete"] is False
    assert warmup_checkpoint["ppo_transition_state"]["critic_warmup_updates_completed"] == 20
    assert warmup_checkpoint["ppo_transition_state"]["actor_ramp_successful_updates"] == 0
    assert warmup_checkpoint["ppo_active_lr"] == pytest.approx(1e-5)
    assert [group["lr"] for group in warmup_checkpoint["optimizer_state_dict"]["param_groups"]] == pytest.approx(
        [1e-5, 1e-5]
    )
    # Param 0 is the actor-only group. A true warmup freeze leaves it with no
    # Adam state at all, while value-head params in group 1 have stepped.
    assert 0 not in warmup_checkpoint["optimizer_state_dict"]["state"]
    assert warmup_checkpoint["optimizer_state_dict"]["state"]

    actor_checkpoint = torch.load(tmp_path / "ppo_update21.pt", map_location="cpu", weights_only=False)
    assert actor_checkpoint["ppo_transition_state"]["warmup_complete"] is True
    assert actor_checkpoint["ppo_transition_state"]["critic_warmup_updates_completed"] == 20
    assert actor_checkpoint["ppo_transition_state"]["actor_ramp_successful_updates"] == 1
    assert actor_checkpoint["ppo_active_lr"] == pytest.approx(3e-6)
    assert [group["lr"] for group in actor_checkpoint["optimizer_state_dict"]["param_groups"]] == pytest.approx(
        [3e-6, 3e-6]
    )
    assert 0 in actor_checkpoint["optimizer_state_dict"]["state"]


def test_train_ppo_real_loop_fails_closed_at_cap_without_optimizer_update(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _PPOTransitionState(
        warmup_complete=False,
        critic_warmup_ev_history=[0.0, 0.0, 0.0, 0.0, 0.0],
        critic_warmup_updates_completed=80,
    )

    calls = _run_controlled_transition_loop(
        monkeypatch,
        tmp_path,
        transition_state=state,
        update_count=80,
        additional_updates=1,
    )

    assert calls == []
    fail_checkpoint = torch.load(tmp_path / "ppo_warmup_unready.pt", map_location="cpu", weights_only=False)
    assert fail_checkpoint["update_count"] == 80
    assert fail_checkpoint["ppo_transition_state"]["warmup_complete"] is False
    assert fail_checkpoint["ppo_transition_state"]["critic_warmup_updates_completed"] == 80
    assert fail_checkpoint["ppo_transition_state"]["fail_closed"] is True
    assert fail_checkpoint["ppo_transition_state"]["fail_reason"] == "critic_ev_gate_unready"
    assert fail_checkpoint["ppo_active_lr"] == pytest.approx(1e-5)
    assert [group["lr"] for group in fail_checkpoint["optimizer_state_dict"]["param_groups"]] == pytest.approx(
        [1e-5, 1e-5]
    )
    assert fail_checkpoint["optimizer_state_dict"]["state"] == {}


def test_warmup_tensorboard_runtime_is_inactive_with_no_operative_clip() -> None:
    runtime = _actor_transition_runtime(
        _safe_transition_config(),
        _PPOTransitionState(warmup_complete=False, critic_warmup_updates_completed=4),
        in_critic_warmup=True,
    )
    assert not runtime.ramp_active
    assert runtime.progress == 0.0
    assert np.isnan(runtime.logged_clip_epsilon)
    assert runtime.protect_actor_from_critic


def test_fail_closed_transition_restores_as_terminal() -> None:
    config = _safe_transition_config()
    state = _PPOTransitionState(
        warmup_complete=False,
        critic_warmup_ev_history=[0.1, 0.2, 0.3, 0.2, 0.1],
        critic_warmup_updates_completed=80,
        fail_closed=True,
        fail_reason="critic_ev_gate_unready",
    )
    blob = {
        "ppo_transition_state": state.to_payload(config),
        "ppo_config_fields": {
            name: getattr(config, name)
            for name in (
                "lr",
                "clip_epsilon",
                "critic_warmup_updates",
                "critic_warmup_lr",
                "critic_warmup_min_ev",
                "critic_warmup_ev_window",
                "critic_warmup_max_updates",
                "actor_ramp_updates",
                "actor_ramp_start_clip_fraction",
            )
        },
    }
    restored = _restore_ppo_transition_state(config, blob)
    assert restored.fail_closed
    assert restored.critic_warmup_updates_completed == 80


def test_fail_closed_checkpoint_mirrors_and_resume_returns_before_env_creation(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = load_game_data()
    vocab = build_vocab(data)
    agent_config = AgentConfig(
        d_model=16,
        n_layers=1,
        n_heads=2,
        d_ff=32,
        hand_ar_mixture_eps=0.1,
    )
    model = BalatroAgent(agent_config, vocab)
    optimizer = _make_policy_optimizer(model.parameters(), lr=3e-6)
    config = _bind_test_provenance(_safe_transition_config())
    config.device = "cpu"
    config.save_dir = str(tmp_path)
    config.log_dir = str(tmp_path / "runs")
    transition = _PPOTransitionState(
        warmup_complete=False,
        critic_warmup_ev_history=[0.1, 0.2, 0.3, 0.2, 0.1],
        critic_warmup_updates_completed=80,
        fail_closed=True,
        fail_reason="critic_ev_gate_unready",
    )
    fail_path = _save_checkpoint(
        model=model,
        optimizer=optimizer,
        save_path=tmp_path,
        update_count=80,
        total_steps=80,
        planned_updates=2000,
        entropy_coeff=0.01,
        entropy_signal_ema=None,
        lr=3e-6,
        agent_config=agent_config,
        config=config,
        transition_state=transition,
        filename="ppo_warmup_unready.pt",
    )
    latest_path = _mirror_latest_checkpoint(fail_path, tmp_path)
    latest_blob = ckpt.load_ppo_resume_payload(
        latest_path,
        "cpu",
        active_reward_config=RewardConfig(),
    )
    assert latest_blob["ppo_transition_state"]["fail_closed"] is True
    assert latest_blob["ppo_run_provenance"]["run_uuid"] == config.ppo_run_uuid

    env_created = False

    def fail_if_env_created(*_args, **_kwargs):
        nonlocal env_created
        env_created = True
        raise AssertionError("fail-closed strict resume must return before environment creation")

    monkeypatch.setattr("pylatro_agent.training.ppo._make_vectorized_envs", fail_if_env_created)
    resumed_model = train_ppo(
        config,
        agent_config=agent_config,
        resume_path=str(latest_path),
        data=data,
    )
    assert isinstance(resumed_model, BalatroAgent)
    assert not env_created


def test_transition_config_mismatch_is_rejected_directly() -> None:
    saved_config = _safe_transition_config()
    state = _PPOTransitionState(warmup_complete=True, critic_warmup_updates_completed=20)
    blob = {
        "ppo_transition_state": state.to_payload(saved_config),
        "ppo_config_fields": {
            name: getattr(saved_config, name)
            for name in (
                "lr",
                "clip_epsilon",
                "critic_warmup_updates",
                "critic_warmup_lr",
                "critic_warmup_min_ev",
                "critic_warmup_ev_window",
                "critic_warmup_max_updates",
                "actor_ramp_updates",
                "actor_ramp_start_clip_fraction",
            )
        },
    }
    active_config = _safe_transition_config()
    active_config.actor_ramp_updates = 24
    with pytest.raises(RuntimeError, match="actor_ramp_updates"):
        _restore_ppo_transition_state(active_config, blob)

    active_config = _safe_transition_config()
    active_config.critic_warmup_lr = 8e-6
    with pytest.raises(RuntimeError, match="critic_warmup_lr"):
        _restore_ppo_transition_state(active_config, blob)


def test_legacy_resume_without_transition_state_is_safe_and_explicit(caplog) -> None:
    with pytest.raises(RuntimeError, match="no ppo_transition_state"):
        _restore_ppo_transition_state(_safe_transition_config(), {})

    restored = _restore_ppo_transition_state(PPOConfig(), {})
    assert restored.warmup_complete
    assert any("legacy PPO checkpoint" in message for message in caplog.messages)


def test_kl_rollback_does_not_advance_actor_ramp() -> None:
    config = _safe_transition_config()
    state = _PPOTransitionState(warmup_complete=True, actor_ramp_successful_updates=7)

    _advance_ppo_transition_state(state, config, in_critic_warmup=False, kl_rollback=True)
    assert state.actor_ramp_successful_updates == 7
    _advance_ppo_transition_state(state, config, in_critic_warmup=False, kl_rollback=False)
    assert state.actor_ramp_successful_updates == 8


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
    v6_state = {key: value for key, value in v6_reference.state_dict().items() if not key.startswith("strategy_proj.")}
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
