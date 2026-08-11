"""Focused tests for complete-episode terminal outcome replay."""

from __future__ import annotations

import numpy as np
import torch

from pylatro_agent.constants import MAX_SEQ_LEN, NUM_ACTIONS, SCALAR_DIM, TOKEN_DIM
from pylatro_agent.survival import DEFAULT_MAX_ANTES, hazard_outcome_probabilities
from pylatro_agent.training.ppo import PPOConfig, _run_terminal_replay_updates
from pylatro_agent.training.sil import EpisodeReplayBuffer, EpisodeTracker


def _observation(
    *, ante: int, win_ante: int = 5, feature: float = 1.0
) -> dict[str, np.ndarray]:
    scalars = np.zeros((1, SCALAR_DIM), dtype=np.float32)
    scalars[0, 0] = feature
    scalars[0, 2] = float(ante)
    scalars[0, 22] = float(win_ante)
    action_mask = np.zeros((1, NUM_ACTIONS), dtype=np.float32)
    action_mask[:, 0] = 1.0
    return {
        "tokens": np.zeros((1, MAX_SEQ_LEN, TOKEN_DIM), dtype=np.int64),
        "token_types": np.zeros((1, MAX_SEQ_LEN), dtype=np.int64),
        "scalars": scalars,
        "attention_mask": np.ones((1, MAX_SEQ_LEN), dtype=np.int64),
        "action_mask": action_mask,
    }


def _record_cross_rollout_loss(buffer: EpisodeReplayBuffer) -> None:
    tracker = EpisodeTracker(num_envs=1, gamma=0.5)
    for step in range(4):
        tracker.record_step(
            _observation(ante=1 if step < 2 else 2),
            np.asarray([0], dtype=np.int64),
            np.asarray([-4.0 if step == 3 else 0.0], dtype=np.float32),
            terminal_rewards=np.asarray(
                [-4.0 if step == 3 else 0.0], dtype=np.float32
            ),
            behavior_log_probs=np.asarray([-0.1 * (step + 1)], dtype=np.float32),
            policy_version=10 if step < 2 else 11,
        )
    tracker.finish_episode(
        0,
        won=False,
        stalled=False,
        final_ante=2,
        win_ante=5,
        terminal_blind="big",
        buffer=buffer,
    )


def test_complete_episode_replay_labels_prefix_before_rollout_boundary() -> None:
    buffer = EpisodeReplayBuffer(capacity_episodes=4, seed=0)
    _record_cross_rollout_loss(buffer)

    assert buffer.num_episodes == 1
    assert buffer.num_transitions == 4
    assert buffer.labeled_transition_fraction == 1.0
    assert buffer.cross_rollout_transition_fraction == 0.5
    episode = buffer._episodes[0]
    np.testing.assert_allclose(episode["terminal_returns"], [-0.5, -1.0, -2.0, -4.0])
    np.testing.assert_array_equal(episode["policy_versions"], [10, 10, 11, 11])
    assert episode["terminal_outcome_class"] == 1

    batch = buffer.sample(
        4,
        torch.device("cpu"),
        samples_per_episode=4,
        include_teacher_forced=True,
    )
    assert batch is not None
    assert int(batch["cross_rollout_flags"].sum().item()) == 2
    assert torch.all(batch["terminal_outcome_target"] == 1)
    assert torch.all(batch["terminal_outcome_mask"] == 1.0)


def test_stalled_episode_remains_unlabeled_and_out_of_replay() -> None:
    buffer = EpisodeReplayBuffer(capacity_episodes=4, seed=0)
    tracker = EpisodeTracker(num_envs=1)
    tracker.record_step(
        _observation(ante=1),
        np.asarray([0], dtype=np.int64),
        np.asarray([0.0], dtype=np.float32),
    )
    tracker.finish_episode(
        0,
        won=False,
        stalled=True,
        final_ante=1,
        win_ante=5,
        buffer=buffer,
    )
    assert buffer.num_episodes == 0
    assert buffer.stalled_episodes_dropped_total == 1


class _TinyTerminalValueHead(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pool = torch.nn.Linear(1, 1)
        self.ante_survival = torch.nn.Linear(1, DEFAULT_MAX_ANTES)
        self.return_residual = torch.nn.Linear(1, 1)

    def forward(
        self,
        feature: torch.Tensor,
        current_antes: torch.Tensor,
        win_antes: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        hidden = torch.tanh(self.pool(feature))
        hazards = self.ante_survival(hidden).sigmoid()
        outcomes = hazard_outcome_probabilities(hazards, current_antes, win_antes)
        residual = self.return_residual(hidden).squeeze(-1)
        terminal = torch.zeros_like(residual)
        expected = terminal + residual
        return {
            "ante_survival": hazards,
            "outcome_probabilities": outcomes,
            "terminal_value": terminal,
            "return_residual": residual,
            "expected_return": expected,
            "expected_score": expected,
            "win_prob": outcomes[:, -1],
        }


class _TinyTerminalModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.policy_weight = torch.nn.Parameter(torch.tensor([3.0]))
        self.value_head = _TinyTerminalValueHead()

    def action_distribution(
        self,
        tokens: torch.Tensor,
        token_types: torch.Tensor,
        scalars: torch.Tensor,
        attention_mask: torch.Tensor,
        action_mask: torch.Tensor,
        temperature: float | torch.Tensor = 1.0,
    ):
        del tokens, token_types, attention_mask, action_mask, temperature
        return None, self.value_head(scalars[:, :1], scalars[:, 2], scalars[:, 22])


def test_terminal_replay_changes_only_hazard_output_parameters() -> None:
    buffer = EpisodeReplayBuffer(capacity_episodes=4, seed=0)
    _record_cross_rollout_loss(buffer)
    model = _TinyTerminalModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    config = PPOConfig(
        terminal_replay_batch_size=4,
        terminal_replay_min_episodes=1,
        terminal_replay_samples_per_episode=4,
        terminal_replay_updates_per_ppo_update=1,
        outcome_loss_coeff=1.0,
        max_grad_norm=100.0,
        rollout_temperature=1.0,
    )
    before = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }

    result = _run_terminal_replay_updates(
        model, optimizer, buffer, config, torch.device("cpu")
    )

    assert result.updates_applied == 1
    assert result.samples == 4
    assert result.diagnostics["label_coverage"] == 1.0
    changed = {
        name for name, parameter in model.named_parameters()
        if not torch.equal(parameter.detach(), before[name])
    }
    assert changed
    assert changed <= {
        "value_head.ante_survival.weight",
        "value_head.ante_survival.bias",
    }
