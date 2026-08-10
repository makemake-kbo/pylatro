"""Focused tests for complete-episode terminal critic replay."""

from __future__ import annotations

import numpy as np
import torch

from pylatro_agent.constants import MAX_SEQ_LEN, NUM_ACTIONS, SCALAR_DIM, TOKEN_DIM
from pylatro_agent.survival import (
    DEFAULT_MAX_ANTES,
    compute_conditional_ante_survival_targets,
    hazard_outcome_probabilities,
)
from pylatro_agent.training.ppo import PPOConfig, _run_terminal_replay_updates
from pylatro_agent.training.sil import EpisodeReplayBuffer, EpisodeTracker


def _observation(*, ante: int, feature: float = 1.0) -> dict[str, np.ndarray]:
    scalars = np.zeros((1, SCALAR_DIM), dtype=np.float32)
    scalars[0, 0] = feature
    scalars[0, 2] = float(ante)
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
                [-4.0 if step == 3 else 0.0],
                dtype=np.float32,
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


def test_conditional_hazards_form_a_coherent_outcome_distribution() -> None:
    hazards = torch.full((2, DEFAULT_MAX_ANTES), 0.5)
    probabilities = hazard_outcome_probabilities(
        hazards,
        current_antes=torch.tensor([1, 2]),
        win_antes=torch.tensor([3, 3]),
    )

    torch.testing.assert_close(probabilities.sum(dim=-1), torch.ones(2))
    torch.testing.assert_close(
        probabilities[0, [0, 1, 2, DEFAULT_MAX_ANTES]],
        torch.tensor([0.5, 0.25, 0.125, 0.125]),
    )
    assert probabilities[1, 0] == 0.0
    torch.testing.assert_close(
        probabilities[1, [1, 2, DEFAULT_MAX_ANTES]],
        torch.tensor([0.5, 0.25, 0.25]),
    )

    targets, mask = compute_conditional_ante_survival_targets(
        final_ante=4,
        won=True,
        current_ante=2,
        win_ante=4,
    )
    np.testing.assert_array_equal(targets[:5], [0.0, 1.0, 1.0, 1.0, 0.0])
    np.testing.assert_array_equal(mask[:5], [0.0, 1.0, 1.0, 1.0, 0.0])


def test_complete_episode_replay_labels_prefix_before_rollout_boundary() -> None:
    buffer = EpisodeReplayBuffer(capacity_episodes=4, seed=0)
    _record_cross_rollout_loss(buffer)

    assert buffer.num_episodes == 1
    assert buffer.num_transitions == 4
    assert buffer.labeled_transition_fraction == 1.0
    assert buffer.cross_rollout_transition_fraction == 0.5
    episode = buffer._episodes[0]
    np.testing.assert_allclose(episode["terminal_returns"], [-0.5, -1.0, -2.0, -4.0])
    np.testing.assert_allclose(episode["behavior_log_probs"], [-0.1, -0.2, -0.3, -0.4])
    np.testing.assert_array_equal(episode["policy_versions"], [10, 10, 11, 11])
    assert episode["completion_policy_version"] == 11
    assert episode["terminal_outcome_class"] == 1
    assert episode["terminal_blind"] == "big"

    # Ante-1 prefix states are told that Ante 1 survived and Ante 2 failed.
    np.testing.assert_array_equal(
        episode["conditional_survival_targets"][0, :3],
        [1.0, 0.0, 0.0],
    )
    np.testing.assert_array_equal(
        episode["conditional_survival_masks"][0, :3],
        [1.0, 1.0, 0.0],
    )
    # Once already in Ante 2, Ante 1 is historical and excluded from loss.
    np.testing.assert_array_equal(
        episode["conditional_survival_masks"][2, :3],
        [0.0, 1.0, 0.0],
    )

    batch = buffer.sample(
        4,
        torch.device("cpu"),
        samples_per_episode=4,
        include_teacher_forced=True,
    )
    assert batch is not None
    assert int(batch["cross_rollout_flags"].sum().item()) == 2
    assert torch.all(batch["win_probability_target"] == 0.0)
    assert torch.all(batch["win_probability_mask"] == 1.0)
    assert torch.all(batch["terminal_outcome_target"] == 1)


class _TinyTerminalValueHead(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.expected_score = torch.nn.Linear(1, 1)
        self.ante_survival = torch.nn.Linear(1, DEFAULT_MAX_ANTES)
        self.win_prob = torch.nn.Linear(1, 1)
        torch.nn.init.zeros_(self.expected_score.weight)
        torch.nn.init.zeros_(self.expected_score.bias)
        torch.nn.init.zeros_(self.ante_survival.weight)
        torch.nn.init.zeros_(self.ante_survival.bias)
        torch.nn.init.zeros_(self.win_prob.weight)
        torch.nn.init.zeros_(self.win_prob.bias)

    def forward(self, feature: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "expected_score": self.expected_score(feature).squeeze(-1),
            "ante_survival": self.ante_survival(feature).sigmoid(),
            "win_prob": self.win_prob(feature).squeeze(-1).sigmoid(),
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
        return None, self.value_head(scalars[:, :1])


def test_terminal_replay_update_changes_only_terminal_output_layers() -> None:
    buffer = EpisodeReplayBuffer(capacity_episodes=4, seed=0)
    _record_cross_rollout_loss(buffer)
    model = _TinyTerminalModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    config = PPOConfig(
        terminal_replay_batch_size=4,
        terminal_replay_min_episodes=1,
        terminal_replay_samples_per_episode=4,
        terminal_replay_updates_per_ppo_update=1,
        survival_loss_coeff=1.0,
        win_probability_loss_coeff=1.0,
        max_grad_norm=100.0,
        rollout_temperature=1.0,
    )
    policy_before = model.policy_weight.detach().clone()
    expected_before = {
        name: parameter.detach().clone()
        for name, parameter in model.value_head.expected_score.named_parameters()
    }
    survival_before = model.value_head.ante_survival.weight.detach().clone()
    win_before = model.value_head.win_prob.weight.detach().clone()

    result = _run_terminal_replay_updates(
        model,
        optimizer,
        buffer,
        config,
        torch.device("cpu"),
    )

    assert result.updates_applied == 1
    assert result.samples == 4
    assert result.diagnostics["label_coverage"] == 1.0
    assert result.diagnostics["cross_rollout_transition_fraction"] == 0.5
    assert result.diagnostics["cross_rollout/outcome_nll"] > 0.0
    assert result.diagnostics["same_rollout/outcome_nll"] > 0.0
    torch.testing.assert_close(model.policy_weight, policy_before)
    for name, parameter in model.value_head.expected_score.named_parameters():
        torch.testing.assert_close(parameter, expected_before[name])
    assert not torch.equal(model.value_head.ante_survival.weight, survival_before)
    assert not torch.equal(model.value_head.win_prob.weight, win_before)
