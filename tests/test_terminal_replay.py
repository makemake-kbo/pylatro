"""Focused tests for complete-episode terminal outcome replay."""

from __future__ import annotations

import numpy as np
import torch

from pylatro_agent.constants import MAX_SEQ_LEN, NUM_ACTIONS, SCALAR_DIM, TOKEN_DIM
from pylatro_agent.survival import DEFAULT_MAX_ANTES, hazard_outcome_probabilities
from pylatro_agent.training.ppo import PPOConfig
from pylatro_agent.training.ppo_optimization import _run_terminal_replay_updates
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
        self.outcome_proj = torch.nn.Linear(1, 1)
        self.ante_survival = torch.nn.Linear(1, DEFAULT_MAX_ANTES)
        self.return_residual = torch.nn.Linear(1, 1)

    def forward(
        self,
        feature: torch.Tensor,
        current_antes: torch.Tensor,
        win_antes: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        hidden = torch.tanh(self.pool(feature))
        hazards = self.ante_survival(torch.tanh(self.outcome_proj(feature))).sigmoid()
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
        "value_head.outcome_proj.weight",
        "value_head.outcome_proj.bias",
        "value_head.ante_survival.weight",
        "value_head.ante_survival.bias",
    }
    # The whole point of the private outcome projection: replay reshapes the
    # features the hazards read without perturbing the return path GAE uses.
    assert "value_head.outcome_proj.weight" in changed
    assert "value_head.pool.weight" not in changed
    assert "value_head.return_residual.weight" not in changed
    assert "policy_weight" not in changed


def _record_episode(
    buffer: EpisodeReplayBuffer,
    *,
    steps: int,
    won: bool,
    final_ante: int = 2,
    ante: int = 1,
) -> None:
    tracker = EpisodeTracker(num_envs=1, gamma=0.5)
    for step in range(steps):
        tracker.record_step(
            _observation(ante=ante),
            np.asarray([0], dtype=np.int64),
            np.asarray([0.0], dtype=np.float32),
            terminal_rewards=np.asarray([0.0], dtype=np.float32),
            behavior_log_probs=np.asarray([-0.1], dtype=np.float32),
            policy_version=1,
        )
        del step
    tracker.finish_episode(
        0,
        won=won,
        stalled=False,
        final_ante=final_ante,
        win_ante=5,
        terminal_blind="big",
        buffer=buffer,
    )


def test_row_uniform_sampling_is_uniform_over_transitions_at_any_batch_size() -> None:
    """Episode-uniform draws over-weight short episodes; row-uniform does not.

    A 2-step Ante-1 death and a 30-step deep run get equal expected row counts
    under the episode quota. Row-uniform draws should split them 2:30, and must
    do so regardless of batch size -- the quota path's row distribution depends
    on how hard the batch truncates the episode list, which would make training
    and holdout batches of different sizes incomparable.
    """

    short_steps, long_steps = 2, 30
    buffer = EpisodeReplayBuffer(capacity_episodes=4, seed=0)
    _record_episode(buffer, steps=short_steps, won=False)
    _record_episode(buffer, steps=long_steps, won=True)

    def long_row_share(*, batch_size: int, row_uniform: bool) -> float:
        long_rows = 0
        total = 0
        for _ in range(200):
            batch = buffer.sample(
                batch_size,
                torch.device("cpu"),
                samples_per_episode=1,
                include_teacher_forced=True,
                row_uniform=row_uniform,
            )
            assert batch is not None
            # The long episode is the winning one.
            long_rows += int(batch["episode_outcomes"].sum().item())
            total += int(batch["episode_outcomes"].numel())
        return long_rows / total

    expected = long_steps / (short_steps + long_steps)
    assert abs(long_row_share(batch_size=2, row_uniform=False) - 0.5) < 0.05
    for batch_size in (2, 8, 64):
        assert abs(long_row_share(batch_size=batch_size, row_uniform=True) - expected) < 0.05


def test_holdout_episodes_are_never_drawn_for_training() -> None:
    buffer = EpisodeReplayBuffer(capacity_episodes=64, seed=3, holdout_fraction=0.5)
    for index in range(40):
        _record_episode(buffer, steps=3, won=index % 2 == 0)

    assert 0 < buffer.num_holdout_episodes < buffer.num_episodes
    holdout_ids = {
        int(ep["episode_id"]) for ep in buffer._episodes if ep["holdout"]
    }

    train_ids: set[int] = set()
    eval_ids: set[int] = set()
    for _ in range(30):
        train = buffer.sample(
            32, torch.device("cpu"), samples_per_episode=8, include_teacher_forced=True
        )
        assert train is not None
        train_ids.update(int(v) for v in train["episode_ids"].tolist())
        evaluation = buffer.sample(
            32,
            torch.device("cpu"),
            samples_per_episode=8,
            include_teacher_forced=True,
            holdout=True,
        )
        assert evaluation is not None
        eval_ids.update(int(v) for v in evaluation["episode_ids"].tolist())

    assert train_ids and eval_ids
    assert not (train_ids & holdout_ids)
    assert eval_ids <= holdout_ids


def test_holdout_assignment_does_not_perturb_training_draws() -> None:
    """Enabling the split must not shift the sampling RNG stream."""

    def first_batch_ids(holdout_fraction: float) -> list[int]:
        buffer = EpisodeReplayBuffer(
            capacity_episodes=64, seed=11, holdout_fraction=holdout_fraction
        )
        for index in range(20):
            _record_episode(buffer, steps=3, won=index % 3 == 0)
        # Force every episode into the training half so the only possible
        # difference is RNG stream drift, not eligibility.
        for episode in buffer._episodes:
            episode["holdout"] = False
        batch = buffer.sample(
            8, torch.device("cpu"), samples_per_episode=2, include_teacher_forced=True
        )
        assert batch is not None
        return [int(v) for v in batch["episode_ids"].tolist()]

    assert first_batch_ids(0.0) == first_batch_ids(0.25)


def test_climatology_reference_is_unbiased_at_small_bucket_sizes() -> None:
    """The in-batch empirical climatology understates its own Brier by 1 - 1/n.

    Left uncorrected that handicap is credited to the model, which is what made
    the 32-row cross-rollout bucket look worse than climatology.
    """

    from pylatro_agent.training.ppo_optimization import _outcome_metric_rows

    torch.manual_seed(0)
    rows, classes = 32, DEFAULT_MAX_ANTES + 1
    targets = torch.randint(0, classes, (rows,))
    probabilities = torch.full((rows, classes), 1.0 / classes)
    value_dict = {
        "outcome_probabilities": probabilities,
        "win_prob": probabilities[:, -1],
    }
    sampled = {
        "terminal_outcome_target": targets,
        "cross_rollout_flags": torch.zeros(rows),
        "current_antes": torch.ones(rows, dtype=torch.long),
    }
    metrics = _outcome_metric_rows(value_dict, sampled)
    climatology = metrics["outcome_climatology_brier"][0]

    one_hot = torch.nn.functional.one_hot(targets, num_classes=classes).float()
    in_sample = (one_hot.mean(0, keepdim=True) - one_hot).square().sum(-1)
    assert torch.allclose(climatology, in_sample * rows / (rows - 1))
    assert climatology.mean() > in_sample.mean()


def test_replay_reports_explicitly_replay_only_holdout_metrics() -> None:
    buffer = EpisodeReplayBuffer(capacity_episodes=64, seed=5, holdout_fraction=0.5)
    for index in range(24):
        _record_episode(buffer, steps=4, won=index % 2 == 0, final_ante=3)
    assert buffer.num_holdout_episodes > 0

    model = _TinyTerminalModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    config = PPOConfig(
        terminal_replay_batch_size=16,
        terminal_replay_min_episodes=1,
        terminal_replay_samples_per_episode=4,
        terminal_replay_updates_per_ppo_update=2,
        terminal_replay_holdout_batch_size=16,
        outcome_loss_coeff=1.0,
        max_grad_norm=100.0,
        rollout_temperature=1.0,
    )

    result = _run_terminal_replay_updates(
        model, optimizer, buffer, config, torch.device("cpu")
    )

    assert result.updates_applied == 2
    assert result.diagnostics["buffer_holdout_episodes"] > 0
    assert result.diagnostics["replay_holdout/samples"] == 16.0
    # PPO may already have trained on this split; do not call it independent.
    for key in ("outcome_nll", "outcome_brier", "outcome_brier_skill"):
        assert key in result.diagnostics
        assert "replay_holdout/" + key in result.diagnostics
    assert not any(key.startswith("holdout/") for key in result.diagnostics)


def test_replay_skips_holdout_metrics_when_the_split_is_disabled() -> None:
    buffer = EpisodeReplayBuffer(capacity_episodes=64, seed=5)
    for index in range(12):
        _record_episode(buffer, steps=4, won=index % 2 == 0, final_ante=3)
    assert buffer.num_holdout_episodes == 0

    result = _run_terminal_replay_updates(
        _TinyTerminalModel(),
        torch.optim.Adam(_TinyTerminalModel().parameters(), lr=0.05),
        buffer,
        PPOConfig(
            terminal_replay_batch_size=16,
            terminal_replay_min_episodes=1,
            terminal_replay_samples_per_episode=4,
            terminal_replay_updates_per_ppo_update=1,
            outcome_loss_coeff=1.0,
            max_grad_norm=100.0,
            rollout_temperature=1.0,
        ),
        torch.device("cpu"),
    )

    assert result.updates_applied == 1
    assert not any(key.startswith("replay_holdout/") for key in result.diagnostics)


def test_singleton_bucket_keeps_scores_but_publishes_no_climatology() -> None:
    """A one-row Ante bucket must not manufacture skill against itself."""

    from pylatro_agent.training.ppo_optimization import _outcome_metric_rows

    classes = DEFAULT_MAX_ANTES + 1
    probabilities = torch.full((3, classes), 1.0 / classes)
    value_dict = {
        "outcome_probabilities": probabilities,
        "win_prob": probabilities[:, -1],
    }
    sampled = {
        "terminal_outcome_target": torch.tensor([0, 0, 1]),
        "cross_rollout_flags": torch.zeros(3),
        # Ante 4 appears exactly once; Ante 2 twice.
        "current_antes": torch.tensor([2, 2, 4]),
    }
    metrics = _outcome_metric_rows(value_dict, sampled)

    assert "ante_4/outcome_brier" in metrics
    assert "ante_4/outcome_nll" in metrics
    assert "ante_4/outcome_climatology_brier" not in metrics
    assert "ante_2/outcome_climatology_brier" in metrics
