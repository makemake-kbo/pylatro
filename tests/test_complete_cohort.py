"""Bounded regressions for the mechanisms in complete_cohort.md."""

from __future__ import annotations

import copy
import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from pylatro import load_game_data
from pylatro_agent.agent import AgentConfig, BalatroAgent
from pylatro_agent.env import BalatroEnv
from pylatro_agent.outcome_study import outcome_reward
from pylatro_agent.training.cohort import (
    CohortBuffer,
    CohortFailure,
    ObservationStore,
    collect_complete_cohort,
    completion_mask,
    fragment_gae,
    frozen_collection_policy,
    policy_digest,
)
from pylatro_agent.training.cohort_io import load_cohort, save_cohort
from pylatro_agent.training.evaluation_envs import EvaluationEnvs
from pylatro_agent.training.ppo_config import PPOConfig, _validate_ppo_config
from pylatro_agent.training.ppo_optimization import _logical_mask_weights, _run_ppo_update
from pylatro_agent.training.ppo_policy import _grammar_distribution, _policy_temperature_for_scalars
from pylatro_agent.value_head import outcome_nll
from pylatro_agent.vocab import build_vocab


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def game_data():
    return load_game_data()


@pytest.fixture
def observation(game_data):
    env = BalatroEnv(
        seed=7000000,
        deck_key="b_blue",
        data=game_data,
        vocab=build_vocab(game_data),
        reward_config=outcome_reward(),
        enable_teacher=False,
    )
    obs, _ = env.reset(seed=7000000)
    env.close()
    return obs


def cohort_config(**overrides):
    values = dict(
        collection_mode="complete_cohort",
        cohort_games=2,
        num_envs=2,
        cohort_training_rows=8,
        cohort_max_episode_steps=12,
        cohort_workers=0,
        cohort_seeds=list(range(7000000, 7000012)),
        total_updates=2,
        ppo_epochs=2,
        rollout_length=4,
        mini_batch_size=4,
        micro_batch_size=2,
        device="cpu",
        precision="fp32",
        terminal_replay_updates_per_ppo_update=0,
        sil_coeff=0,
        return_path_coeff=0,
        reward_config=outcome_reward(),
        gamma=1,
        gae_lambda=0.97,
        outcome_label_mode="complete_episode",
        advantage_clip_sigma=0,
        target_kl=None,
        max_no_progress_steps=256,
        hand_ar_mixture_eps=0.5,
        deck_key="b_blue",
        eval_seeds=[190000],
        training_excluded_seeds=[210000],
        eval_games=1,
        eval_sampled_games=0,
        eval_before_training=False,
        eval_interval=10000,
        eval_workers=0,
    )
    values.update(overrides)
    return PPOConfig(**values)


def tiny_model(game_data):
    torch.manual_seed(13)
    return BalatroAgent(
        AgentConfig(
            d_model=16, n_layers=1, n_heads=2, d_ff=32, dropout=0.2, hand_ar_mixture_eps=0.5, win_only_value=True
        ),
        build_vocab(game_data),
    )


class ScriptedEnvs:
    """A short loss and a long win with the same initial observation."""

    def __init__(self, observation, *, failure=None, inspect=None, **kwargs):
        self.obs = observation
        self.failure = failure
        self.inspect = inspect
        self.steps = {}
        self.resets = []
        self.closed = False

    def reset(self, seed):
        self.steps[seed] = 0
        self.resets.append(seed)
        return copy.deepcopy(self.obs), "SELECT_BLIND"

    def step_training(self, actions):
        if self.inspect:
            self.inspect()
        results = {}
        for seed, _action in reversed(actions):  # Collector must not trust completion order.
            self.steps[seed] += 1
            won = seed % 2 == 1
            finished = self.steps[seed] >= (7 if won else 2)
            if self.failure == "worker":
                raise RuntimeError("Scripted worker failure")
            stalled = self.failure == "stall" and finished
            results[seed] = (
                copy.deepcopy(self.obs),
                10.0 if finished and won else 0.0,
                finished and not stalled,
                stalled,
                {
                    "ante": 9 if finished and won else 1,
                    "won": finished and won,
                    "stalled": stalled,
                    "reward_terminal": 10.0 if finished and won else 0.0,
                },
                0.01,
            )
        return results

    def close(self):
        self.closed = True


def collected(observation, game_data, *, config=None, chunk_size=256):
    config = config or cohort_config()
    model = tiny_model(game_data)
    envs = ScriptedEnvs(observation)
    episodes, counters = collect_complete_cohort(
        model,
        envs,
        config,
        config.cohort_seeds[: config.cohort_games],
        [0] * config.num_envs,
        policy_version=0,
        chunk_size=chunk_size,
    )
    return model, episodes, counters


def test_completion_selected_early_wins_are_missing(observation, game_data):
    config = cohort_config(cohort_training_rows=9)
    _, episodes, _ = collected(observation, game_data, config=config)
    full = CohortBuffer(episodes, config, draw_indices=np.arange(9))
    selected = CohortBuffer(
        episodes, replace(config, outcome_label_mode="completion_window"), draw_indices=np.arange(9)
    )
    starts = full.arrays["step"] == 0
    assert full.arrays["terminal_outcome_mask"][starts].tolist() == [1, 1]
    assert selected.arrays["terminal_outcome_mask"][starts].tolist() == [1, 0]
    assert full.diagnostics()["label_coverage"]["winning_start"] == {"rows": 1, "labels": 1}
    # Selection changes the optimal start-state forecast, although the input
    # distribution and outcomes are fixed. This is our local Rubin application.
    target = torch.tensor([0, 8])
    probabilities = torch.zeros(2, 9)
    probabilities[:, [0, 8]] = 0.5
    pessimistic = probabilities.clone()
    pessimistic[:, 0], pessimistic[:, 8] = 0.99, 0.01
    assert outcome_nll(probabilities, target) < outcome_nll(pessimistic, target)
    mask = torch.tensor([1.0, 0.0])
    assert outcome_nll(pessimistic, target, mask) < outcome_nll(probabilities, target, mask)


@pytest.mark.parametrize("chunk", [64, 128, 256])
def test_storage_chunking_preserves_observations_and_targets(observation, game_data, chunk):
    config = cohort_config(cohort_training_rows=9)
    _, episodes, _ = collected(observation, game_data, config=config, chunk_size=chunk)
    buffer = CohortBuffer(episodes, config, draw_indices=np.arange(9))
    batch = buffer.materialize(np.arange(9), torch.device("cpu"))
    for key, value in observation.items():
        np.testing.assert_array_equal(batch[key][0].numpy(), value)
    np.testing.assert_array_equal(buffer.arrays["terminal_outcome_target"], [0, 0, *([8] * 7)])
    assert buffer.arrays["terminal_outcome_mask"].sum() == 9


def test_observation_store_copies_history_and_excludes_unwritten_rows(observation):
    store = ObservationStore(chunk_size=2)
    original = copy.deepcopy(observation)
    for _ in range(5):
        store.append(observation)
    observation["history_events"][...] = 9
    observation["scalars"][...] = -1
    for key, value in store.take([4, 0, 3]).items():
        np.testing.assert_array_equal(value[0], original[key])
    with pytest.raises(IndexError):
        store.take([5])
    with pytest.raises(ValueError, match="binary"):
        store.append({**original, "action_mask": np.full_like(original["action_mask"], 2)})


@pytest.mark.parametrize("change", ["shape", "dtype", "field"])
def test_observation_schema_cannot_change_after_a_chunk_flush(observation, change):
    store = ObservationStore(chunk_size=1)
    store.append(observation)
    changed = copy.deepcopy(observation)
    if change == "shape":
        changed["scalars"] = changed["scalars"][:-1]
    elif change == "dtype":
        changed["scalars"] = changed["scalars"].astype(np.float64)
    else:
        del changed["history_events"]
    with pytest.raises(ValueError, match="schema"):
        store.append(changed)
    assert store.size == 1


def test_gae_bootstraps_artificial_cuts_but_not_game_termination():
    advantages, returns = fragment_gae([0, 0, 10], [1, 2, 3], [2, 3, 4], gamma=1, gae_lambda=1, fragment_length=4)
    np.testing.assert_allclose(advantages, [2, 1, 7])
    np.testing.assert_allclose(returns, [3, 3, 10])
    terminal, returns = fragment_gae([0], [100], [9], gamma=1, gae_lambda=1, fragment_length=4)
    np.testing.assert_allclose(terminal, [-100])
    np.testing.assert_allclose(returns, [0])
    np.testing.assert_array_equal(completion_mask([2, 3, 4, 5], 4), [0, 0, 1, 1])
    with pytest.raises(ValueError):
        completion_mask([2, 4], 4)


def test_collection_freezes_encoder_dropout_and_has_one_policy(observation, game_data):
    model = tiny_model(game_data)
    before = policy_digest(model)

    def inspect():
        assert not any(module.training for module in model.modules())
        assert not any(p.requires_grad for p in model.parameters())

    envs = ScriptedEnvs(observation, inspect=inspect)
    config = cohort_config(cohort_games=4)
    episodes, counters = collect_complete_cohort(
        model, envs, config, config.cohort_seeds[:4], [0, 0], policy_version=42
    )
    assert [e.seed for e in episodes] == config.cohort_seeds[:4]
    assert set(envs.resets) == set(config.cohort_seeds[:4])
    assert counters == [4, 14]
    assert {e.policy_sha256 for e in episodes} == {before}
    assert {e.policy_version for e in episodes} == {42}
    assert policy_digest(model) == before
    assert model.training
    assert all(p.requires_grad for p in model.parameters())


@pytest.mark.parametrize("failure,limit", [("stall", 12), ("worker", 12), (None, 3)])
def test_failed_cohort_is_never_replaced(observation, game_data, failure, limit):
    model = tiny_model(game_data)
    envs = ScriptedEnvs(observation, failure=failure)
    config = cohort_config(cohort_max_episode_steps=limit)
    counters = [0, 0]
    before = policy_digest(model)
    with pytest.raises(CohortFailure) as error:
        collect_complete_cohort(model, envs, config, config.cohort_seeds[:2], counters, policy_version=0)
    assert counters == [0, 0]
    assert envs.resets == config.cohort_seeds[:2]
    assert policy_digest(model) == before
    assert "steps" in error.value.diagnostics


def test_frozen_policy_detects_buffer_mutation(game_data):
    model = tiny_model(game_data)
    with pytest.raises(RuntimeError, match="changed"), frozen_collection_policy(model):
        next(model.parameters()).add_(1.0)


def test_sampling_settings_are_frozen_and_saved_with_the_cohort(observation, game_data):
    model, episodes, _ = collected(observation, game_data)
    with pytest.raises(ValueError, match="sampling configuration"):
        CohortBuffer(episodes, cohort_config(rollout_temperature=0.8))
    with pytest.raises(RuntimeError, match="configuration changed"), frozen_collection_policy(model):
        model.config.hand_ar_mixture_eps = 0.1
    model.config.hand_ar_mixture_eps = 0.5
    config = cohort_config()
    envs = ScriptedEnvs(observation, inspect=lambda: setattr(config, "rollout_temperature", 0.8))
    with pytest.raises(CohortFailure, match="Sampling configuration changed"):
        collect_complete_cohort(model, envs, config, config.cohort_seeds[:2], [0, 0], policy_version=0)


def test_intermediate_rewards_cannot_cancel_to_hide_reward_drift(observation, game_data):
    class DriftingRewards(ScriptedEnvs):
        def step_training(self, actions):
            rows = super().step_training(actions)
            return {
                seed: (
                    obs,
                    reward + (1 if self.steps[seed] == 1 else -1 if terminal else 0),
                    terminal,
                    truncated,
                    info,
                    seconds,
                )
                for seed, (obs, reward, terminal, truncated, info, seconds) in rows.items()
            }

    config = cohort_config()
    with pytest.raises(CohortFailure, match="non-terminal-only reward"):
        collect_complete_cohort(
            tiny_model(game_data),
            DriftingRewards(observation),
            config,
            config.cohort_seeds[:2],
            [0, 0],
            policy_version=0,
        )


def test_collection_probabilities_match_training_path(observation, game_data):
    model, episodes, _ = collected(observation, game_data)
    buffer = CohortBuffer(episodes, cohort_config(cohort_training_rows=9), draw_indices=np.arange(9))
    batch = buffer.materialize(np.arange(9), torch.device("cpu"))
    model.eval()
    with torch.no_grad():
        dist, _ = _grammar_distribution(
            model, batch, temperature=_policy_temperature_for_scalars(batch["scalars"], cohort_config())
        )
    torch.testing.assert_close(dist.log_prob(batch["actions"]), batch["old_log_probs"], rtol=1e-5, atol=1e-6)
    assert torch.all(batch["action_mask"].gather(1, batch["actions"][:, None]) == 1)


def test_lazy_metadata_and_partial_batch_weights(observation, game_data, monkeypatch):
    _, episodes, _ = collected(observation, game_data)
    config = cohort_config(cohort_training_rows=9, outcome_label_mode="completion_window")
    buffer = CohortBuffer(episodes, config, draw_indices=np.arange(9))
    calls = []
    original = buffer.materialize

    def materialize(indices, *args):
        calls.append(len(indices))
        return original(indices, *args)

    monkeypatch.setattr(buffer, "materialize", materialize)
    batches = buffer.get_batches(5, torch.device("cpu"), micro_batch_size=2)
    weights, counts = _logical_mask_weights(batches, "terminal_outcome_mask", 3)
    assert calls == []
    assert len(batches) == 5
    assert sum(counts) == buffer.arrays["terminal_outcome_mask"].sum()
    assert sum(weights[:3]) == pytest.approx(1)
    assert sum(weights[3:]) == pytest.approx(1)
    assert [len(batch["actions"]) for batch in batches] == [2, 2, 1, 2, 2]
    assert max(calls) == 2
    buffer.arrays["terminal_outcome_mask"][:] = 0
    assert _logical_mask_weights(batches, "terminal_outcome_mask", 3) == ([0.0] * 5, [0.0] * 5)


def test_draws_have_fixed_exposure_and_arms_share_other_targets(observation, game_data):
    _, episodes, _ = collected(observation, game_data)
    config = cohort_config(cohort_training_rows=20)
    np.random.seed(19)
    full = CohortBuffer(episodes, config)
    np.random.seed(19)
    control = CohortBuffer(episodes, replace(config, outcome_label_mode="completion_window"))
    assert len(full.draw_indices) == 20
    assert full.diagnostics()["duplicate_draw_fraction"] > 0
    np.testing.assert_array_equal(full.draw_indices, control.draw_indices)
    for key in full.arrays:
        if key != "terminal_outcome_mask":
            np.testing.assert_array_equal(full.arrays[key], control.arrays[key])


def test_microbatch_updates_match_and_outcome_trains_shared_encoder(observation, game_data):
    model, episodes, _ = collected(observation, game_data)
    config = cohort_config(
        cohort_training_rows=9, mini_batch_size=5, micro_batch_size=2, ppo_epochs=1, value_loss_coeff=0, entropy_coeff=0
    )
    initial = copy.deepcopy(model.state_dict())
    trained = []
    for micro in (2, 5):
        current = copy.deepcopy(model)
        optimizer = torch.optim.Adam(current.parameters(), lr=1e-5)
        buffer = CohortBuffer(episodes, config, draw_indices=np.arange(9))
        buffer.arrays["advantages"][:] = 0  # Outcome-only gradient routing test.
        np.random.seed(11)
        stats = _run_ppo_update(
            current,
            optimizer,
            buffer,
            0.0,
            replace(config, micro_batch_size=micro),
            (5 + micro - 1) // micro,
            micro,
            torch.device("cpu"),
            False,
        )
        assert stats.ppo_samples_processed == 9
        assert len(stats.grad_norms) == 2
        trained.append(current.state_dict())
    for key in trained[0]:
        torch.testing.assert_close(trained[0][key], trained[1][key], atol=2e-6, rtol=2e-5)
    assert any(
        not torch.equal(value, trained[0][key]) for key, value in initial.items() if not key.startswith("value_head.")
    )


def test_cohort_artifact_preserves_all_observations_and_targets(observation, game_data, tmp_path):
    _, episodes, _ = collected(observation, game_data)
    config = cohort_config()
    buffer = CohortBuffer(episodes, config)
    save_cohort(tmp_path / "cohort", episodes, buffer, config)
    restored, draw, manifest = load_cohort(tmp_path / "cohort")
    assert manifest["teacher_labels"] is False
    other = CohortBuffer(restored, config, draw_indices=draw)
    for key in buffer.materialize(np.arange(9), "cpu"):
        torch.testing.assert_close(
            buffer.materialize(np.arange(9), "cpu")[key], other.materialize(np.arange(9), "cpu")[key], rtol=0, atol=0
        )
    for left, right in zip(episodes, restored, strict=True):
        for key in left.final_observation:
            np.testing.assert_array_equal(left.final_observation[key], right.final_observation[key])
    path = tmp_path / "cohort/manifest.json"
    for key, invalid in (("tokenizer_version", -1), ("tokenizer_semantics", "changed")):
        path.write_text(json.dumps({**manifest, key: invalid}))
        with pytest.raises(ValueError, match="observation schema"):
            load_cohort(tmp_path / "cohort")


@pytest.mark.parametrize("deck", ["b_blue", "b_red"])
@pytest.mark.parametrize("worker_count", [0, 2])
def test_training_worker_returns_real_reward_and_final_observation(game_data, deck, worker_count):
    kwargs = dict(
        data=game_data,
        vocab=build_vocab(game_data),
        deck_key=deck,
        reward_config=outcome_reward(),
        enable_teacher=False,
        max_steps=256,
    )
    env = BalatroEnv(seed=7000000, **kwargs)
    expected, _ = env.reset(seed=7000000)
    workers = EvaluationEnvs(worker_count, **kwargs)
    actual, _ = workers.reset(7000000)
    try:
        for _ in range(4):
            for key in expected:
                np.testing.assert_array_equal(actual[key], expected[key])
            action = int(np.flatnonzero(expected["action_mask"])[0])
            expected, reward, terminal, truncated, info = env.step(action)
            actual, other_reward, other_terminal, other_truncated, other_info, _ = workers.step_training(
                [(7000000, action)]
            )[7000000]
            assert (reward, terminal, truncated) == (other_reward, other_terminal, other_truncated)
            for key in info:
                assert other_info[key] == info[key]
            if terminal or truncated:
                break
    finally:
        workers.close()
        env.close()


@pytest.mark.parametrize(
    "change",
    [
        dict(collection_mode="unknown"),
        dict(cohort_games=3),
        dict(cohort_seeds=[1, 1]),
        dict(cohort_seeds=[190000]),
        dict(terminal_replay_updates_per_ppo_update=1),
        dict(return_path_coeff=0.01),
        dict(sil_coeff=0.1),
        dict(gamma=0.99),
        dict(adaptive_entropy=True),
        dict(total_updates=None),
        dict(cohort_max_episode_steps=0),
    ],
)
def test_invalid_cohort_configuration_fails_before_collection(change):
    with pytest.raises(ValueError):
        _validate_ppo_config(cohort_config(**change))


def test_legacy_defaults_are_unchanged():
    _validate_ppo_config(PPOConfig())
    assert PPOConfig().collection_mode == "fixed_steps"
    with pytest.raises(ValueError, match="require"):
        _validate_ppo_config(PPOConfig(outcome_label_mode="complete_episode"))
