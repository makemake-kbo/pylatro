"""Experiment integrity and the early-policy intervention's causal boundary."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict

import pytest
import test_complete_cohort as fixtures
import test_outcome_credit_branch as branch_fixtures
import torch

from pylatro_agent.checkpoint import load_checkpoint_payload
from pylatro_agent.constants import CURRENT_ANTE_SCALAR_INDEX
from pylatro_agent.horizon_transfer import EarlyAntePolicy
from pylatro_agent.training.cohort import CohortBuffer
from pylatro_agent.training.outcome_credit import _same_state
from pylatro_agent.training.ppo_observations import _obs_dicts_to_batch
from pylatro_agent.training.ppo_policy import _grammar_distribution
from tools.long_horizon_study import (
    ARMS,
    PRESERVED,
    branch_payload,
    comparison,
    load_protocol,
    recovery_payload,
    replacement_budget,
)

cpu_threads = fixtures.cpu_threads
game_data = fixtures.game_data
observation = fixtures.observation
parent = branch_fixtures.parent


def test_three_branches_preserve_learning_state_and_isolate_credit(parent, tmp_path):
    original = load_checkpoint_payload(parent, "cpu")
    before = copy.deepcopy(original)
    branches = {}
    for arm in ARMS:
        branches[arm] = branch_payload(
            original,
            arm=arm,
            directory=tmp_path / arm,
            seeds=[10000000, 10000001],
            reservations=[240000, 250000],
            sources={"fixture": "a" * 64},
            source_digest="b" * 64,
            parent=parent,
            parent_hash="c" * 64,
        )
        for key in PRESERVED:
            assert _same_state(original[key], branches[arm][key]), key
        assert branches[arm]["cohort_ppo_config"]["cohort_seeds"] == [7000000, 7000001, 10000000, 10000001]
        assert {240000, 250000} <= set(branches[arm]["training_reserved_seeds"])
    assert _same_state(before, original)
    # Apart from explicit run identifiers and destinations, MC differs from its
    # outcome-only GAE control in exactly the advantage estimator.
    a, b = (branches[arm]["cohort_ppo_config"] for arm in ("terminal_mc", "terminal_gae"))
    differences = {key for key in a if a[key] != b[key]}
    assert differences == {"advantage_estimator", "ppo_run_uuid", "ppo_recipe_id", "save_dir", "log_dir"}
    assert branches["terminal_mc"]["agent_config"] == branches["terminal_gae"]["agent_config"]
    assert branches["legacy_gae"]["cohort_ppo_config"]["return_value_mode"] == "terminal_plus_residual"
    assert a["return_value_mode"] == "terminal_only" and a["value_loss_coeff"] == 0


def test_branch_rejects_reuse_of_parent_training_games(parent, tmp_path):
    original = load_checkpoint_payload(parent, "cpu")
    with pytest.raises(ValueError, match="repeats consumed"):
        branch_payload(
            original,
            arm="terminal_mc",
            directory=tmp_path,
            seeds=[7000000, 10000001],
            reservations=[],
            sources={},
            source_digest="b" * 64,
            parent=parent,
            parent_hash="c" * 64,
        )


def test_logical_kl_recovery_preserves_prefix_and_records_old_source(parent, tmp_path):
    original = branch_payload(
        load_checkpoint_payload(parent, "cpu"), arm="terminal_mc", directory=tmp_path / "old",
        seeds=[10000000, 10000001], reservations=[240000], sources={"old": "a" * 64},
        source_digest="b" * 64, parent=parent, parent_hash="c" * 64,
    )
    before = copy.deepcopy(original)
    recipe = {**original["cohort_ppo_config"], "ppo_source_sha256": "d" * 64,
              "save_dir": str(tmp_path / "new/checkpoints"), "log_dir": str(tmp_path / "new/logs"),
              "ppo_run_uuid": "recovery-run", "ppo_recipe_id": "logical-kl"}
    recovered = recovery_payload(
        original, recipe=recipe, sources={"new": "e" * 64}, source_digest="d" * 64,
        previous_path=tmp_path / "old/initial.pt", previous_hash="f" * 64,
    )
    for key in PRESERVED:
        assert _same_state(recovered[key], original[key]), key
    assert _same_state(before, original)
    history = recovered["cohort_branch"]["recovered_prefix"]
    assert history["source_sha256"] == "b" * 64
    assert history["sha256"] == "f" * 64
    assert recovered["ppo_run_provenance"]["source_sha256"] == "d" * 64
    for field, value in [("lr", 1.0), ("target_kl_max", .8), ("cohort_seeds", [200, 201])]:
        with pytest.raises(ValueError, match="learning recipe"):
            recovery_payload(
                original, recipe={**recipe, field: value}, sources={}, source_digest="d" * 64,
                previous_path=parent, previous_hash="f" * 64,
            )


def test_recovery_protocol_rejects_changed_inherited_evidence(monkeypatch, tmp_path):
    from pylatro_agent.transfer import sha256
    from tools import long_horizon_study as study

    evidence = tmp_path / "prefix.json"
    evidence.write_text("original")
    protocol = {"source_hashes": {}, "parents": {},
                "recovery": {"evidence": {str(evidence): sha256(evidence)}}}
    (tmp_path / "manifest.json").write_text(json.dumps(protocol))
    monkeypatch.setattr(study, "source_hashes", lambda: {})
    assert load_protocol(tmp_path) == protocol
    evidence.write_text("changed")
    with pytest.raises(ValueError, match="Recovery evidence changed"):
        load_protocol(tmp_path)


def test_physical_batch_revision_preserves_learning_state_and_logical_exposure(parent, tmp_path):
    original = load_checkpoint_payload(parent, "cpu")
    payload = branch_payload(
        original, arm="terminal_mc", directory=tmp_path / "smaller_batch",
        seeds=[310000000, 310000001], reservations=[270000, 280000],
        sources={"fixture": "a" * 64}, source_digest="b" * 64, parent=parent, parent_hash="c" * 64,
        micro_batch_size=1,
    )
    for key in PRESERVED:
        assert _same_state(original[key], payload[key]), key
    recipe = payload["cohort_ppo_config"]
    assert recipe["micro_batch_size"] == 1
    for key in ("mini_batch_size", "cohort_training_rows", "ppo_epochs", "lr"):
        assert recipe[key] == original["cohort_ppo_config"][key]
    assert payload["cohort_branch"]["overrides"]["micro_batch_size"] == {"from": 2, "to": 1}


def test_replacement_carries_spent_compute_and_rejects_a_live_job(tmp_path):
    (tmp_path / "manifest.json").write_text("{}")
    ledger = {
        "limit_seconds": 1000, "elapsed_seconds": 200,
        "jobs": [{"status": "complete", "elapsed_seconds": 200}],
    }
    (tmp_path / "compute.json").write_text(json.dumps(ledger))
    limit, evidence = replacement_budget(tmp_path)
    assert limit == 800
    assert evidence["spent_seconds"] == 200
    ledger["jobs"][0]["status"] = "running"
    (tmp_path / "compute.json").write_text(json.dumps(ledger))
    with pytest.raises(ValueError, match="running"):
        replacement_budget(tmp_path)


def test_early_transfer_uses_only_declared_policy_and_does_not_draw_rng(game_data, observation):
    early = fixtures.tiny_model(game_data).eval()
    late = copy.deepcopy(early).eval()
    with torch.no_grad():
        late.action_grammar_head.macro_head.bias.add_(torch.arange(len(late.action_grammar_head.macro_head.bias)))
    hybrid = EarlyAntePolicy(early, late, through_ante=3).eval()
    observations = [copy.deepcopy(observation) for _ in range(4)]
    for obs, ante in zip(observations, (1, 3, 4, 8), strict=True):
        obs["scalars"][CURRENT_ANTE_SCALAR_INDEX] = ante
    batch = _obs_dicts_to_batch(observations, torch.device("cpu"))
    state = torch.get_rng_state().clone()
    with torch.inference_mode():
        actual, _ = _grammar_distribution(hybrid, batch, temperature=0.8)
        left, _ = _grammar_distribution(early, batch, temperature=0.8)
        right, _ = _grammar_distribution(late, batch, temperature=0.8)
    assert torch.equal(state, torch.get_rng_state())
    for key in asdict(actual.output):
        value = getattr(actual.output, key)
        if value is not None:
            torch.testing.assert_close(value[:2], getattr(left.output, key)[:2], rtol=2e-5, atol=2e-6)
            torch.testing.assert_close(value[2:], getattr(right.output, key)[2:], rtol=2e-5, atol=2e-6)
    assert hybrid.route_counts == {"early": 2, "continuation": 2}


def test_high_training_performance_cannot_pass_a_negative_heldout_comparison():
    outcomes = {
        arm: {
            r: {
                mode: [
                    {"seed": s, "sampling_seed": 0 if mode == "sampled" else None, "won": arm == "control"}
                    for s in range(20)
                ]
                for mode in ("sampled", "greedy")
            }
            for r in (0, 1)
        }
        for arm in ("candidate", "control")
    }
    result = comparison(outcomes, "candidate", "control")
    assert not result["passes_gameplay_gate"]
    assert result["combined"]["sampled"]["paired_ci95"] == [-1.0, -1.0]


def test_long_horizon_telemetry_distinguishes_direct_credit(observation):
    from pylatro_agent.training.cohort import CohortEpisode, ObservationStore, sampling_configuration

    config = fixtures.cohort_config(cohort_training_rows=8)
    store = ObservationStore()
    for _ in range(160):
        store.append(observation)
    episode = CohortEpisode(123, 0, 0, "fixed", observations=store)
    episode.actions = [0] * 160
    episode.rewards = [0.0] * 159 + [10.0]
    episode.values = [3.0] * 160
    episode.log_probs = [0.0] * 160
    episode.slot_steps = list(range(160))
    episode.outcome = {"terminated": True, "won": True, "final_ante": 8}
    episode.sampling_configuration = sampling_configuration(config)
    fragments = CohortBuffer([episode], config, draw_indices=list(range(8))).diagnostics()["horizon_credit"]
    config.advantage_estimator = "complete_episode_mc"
    mc = CohortBuffer([episode], config, draw_indices=list(range(8))).diagnostics()["horizon_credit"]
    assert fragments["100_plus"]["mean_direct_terminal_weight"] == 0
    assert mc["100_plus"]["mean_direct_terminal_weight"] == 1
    assert mc["100_plus"]["return_target_mse_to_completed_return"] == 0
    assert mc["100_plus"]["distinct_optimizer_rows"] == 8
