"""Study provenance, fixed-data diagnosis, and paired-statistics regressions."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict

import pytest
import test_complete_cohort as fixtures
import torch
from test_complete_cohort import cohort_config, collected, tiny_model

from pylatro_agent import cohort_study
from pylatro_agent.checkpoint import load_checkpoint_payload
from pylatro_agent.cohort_study import (
    audit_seed_panels,
    branch_checkpoint,
    compare_arms,
    critic_summary,
    validate_panel,
)
from pylatro_agent.training.cohort import CohortBuffer
from pylatro_agent.training.cohort_config import validate_cohort_resume
from pylatro_agent.training.cohort_io import save_cohort
from pylatro_agent.training.ppo_checkpoint import _save_checkpoint
from pylatro_agent.training.ppo_evaluation import evaluate_model
from pylatro_agent.transfer import sha256
from pylatro_agent.vocab import build_vocab
from tools import study_complete_cohort as study_cli
from tools.study_complete_cohort import assign_budget, diagnose_labels
from tools.train_outcome_ppo import same_state_tree

cpu_threads = fixtures.cpu_threads
game_data = fixtures.game_data
observation = fixtures.observation


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def history(repo, *, used_seed=41):
    prior = repo / "eval/prior"
    source = repo / "data/seeds.json"
    write_json(source, {"seeds": [used_seed]})
    write_json(
        prior / "manifest.json",
        {
            "seed_separation": [{"path": str(source), "sha256": sha256(source)}],
            "development_seeds": [100],
            "confirmation_seeds": [200],
            "critic_training_seeds": [300],
        },
    )
    (prior / "training_episodes.jsonl").write_text(json.dumps({"seed": 500}) + "\n")
    return prior


def test_seed_audit_checks_manifests_actual_training_and_previous_evaluation(tmp_path):
    prior = history(tmp_path)
    training = {"0": [700], "1": [800]}
    result = audit_seed_panels(tmp_path, prior, training, [100], [900])
    assert len(result["training_manifests"]) == 2
    assert result["historical_inventory"]
    for panel in ([41], [500], [300], [200]):
        with pytest.raises(ValueError):
            audit_seed_panels(tmp_path, prior, training, [100], panel)
    with pytest.raises(ValueError, match="replicas overlap"):
        audit_seed_panels(tmp_path, prior, {"0": [700], "1": [700]}, [100], [900])
    with pytest.raises(ValueError):
        audit_seed_panels(tmp_path, prior, {"0": [200], "1": [800]}, [100], [900])


def test_changed_manifest_cannot_certify_seed_separation(tmp_path):
    prior = history(tmp_path)
    write_json(tmp_path / "data/seeds.json", {"seeds": [900]})
    with pytest.raises(ValueError, match="changed"):
        audit_seed_panels(tmp_path, prior, {"0": [700], "1": [800]}, [100], [900])


def test_diagnostic_exclusions_are_not_mistaken_for_training_data(tmp_path):
    prior = history(tmp_path)
    panel = prior / "development_seeds.json"
    write_json(panel, [100])
    write_json(
        prior / "diagnostic/manifest.json",
        {
            "seed_separation": [{"path": str(panel), "sha256": sha256(panel)}],
            "seeds": [600],
        },
    )
    result = audit_seed_panels(tmp_path, prior, {"0": [700], "1": [800]}, [100], [900])
    assert 100 in result["historical_reserved_seeds"]
    assert str(panel) not in {item["path"] for item in result["training_manifests"]}


def test_cohort_branch_preserves_model_adam_rng_and_parent(game_data, tmp_path):
    model = tiny_model(game_data)
    config = cohort_config()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    sum(p.sum() for p in model.parameters()).backward()
    optimizer.step()
    source = _save_checkpoint(
        model=model,
        optimizer=optimizer,
        save_path=tmp_path,
        update_count=42,
        total_steps=8192 * 42,
        planned_updates=42,
        entropy_coeff=config.entropy_coeff,
        entropy_signal_ema=None,
        lr=config.lr,
        agent_config=model.config,
        config=config,
    )
    before_hash = sha256(source)
    destination = tmp_path / "branch.pt"
    branch_checkpoint(source, destination, config, parent_sha256=before_hash)
    old, new = load_checkpoint_payload(source, "cpu"), load_checkpoint_payload(destination, "cpu")
    for key in ("state_dict", "optimizer_state_dict", "rng_states", "total_steps", "update_count"):
        assert same_state_tree(old[key], new[key]), key
    assert sha256(source) == before_hash
    assert new["cohort_state"]["initial_update_count"] == 42
    assert new["cohort_state"]["seed_cursor"] == 0
    validate_cohort_resume(config, new)
    with pytest.raises(FileExistsError):
        branch_checkpoint(source, destination, config, parent_sha256=before_hash)


def test_accelerator_budget_starts_unassigned_and_cannot_be_reset(tmp_path):
    write_json(tmp_path / "compute.json", {"limit_seconds": 0, "jobs": [], "elapsed_seconds": 0})
    for hours in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            assign_budget(tmp_path, hours)
    assign_budget(tmp_path, 1)
    assert json.loads((tmp_path / "compute.json").read_text())["limit_seconds"] == 3600
    with pytest.raises(ValueError, match="already"):
        assign_budget(tmp_path, 1)


def make_outcomes(seeds):
    result = {}
    for arm in ("completion_window", "complete_episode"):
        result[arm] = {}
        for replica in (0, 1):
            result[arm][replica] = {}
            for mode, repeats in (("sampled", [0, 1, 2]), ("greedy", [None])):
                result[arm][replica][mode] = [
                    {
                        "seed": seed,
                        "sampling_seed": repeat,
                        "won": arm == "complete_episode",
                        "final_ante": 9 if arm == "complete_episode" else 1,
                        "max_ante": 9 if arm == "complete_episode" else 1,
                        "stalled": False,
                        "censored": False,
                        "cleared_antes": list(range(1, 9)) if arm == "complete_episode" else [],
                    }
                    for seed in seeds
                    for repeat in repeats
                ]
    return result


def test_paired_statistics_cluster_repeats_and_replicas_by_game_seed():
    seeds = [1, 2, 3]
    report = compare_arms(make_outcomes(seeds), seeds, confirmation=True)
    sampled = report["combined"]["sampled"]
    assert sampled["paired"]["game_seeds"] == 3
    assert sampled["paired"]["games_per_arm"] == 18
    assert sampled["paired"]["paired_ci95"] == [1.0, 1.0]
    assert report["verdict"] == "validated"
    assert compare_arms(make_outcomes(seeds), seeds)["verdict"] == "promising"
    changed = make_outcomes(seeds)
    for row in changed["complete_episode"][1]["sampled"]:
        row["won"] = False
    report = compare_arms(changed, seeds, confirmation=True)
    assert report["combined"]["sampled"]["paired"]["sampled_or_greedy_delta"] == 0.5
    assert not report["acceptance"]["both_replicas_positive"]
    assert report["verdict"] != "validated"


def test_duplicate_missing_or_wrong_repeats_fail():
    seeds = [1, 2, 3]
    for mutation in ("missing", "duplicate", "repeat"):
        outcomes = make_outcomes(seeds)
        rows = outcomes["complete_episode"][0]["sampled"]
        if mutation == "missing":
            rows.pop()
        elif mutation == "duplicate":
            rows.append(rows[0])
        else:
            rows[0]["sampling_seed"] = 4
        with pytest.raises(ValueError, match="comparison"):
            compare_arms(outcomes, seeds)
    with pytest.raises(ValueError):
        validate_panel([], [1], [0])


def test_forecast_denominators_distinguish_games_and_decisions():
    p = [0.5, 0, 0, 0, 0, 0, 0, 0, 0.5]
    rows = [
        dict(won=True, final_ante=9, critic_forecasts=[dict(outcome_probabilities=p, expected_return=4.0)]),
        dict(won=False, final_ante=1, critic_forecasts=[dict(outcome_probabilities=p, expected_return=0.0)] * 3),
    ]
    result = critic_summary(rows)
    assert result["games"] == 2 and result["decisions"] == 4
    assert result["game_weighted"]["return_mse"] == 18
    assert result["decision_weighted"]["return_mse"] == 9
    assert result["start"]["predicted_win_rate"] == 0.5
    assert result["start"]["observed_win_rate"] == 0.5


def test_return_forecast_logging_is_opt_in_and_preserves_policy_rng(game_data):
    model = tiny_model(game_data)
    results = []
    rng = torch.get_rng_state().clone()
    evaluate_model(
        model,
        game_data,
        build_vocab(game_data),
        1,
        torch.device("cpu"),
        seeds=[190000],
        max_no_progress_steps=3,
        eval_batch_size=1,
        reward_config=cohort_config().reward_config,
        greedy=False,
        sampling_seed=0,
        outcomes_out=results,
        record_return_forecasts=True,
    )
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    assert "expected_return" in results[0]["critic_forecasts"][0]
    before = results[0]
    results = []
    evaluate_model(
        model,
        game_data,
        build_vocab(game_data),
        1,
        torch.device("cpu"),
        seeds=[190000],
        max_no_progress_steps=3,
        eval_batch_size=1,
        reward_config=cohort_config().reward_config,
        greedy=False,
        sampling_seed=0,
        outcomes_out=results,
    )
    assert "expected_return" not in results[0]["critic_forecasts"][0]
    assert before["action_counts"] == results[0]["action_counts"]
    assert before["won"] == results[0]["won"]


def test_fixed_data_diagnostic_uses_same_rows_and_no_actor_updates(observation, game_data, tmp_path):
    config = cohort_config(cohort_games=4)
    model, episodes, _ = collected(observation, game_data, config=config)
    buffer = CohortBuffer(episodes, config)
    save_cohort(tmp_path / "cohort", episodes, buffer, config)
    checkpoint = _save_checkpoint(
        model=model,
        optimizer=torch.optim.Adam(model.parameters()),
        save_path=tmp_path,
        update_count=0,
        total_steps=0,
        planned_updates=1,
        entropy_coeff=0.001,
        entropy_signal_ema=None,
        lr=1e-5,
        agent_config=model.config,
        config=config,
    )
    original_hash = sha256(checkpoint)
    results = diagnose_labels(tmp_path / "cohort", checkpoint, tmp_path / "diagnostic", updates=2)
    assert sha256(checkpoint) == original_hash
    assert not set(results["training_seeds"]) & set(results["validation_seeds"])
    assert not results["actor_updated"] and not results["encoder_updated"]
    for replica in results["replicas"].values():
        arms = replica["arms"]
        assert arms["completion"]["available_labels"] == arms["random"]["available_labels"]
        assert {arm["physical_rows"] for arm in arms.values()} == {640}


@pytest.fixture
def prepared_study(tmp_path, monkeypatch, game_data):
    repo = tmp_path / "repo"
    prior = history(repo)
    model = tiny_model(game_data)
    selected = {"recipe": "gae097", "updates": 42, "checkpoints": {}, "checkpoint_hashes": {}}
    for replica in (0, 1):
        config = cohort_config(
            collection_mode="fixed_steps",
            outcome_label_mode="completion_window",
            seed=replica,
            num_envs=32,
            rollout_length=256,
            lr=1e-5,
            ppo_epochs=2,
            mini_batch_size=640,
            micro_batch_size=320,
            entropy_coeff=0.001,
            stake=1,
            win_ante=8,
            device="mps",
        )
        folder = prior / "trials" / f"gae097_seed{replica}"
        folder.mkdir(parents=True)
        checkpoint = _save_checkpoint(
            model=model,
            optimizer=torch.optim.Adam(model.parameters()),
            save_path=folder,
            update_count=42,
            total_steps=42 * 8192,
            planned_updates=42,
            entropy_coeff=0.001,
            entropy_signal_ema=None,
            lr=1e-5,
            agent_config=model.config,
            config=config,
        )
        write_json(folder / "recipe.json", {"ppo_config": asdict(config)})
        selected["checkpoints"][f"seed{replica}"] = str(checkpoint)
        selected["checkpoint_hashes"][f"seed{replica}"] = sha256(checkpoint)
    write_json(prior / "candidate_selection.json", selected)
    monkeypatch.setattr(cohort_study, "implementation_hashes", lambda: {"fixture": "a" * 64})
    destination = tmp_path / "study"
    protocol = cohort_study.prepare_study(destination, prior=prior, repo=repo)
    return destination, protocol, prior, repo


def test_prepare_study_freezes_recipe_without_starting_training(prepared_study):
    destination, protocol, prior, repo = prepared_study
    assert protocol["status"] == "prepared"
    assert len(protocol["recipes"]) == 4
    assert json.loads((destination / "compute.json").read_text())["jobs"] == []
    assert json.loads((destination / "compute.json").read_text())["limit_seconds"] == 0
    assert not list(destination.rglob("*.pt"))
    assert all(
        recipe["config"]["terminal_replay_updates_per_ppo_update"] == 0 for recipe in protocol["recipes"].values()
    )
    for arm in ("completion_window", "complete_episode"):
        config = protocol["recipes"][f"{arm}_seed0"]["config"]
        assert config["cohort_seeds"] == list(range(7000000, 7001280))
    with pytest.raises(FileExistsError):
        cohort_study.prepare_study(destination, prior=prior, repo=repo)


def test_unassigned_budget_never_starts_accelerator_training(prepared_study, monkeypatch):
    root, protocol, *_ = prepared_study
    write_json(
        root / "cpu_verification/manifest.json",
        {
            "status": "passed",
            "source_hashes": protocol["source_hashes"],
        },
    )

    def unexpected_training(*args, **kwargs):
        pytest.fail("An unassigned budget must not launch training")

    monkeypatch.setattr(study_cli, "train_ppo", unexpected_training)
    with pytest.raises(TimeoutError, match="remaining 0"):
        study_cli.train_arm(root, protocol, "complete_episode", 0, 10, device="mps", estimate_seconds=1)
    assert json.loads((root / "compute.json").read_text())["jobs"] == []
    assert not list(root.rglob("*.pt"))


def test_harness_evaluation_report_freeze_and_confirmation_gates(prepared_study, monkeypatch):
    root, protocol, *_ = prepared_study
    # Small panels and mocked games exercise orchestration, not policy strength.
    protocol["development_seeds"] = [190000, 190001, 190002]
    protocol["confirmation_seeds"] = [210000, 210001, 210002]
    for arm in cohort_study.ARMS:
        for replica in cohort_study.REPLICAS:
            config = cohort_study.config_from_dict(protocol["recipes"][f"{arm}_seed{replica}"]["config"])
            parent = protocol["parents"][str(replica)]
            initial = root / f"{arm}_seed{replica}/initial.pt"
            branch_checkpoint(parent["path"], initial, config, parent_sha256=parent["sha256"])
            for endpoint in (10, 20):
                payload = load_checkpoint_payload(initial, "cpu")
                payload["update_count"] = 42 + endpoint
                payload["cohort_state"].update(
                    seed_cursor=endpoint * 64,
                    optimizer_examples=endpoint * 8192 * 2,
                )
                path = study_cli.checkpoint_for(root, protocol, arm, replica, endpoint)
                path.parent.mkdir(exist_ok=True)
                torch.save(payload, path)
                study_cli.validate_endpoint_payload(payload, protocol, arm, replica, endpoint)
                invalid = copy.deepcopy(payload)
                invalid["cohort_state"]["optimizer_examples"] -= 1
                with pytest.raises(ValueError, match="exposure"):
                    study_cli.validate_endpoint_payload(invalid, protocol, arm, replica, endpoint)

    calls = []

    def fake_evaluate(model, reward, data, vocab, seeds, device, output, **kwargs):
        assert str(device) == "cpu" and kwargs["record_return_forecasts"]
        calls.append(output)
        won = output.parent.name.startswith("complete_episode")
        rows = [
            {
                "seed": seed,
                "sampling_seed": None if kwargs["greedy"] else kwargs["sampling_seed"],
                "won": won,
                "final_ante": 9 if won else 1,
                "max_ante": 9 if won else 1,
                "cleared_antes": list(range(1, 9)) if won else [],
                "censored": False,
                "stalled": False,
            }
            for seed in seeds
        ]
        write_json(output / "manifest.json", {"status": "complete", **kwargs["metadata"]})
        (output / "outcomes.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))

    monkeypatch.setattr(study_cli, "evaluate_artifact", fake_evaluate)
    for endpoint in (10, 20):
        result = study_cli.evaluate_endpoint(root, protocol, endpoint)
        assert result["verdict"] == "promising"
        assert result["combined"]["sampled"]["complete_episode"]["games"] == 18
    assert len(calls) == 32
    # Reusing verified output does not execute the games again.
    study_cli.evaluate_endpoint(root, protocol, 10)
    assert len(calls) == 32
    path = calls[0] / "outcomes.jsonl"
    original = path.read_text()
    path.write_text(original + original.splitlines()[0] + "\n")
    with pytest.raises(ValueError, match="comparison"):
        study_cli.report_endpoint(root, protocol, 10)
    path.write_text(original)
    with pytest.raises(FileNotFoundError):
        study_cli.evaluate_endpoint(root, protocol, 20, stage="confirmation")
    assert not (root / "evaluations/confirmation").exists()
    selection = study_cli.freeze_selection(root, protocol)
    assert selection["endpoint"] == 20  # Predefined tie break, not a new choice.
    with pytest.raises(FileExistsError):
        study_cli.freeze_selection(root, protocol)
    with pytest.raises(ValueError, match="assigned study budget"):
        study_cli.evaluate_endpoint(root, protocol, 20, stage="confirmation")
    assign_budget(root, 1)
    confirmed = study_cli.evaluate_endpoint(root, protocol, 20, stage="confirmation")
    assert confirmed["verdict"] == "validated"  # Deliberately simulated wins only.
    write_json(root / "complete_episode_seed0/logs/failure.json", {"error": "simulated failure"})
    assert study_cli.report_endpoint(root, protocol, 20, stage="confirmation")["verdict"] == "inconclusive"
    checkpoint = study_cli.checkpoint_for(root, protocol, "complete_episode", 0, 20)
    checkpoint.write_bytes(b"changed after selection")
    with pytest.raises(ValueError, match="frozen selection"):
        study_cli.evaluate_endpoint(root, protocol, 20, stage="confirmation")
