"""Pinned protocols, provenance and statistics for the complete-cohort study."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import uuid
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np
import torch

from .checkpoint import load_checkpoint_payload
from .learning_bottleneck import atomic_json, read_json
from .reward import RewardConfig
from .training.cohort_training import initial_cohort_state
from .training.ppo_config import PPOConfig, _ppo_run_provenance, _validate_ppo_config
from .training.ppo_evaluation import training_seed_reservations
from .transfer import (
    check_seed_separation,
    manifest_seeds,
    paired_statistics,
    restore_configuration,
    sha256,
    source_hashes,
    summarize,
)

REPO = Path(__file__).resolve().parents[2]
PRIOR_ROOT = REPO / "eval/outcome_ppo_20260913"
ARMS = ("completion_window", "complete_episode")
REPLICAS = (0, 1)
ENDPOINTS = (0, 10, 20)
DEVELOPMENT = list(range(190000, 190200))
CONFIRMATION = list(range(210000, 210400))


def config_from_dict(values):
    values = copy.deepcopy(values)
    if isinstance(values.get("reward_config"), dict):
        values["reward_config"] = RewardConfig(**values["reward_config"])
    return PPOConfig(**values)


def implementation_hashes():
    hashes = source_hashes()
    for path in [REPO / "docs/transfer/complete_cohort.md", *sorted((REPO / "tests").glob("test_complete_cohort*.py"))]:
        hashes[str(path.relative_to(REPO))] = sha256(path)
    return hashes


def verify_sources(protocol):
    active = implementation_hashes()
    if active != protocol["source_hashes"]:
        changed = sorted(
            key
            for key in set(active) | set(protocol["source_hashes"])
            if active.get(key) != protocol["source_hashes"].get(key)
        )
        raise ValueError(f"Study source changed after preparation: {changed[:10]}")


def _seed_lists(value, *, training):
    """Only explicit lists confer evidence; never infer seeds from numeric ranges."""
    found = set()
    if isinstance(value, dict):
        for key, item in value.items():
            is_list = isinstance(item, list) and item and all(type(s) is int for s in item)
            if is_list and "seed" in key:
                if not training or key in (
                    "training_seeds",
                    "critic_training_seeds",
                    "correction_seeds",
                    "cohort_seeds",
                ):
                    found.update(item)
            elif isinstance(item, (dict, list)):
                found.update(_seed_lists(item, training=training))
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, (dict, list)):
                found.update(_seed_lists(item, training=training))
    return found


def audit_seed_panels(repo, prior, training, development, confirmation):
    """Audit inherited manifests plus actual PPO game logs and reserved panels."""
    repo, prior = Path(repo), Path(prior)
    all_training = [s for seeds in training.values() for s in seeds]
    if len(set(all_training)) != len(all_training):
        raise ValueError("Training replicas overlap")
    if any(len(panel) != len(set(panel)) or not panel for panel in (all_training, development, confirmation)):
        raise ValueError("Seed panels must be nonempty and distinct")
    if set(development) & set(confirmation) or set(all_training) & set(development + confirmation):
        raise ValueError("Study panels overlap")
    manifests = set()
    historical_reserved, historical_training = set(), set()
    inventory = []
    # Existing studies explicitly retain the source-data seed manifests. Also
    # inspect evaluation manifests so a previously inspected panel cannot be
    # silently relabeled as new confirmation.
    for path in sorted((repo / "eval").rglob("manifest.json")):
        value = read_json(path)
        if not isinstance(value, dict):
            continue
        for item in value.get("seed_separation", []):
            source = Path(item["path"])
            source = source if source.is_absolute() else repo / source
            if not source.is_file() or sha256(source) != item["sha256"]:
                raise ValueError(f"Historical seed manifest missing or changed: {source}")
            # Nested diagnostic manifests list *all exclusion panels* here,
            # including prior development/confirmation. Their key name alone
            # does not establish that those games trained the parent model.
            if path.parent.parent == repo / "eval" or path == prior / "manifest.json":
                manifests.add(source)
            else:
                historical_reserved.update(manifest_seeds(source))
                inventory.append({"path": str(source), "sha256": item["sha256"], "role": "diagnostic exclusion panel"})
        explicit = _seed_lists(value, training=False)
        if explicit:
            historical_reserved.update(explicit)
            historical_training.update(_seed_lists(value, training=True))
            inventory.append({"path": str(path), "sha256": sha256(path)})
    if not (prior / "manifest.json").is_file() or not manifests:
        raise ValueError("Prior study and explicit source training manifests are required")
    for path in sorted((repo / "eval").rglob("training_episodes.jsonl")):
        manifests.add(path)
    for path in sorted((repo / "eval").rglob("training_seeds.json")):
        manifests.add(path)
    evidence = check_seed_separation(all_training + development + confirmation, sorted(manifests))
    if set(development + confirmation) & historical_training:
        raise ValueError("Evaluation panel occurs in an explicit historical training seed list")
    if set(confirmation) & historical_reserved:
        raise ValueError("Confirmation seeds were already used or reserved in an earlier study")
    if set(all_training) & (historical_reserved | historical_training):
        raise ValueError("New training seeds overlap historical study panels")
    return {
        "training_manifests": evidence,
        "historical_inventory": inventory,
        "historical_reserved_seeds": sorted(historical_reserved),
        "method": "Explicit source seed lists, recorded PPO episodes, and available study/evaluation manifests",
    }


def prepare_study(root, *, prior=PRIOR_ROOT, repo=REPO):
    root, prior = Path(root).resolve(), Path(prior).resolve()
    if root.exists():
        raise FileExistsError(root)
    selection = read_json(prior / "candidate_selection.json")
    if selection["recipe"] != "gae097" or selection["updates"] != 42:
        raise ValueError("Expected the frozen gae097 update-42 initialization")
    training = {
        str(replica): list(range(7000000 + replica * 100000, 7000000 + replica * 100000 + 1280)) for replica in REPLICAS
    }
    audit = audit_seed_panels(repo, prior, training, DEVELOPMENT, CONFIRMATION)
    sources = implementation_hashes()
    source_digest = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    recipes, parents = {}, {}
    for replica in REPLICAS:
        path = Path(selection["checkpoints"][f"seed{replica}"])
        path = path if path.is_absolute() else Path(repo) / path
        if sha256(path) != selection["checkpoint_hashes"][f"seed{replica}"]:
            raise ValueError("Selected initialization checksum mismatch")
        payload = load_checkpoint_payload(path, "cpu")
        architecture, reward = restore_configuration(payload)
        parent_recipe_path = prior / "trials" / f"gae097_seed{replica}" / "recipe.json"
        parent_values = read_json(parent_recipe_path)["ppo_config"]
        unknown = set(parent_values) - {f.name for f in fields(PPOConfig)}
        if unknown:
            raise ValueError(f"Unknown parent PPO fields: {unknown}")
        if payload["update_count"] != 42 or architecture.hand_ar_mixture_eps != 0.5:
            raise ValueError("Parent update or mixture differs from the selected baseline")
        parents[str(replica)] = {
            "path": str(path),
            "sha256": sha256(path),
            "recipe_path": str(parent_recipe_path),
            "recipe_sha256": sha256(parent_recipe_path),
        }
        for arm in ARMS:
            directory = root / f"{arm}_seed{replica}"
            config = config_from_dict(parent_values)
            changes = {
                "collection_mode": "complete_cohort",
                "outcome_label_mode": arm,
                "cohort_games": 64,
                "cohort_training_rows": 8192,
                "cohort_max_episode_steps": 2048,
                "cohort_workers": 4,
                "cohort_seeds": training[str(replica)],
                "total_updates": 62,
                "terminal_replay_updates_per_ppo_update": 0,
                "eval_seeds": DEVELOPMENT,
                "eval_games": 200,
                "eval_sampled_games": 200,
                "training_excluded_seeds": sorted(
                    set(audit["historical_reserved_seeds"] + CONFIRMATION) - set(DEVELOPMENT)
                ),
                "eval_before_training": False,
                "eval_interval": 100000,
                "eval_device": "cpu",
                "save_dir": str(directory / "checkpoints"),
                "log_dir": str(directory / "logs"),
                "stop_request_path": str(directory / "STOP"),
                "checkpoint_interval": 1,
                "ppo_run_uuid": str(uuid.uuid4()),
                "ppo_source_sha256": source_digest,
                "ppo_recipe_id": f"complete_cohort_v1_{arm}",
                "reward_config": reward,
            }
            for key, value in changes.items():
                setattr(config, key, value)
            _validate_ppo_config(config)
            expected = {
                "gamma": 1.0,
                "gae_lambda": 0.97,
                "lr": 1e-5,
                "ppo_epochs": 2,
                "mini_batch_size": 640,
                "micro_batch_size": 320,
                "hand_ar_mixture_eps": 0.5,
                "rollout_temperature": 1.0,
                "entropy_coeff": 0.001,
                "deck_key": "b_blue",
                "stake": 1,
                "win_ante": 8,
                "precision": "fp32",
                "num_envs": 32,
                "rollout_length": 256,
            }
            if any(getattr(config, k) != v for k, v in expected.items()):
                raise ValueError("Parent recipe differs from the predefined pilot")
            active = asdict(config)
            recipes[f"{arm}_seed{replica}"] = {
                "config": active,
                "overrides": {
                    k: {"from": parent_values.get(k), "to": v} for k, v in active.items() if parent_values.get(k) != v
                },
            }
    protocol = {
        "version": 1,
        "status": "prepared",
        "source_hashes": sources,
        "source_sha256": source_digest,
        "parents": parents,
        "recipes": recipes,
        "training_seeds": training,
        "development_seeds": DEVELOPMENT,
        "confirmation_seeds": CONFIRMATION,
        "sampling_seeds": [0, 1, 2],
        "training_replicates": [0, 1],
        "endpoints": list(ENDPOINTS),
        "seed_audit": audit,
        "teacher_guidance": False,
        "initialization_selection": {
            "path": str(prior / "candidate_selection.json"),
            "sha256": sha256(prior / "candidate_selection.json"),
        },
        "interpretation": "Targets evaluate the frozen collection policy. Gameplay benefit remains unproven.",
    }
    root.mkdir(parents=True)
    atomic_json(root / "manifest.json", protocol)
    atomic_json(
        root / "compute.json",
        {"limit_seconds": 0, "elapsed_seconds": 0, "jobs": [], "authorization": "No new accelerator budget assigned"},
    )
    return protocol


def branch_checkpoint(parent, destination, config, *, parent_sha256):
    _validate_ppo_config(config)
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    if sha256(parent) != parent_sha256:
        raise ValueError("Parent checkpoint checksum mismatch")
    original = load_checkpoint_payload(parent, "cpu")
    architecture, reward = restore_configuration(original)
    if asdict(reward) != asdict(config.reward_config) or architecture.hand_ar_mixture_eps != config.hand_ar_mixture_eps:
        raise ValueError("Cohort branch cannot change reward or policy configuration")
    payload = dict(original)  # Tensor storage and Adam/RNG contents are untouched.
    payload["ppo_config_fields"] = {**original["ppo_config_fields"], **asdict(config)}
    payload["cohort_ppo_config"] = asdict(config)
    payload["cohort_state"] = initial_cohort_state(config, original["update_count"])
    payload["ppo_run_provenance"] = _ppo_run_provenance(config)
    payload["planned_updates"] = config.total_updates
    payload["training_reserved_seeds"] = training_seed_reservations(config)
    payload["cohort_branch"] = {
        "parent": str(parent),
        "parent_sha256": parent_sha256,
        "collection_version": 1,
        "overrides": {
            key: {"from": original["ppo_config_fields"].get(key), "to": value}
            for key, value in asdict(config).items()
            if original["ppo_config_fields"].get(key) != value
        },
        "new_reservation_basis": "Prepared study seed audit of inherited data manifests and actual PPO game logs",
        "rng": {
            "preserved_keys": sorted(original["rng_states"]),
            "missing_mps_state": "torch_mps" not in original["rng_states"],
            "missing_mps_initialization": "config.seed, identically in both paired arms",
        },
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)
    if sha256(parent) != parent_sha256:
        raise RuntimeError("Parent checkpoint changed while branching")
    return payload["cohort_branch"]


def validate_panel(rows, seeds, repeats):
    expected = {(seed, repeat) for seed in seeds for repeat in repeats}
    actual = [(row["seed"], row.get("sampling_seed")) for row in rows]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("Duplicate, missing, or unexpected comparison seeds/repeats")


def combine_replicas(rows_by_replica):
    return [
        {**row, "sampling_seed": f"{replica}:{row.get('sampling_seed')}"}
        for replica in REPLICAS
        for row in rows_by_replica[replica]
    ]


def critic_summary(rows):
    game_scores, starts, decision_scores = [], [], []
    for row in rows:
        if row.get("censored") or row.get("stalled"):
            continue
        target = 8 if row["won"] else row["final_ante"] - 1
        scores = []
        for forecast in row.get("critic_forecasts", []):
            p = np.asarray(forecast["outcome_probabilities"], float)
            if len(p) != 9 or not np.isfinite(p).all() or (p < 0).any() or not np.isclose(p.sum(), 1):
                raise ValueError("Invalid critic probability forecast")
            score = {
                "outcome_nll": -math.log(max(p[target], 1e-12)),
                "outcome_brier": float(((p - np.eye(9)[target]) ** 2).sum()),
                "win_brier": float((p[-1] - row["won"]) ** 2),
            }
            if "expected_return" in forecast:
                error = forecast["expected_return"] - 10 * row["won"]
                if not math.isfinite(error):
                    raise ValueError("Non-finite return forecast")
                score.update(return_mse=error * error, return_bias=error)
            scores.append(score)
        if scores:
            decision_scores.extend(scores)
            game_scores.append({key: float(np.mean([s[key] for s in scores])) for key in scores[0]})
            starts.append((row["critic_forecasts"][0]["outcome_probabilities"][-1], float(row["won"])))
    result = {
        "games": len(game_scores),
        "decisions": len(decision_scores),
        "start": {},
        "interpretation": "Scores measure forecasts for this policy's fresh games, not action correctness",
    }
    for name, scores in (("game_weighted", game_scores), ("decision_weighted", decision_scores)):
        result[name] = {key: float(np.mean([s[key] for s in scores])) for key in scores[0]} if scores else {}
    if starts:
        p, y = np.asarray(starts).T
        result["start"] = {
            "predicted_win_rate": float(p.mean()),
            "observed_win_rate": float(y.mean()),
            "win_brier": float(((p - y) ** 2).mean()),
        }
    return result


def compare_arms(outcomes, seeds, *, confirmation=False):
    """Average repeats and replicas within a game seed before bootstrapping."""
    result = {
        "replicas": {},
        "combined": {},
        "confirmation": confirmation,
        "uncertainty": "Game-seed intervals conditional on these two trained replicas",
    }
    for replica in REPLICAS:
        result["replicas"][str(replica)] = {}
        for mode, repeats in (("sampled", [0, 1, 2]), ("greedy", [None])):
            selected = outcomes["complete_episode"][replica][mode]
            control = outcomes["completion_window"][replica][mode]
            validate_panel(selected, seeds, repeats)
            validate_panel(control, seeds, repeats)
            result["replicas"][str(replica)][mode] = paired_statistics(selected, control)
    for mode in ("sampled", "greedy"):
        selected = combine_replicas({r: outcomes["complete_episode"][r][mode] for r in REPLICAS})
        control = combine_replicas({r: outcomes["completion_window"][r][mode] for r in REPLICAS})
        result["combined"][mode] = {
            "paired": paired_statistics(selected, control),
            "complete_episode": summarize(selected),
            "completion_window": summarize(control),
            "critic_complete_episode": critic_summary(selected),
            "critic_completion_window": critic_summary(control),
        }
    sampled = result["combined"]["sampled"]["paired"]
    greedy = result["combined"]["greedy"]["paired"]
    failures = any(
        row.get("stalled") or row.get("censored")
        for arm in outcomes.values()
        for replica in arm.values()
        for rows in replica.values()
        for row in rows
    )
    result["acceptance"] = {
        "sampled_gain_at_least_five_points": sampled["sampled_or_greedy_delta"] >= 0.05 - 1e-12,
        "sampled_ci_excludes_zero": sampled["paired_ci95"][0] > 0,
        "both_replicas_positive": all(
            result["replicas"][str(r)]["sampled"]["sampled_or_greedy_delta"] > 0 for r in REPLICAS
        ),
        "greedy_rules_out_five_point_regression": greedy["paired_ci95"][0] > -0.05,
        "no_gameplay_failures": not failures,
    }
    passed = all(result["acceptance"].values())
    result["verdict"] = (
        "validated"
        if confirmation and passed
        else "promising"
        if passed
        else "negative"
        if sampled["paired_ci95"][1] < 0
        else "inconclusive"
    )
    return result
