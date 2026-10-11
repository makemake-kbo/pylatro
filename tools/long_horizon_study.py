"""Matched long-horizon credit study with untouched confirmation game seeds."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tarfile
import time
import uuid
from dataclasses import asdict, replace
from pathlib import Path

import torch

from pylatro import load_game_data
from pylatro_agent.checkpoint import load_checkpoint_payload, load_ppo_resume_payload
from pylatro_agent.cohort_study import (
    _seed_lists,
    combine_replicas,
    config_from_dict,
    critic_summary,
    validate_panel,
)
from pylatro_agent.experiment_budget import accelerator_job
from pylatro_agent.horizon_transfer import EarlyAntePolicy
from pylatro_agent.learning_bottleneck import atomic_json, read_json
from pylatro_agent.training.cohort_config import validate_cohort_resume
from pylatro_agent.training.cohort_training import _validate_state
from pylatro_agent.training.outcome_credit import _parent_configuration, _same_state
from pylatro_agent.training.ppo import train_ppo
from pylatro_agent.training.ppo_config import _ppo_run_provenance, _validate_ppo_config
from pylatro_agent.training.ppo_evaluation import training_seed_reservations
from pylatro_agent.transfer import (
    check_seed_separation,
    evaluate_artifact,
    load_policy,
    paired_statistics,
    restore_configuration,
    sha256,
    source_hashes,
    summarize,
)
from pylatro_agent.vocab import build_vocab

REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO / "eval/long_horizon_generalization_20260921"
ARMS = ("terminal_mc", "terminal_gae", "legacy_gae")
REPLICAS = (0, 1)
ENDPOINTS = (10, 20)
PRESERVED = ("state_dict", "optimizer_state_dict", "rng_states", "cohort_state", "total_steps", "update_count")


def audit_seeds(root, training, panels):
    proposed = [seed for values in training.values() for seed in values]
    proposed += [seed for values in panels.values() for seed in values]
    if not proposed or len(proposed) != len(set(proposed)):
        raise ValueError("New training and evaluation panels must be disjoint")
    prior = read_json(REPO / "eval/complete_cohort_long_20260915/manifest.json")
    inherited = prior["seed_audit"]["inherited_data"]
    for item in inherited:
        if sha256(item["path"]) != item["sha256"]:
            raise ValueError(f"Inherited seed evidence changed: {item['path']}")
    evidence = check_seed_separation(proposed, [item["path"] for item in inherited])
    historical, inventory = set(), []
    for path in sorted((REPO / "eval").rglob("manifest.json")):
        if root in path.parents or "cpu_verification" in path.parts:
            continue
        found = _seed_lists(read_json(path), training=False)
        if found:
            historical.update(found)
            inventory.append({"path": str(path), "sha256": sha256(path)})
    if set(proposed) & historical:
        raise ValueError("New panels overlap previous training, development, or reserved game seeds")
    return {"inherited_data": evidence, "historical_manifests": inventory, "historical_seed_count": len(historical)}


def branch_payload(
    original, *, arm, directory, seeds, reservations, sources, source_digest, parent, parent_hash,
    micro_batch_size=None,
):
    """Preserve learned state and consumed games; explicitly branch the recipe."""
    if arm not in ARMS:
        raise ValueError("Unknown credit treatment")
    config, migrations = _parent_configuration(original)
    architecture, _ = restore_configuration(original)
    old_config = asdict(config)
    cursor = original["cohort_state"]["seed_cursor"]
    config.cohort_seeds = config.cohort_seeds[:cursor] + list(seeds)
    if set(seeds) & set(config.cohort_seeds[:cursor]):
        raise ValueError("New training repeats consumed parent games")
    config.training_excluded_seeds = sorted(set(config.training_excluded_seeds) | set(reservations))
    config.total_updates = original["update_count"] + len(seeds) // config.cohort_games
    config.save_dir = str(directory / "checkpoints")
    config.log_dir = str(directory / "logs")
    config.stop_request_path = str(directory.parent / "STOP")
    config.ppo_run_uuid = str(uuid.uuid4())
    config.ppo_source_sha256 = source_digest
    config.ppo_recipe_id = f"long_horizon_credit_v1_{arm}_seed{config.seed}"
    if micro_batch_size is not None:
        if type(micro_batch_size) is not int or not 0 < micro_batch_size <= config.mini_batch_size:
            raise ValueError("Physical batch must be a positive integer no larger than the logical batch")
        config.micro_batch_size = micro_batch_size
    if arm != "legacy_gae":
        config.return_value_mode = "terminal_only"
        config.value_loss_coeff = 0.0
        architecture = replace(architecture, return_value_mode="terminal_only")
    config.advantage_estimator = "complete_episode_mc" if arm == "terminal_mc" else "fragment_gae"
    _validate_ppo_config(config)
    updated = asdict(config)
    payload = dict(original)
    # A new experimental branch has its own complete provenance. Never carry a
    # preparation-only record whose recipe describes a different treatment.
    payload.pop("outcome_credit_branch", None)
    payload.update(
        agent_config=asdict(architecture),
        cohort_ppo_config=updated,
        ppo_config_fields={**original["ppo_config_fields"], **updated},
        ppo_run_provenance=_ppo_run_provenance(config),
        planned_updates=config.total_updates,
        training_reserved_seeds=training_seed_reservations(config),
    )
    payload["cohort_branch"] = {
        "kind": "long_horizon_credit_control",
        "arm": arm,
        "parent": str(parent),
        "parent_sha256": parent_hash,
        "parent_branch": original.get("cohort_branch"),
        "source_hashes": sources,
        "source_sha256": source_digest,
        "configuration_migrations": migrations,
        "consumed_seed_count": cursor,
        "preserved_state": list(PRESERVED),
        "overrides": {k: {"from": old_config[k], "to": v} for k, v in updated.items() if old_config[k] != v},
    }
    validate_cohort_resume(config, payload)
    _validate_state(payload["cohort_state"], config, payload["update_count"], config.total_updates)
    for key in PRESERVED:
        if not _same_state(original[key], payload[key]):
            raise ValueError(f"Branch altered {key}")
    return payload


def replacement_budget(previous):
    """Carry only unspent self-imposed compute into a declared replacement."""
    previous = Path(previous).resolve()
    ledger_path = previous / "compute.json"
    ledger = read_json(ledger_path)
    if any(job["status"] == "running" for job in ledger["jobs"]):
        raise ValueError("Previous study still has a running accelerator job")
    spent = sum(job["elapsed_seconds"] for job in ledger["jobs"])
    limit = ledger["limit_seconds"]
    if (
        not all(math.isfinite(value) for value in (spent, limit, ledger["elapsed_seconds"]))
        or spent < 0 or abs(spent - ledger["elapsed_seconds"]) > 1e-3 or spent >= limit
    ):
        raise ValueError("Previous compute ledger is inconsistent or exhausted")
    return limit - spent, {
        "root": str(previous), "study_sha256": sha256(previous / "manifest.json"),
        "compute_sha256": sha256(ledger_path), "spent_seconds": spent, "prior_limit_seconds": limit,
        "reason": "Explicit resource revision; prior partial candidates are not carried forward",
    }


def prepare(root, *, micro_batch_size=None, training_seed_base=300000000, evaluation_seed_base=230000, replaces=None):
    if (root / "manifest.json").exists() or (root / "compute.json").exists():
        raise FileExistsError("Study already prepared")
    limit, replacement = replacement_budget(replaces) if replaces is not None else (12 * 3600, None)
    root.mkdir(parents=True, exist_ok=True)
    training = {
        str(r): list(range(training_seed_base + r * 100000, training_seed_base + r * 100000 + 20 * 64))
        for r in REPLICAS
    }
    panels = {
        "development": list(range(evaluation_seed_base, evaluation_seed_base + 200)),
        "confirmation": list(range(evaluation_seed_base + 10000, evaluation_seed_base + 10400)),
        "early_transfer": list(range(evaluation_seed_base + 20000, evaluation_seed_base + 20400)),
    }
    audit = audit_seeds(root, training, panels)
    sources = source_hashes()
    digest = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    protocol = {
        "version": 2,
        "created_unix": time.time(),
        "source_hashes": sources,
        "source_sha256": digest,
        "arms": ARMS,
        "replicas": REPLICAS,
        "endpoints": ENDPOINTS,
        "training_seeds": training,
        "panels": panels,
        "seed_audit": audit,
        "parents": {},
        "branches": {},
        "development_repeats": [0],
        "confirmation_repeats": [0, 1],
        "teacher_guidance": False,
        "replacement": replacement,
        "physical_batch_override": micro_batch_size,
        "parent_selection": "Update 122 is the last equally trained endpoint of both stopped continuation replicas",
        "comparison": (
            "terminal_mc versus terminal_gae isolates actor credit; terminal_gae versus legacy_gae isolates value"
        ),
        "selection_rule": "Largest development sampled MC-minus-terminal-GAE difference; ties choose earlier endpoint",
        "acceptance": {
            "mc_gain_vs_terminal_gae": "At least 5pp, paired 95% lower bound >0, each training replica positive",
            "mc_gain_vs_frozen_parent": "At least 5pp, paired 95% lower bound >0, each training replica positive",
            "greedy_guardrail": "Paired lower bound >-5pp against terminal GAE and frozen parent",
            "early_transfer": (
                "MC acts through Ante 3, terminal-GAE thereafter; positive terminal-win CI versus all-GAE"
            ),
            "integrity": "Complete disjoint panels, fixed sources, full optimizer exposure, no failed/censored games",
        },
        "interpretation": (
            "Fresh procedural-seed generalization on Blue Deck White Stake full Ante-8 games; not unseen decks"
        ),
    }
    reservations = [s for panel in panels.values() for s in panel]
    for replica in REPLICAS:
        parent = REPO / f"eval/complete_cohort_long_20260915/seed{replica}/checkpoints/ppo_update122.pt"
        original = load_checkpoint_payload(parent, "cpu")
        if original["update_count"] != 122:
            raise ValueError("Unexpected parent endpoint")
        parent_hash = sha256(parent)
        protocol["parents"][str(replica)] = {"path": str(parent), "sha256": parent_hash}
        for arm in ARMS:
            name = f"{arm}_seed{replica}"
            directory = root / name
            directory.mkdir(exist_ok=False)
            payload = branch_payload(
                original,
                arm=arm,
                directory=directory,
                seeds=training[str(replica)],
                reservations=reservations,
                sources=sources,
                source_digest=digest,
                parent=parent,
                parent_hash=parent_hash,
                micro_batch_size=micro_batch_size,
            )
            initial = directory / "initial.pt"
            torch.save(payload, initial)
            restored = load_ppo_resume_payload(
                initial,
                "cpu",
                active_reward_config=config_from_dict(payload["cohort_ppo_config"]).reward_config,
                active_win_ante=8,
            )
            for key in PRESERVED:
                if not _same_state(original[key], restored[key]):
                    raise ValueError(f"Saved branch altered {key}")
            protocol["branches"][name] = {
                "initial": str(initial),
                "sha256": sha256(initial),
                "recipe": payload["cohort_ppo_config"],
            }
        if sha256(parent) != parent_hash:
            raise ValueError("Parent changed during preparation")
    atomic_json(root / "manifest.json", protocol)
    atomic_json(
        root / "compute.json",
        {
            "limit_seconds": limit,
            "elapsed_seconds": 0,
            "jobs": [],
            "scope": (
                "Unspent portion of the superseded experiment's self-imposed cap"
                if replacement else
                "Self-imposed cap for this 120-update experiment; no prior ledger allowance is reused"
            ),
            "replacement": replacement,
            "authorization": "Active user goal authorizes model improvement and empirical generalization validation",
        },
    )
    return {"status": "prepared", "branches": len(protocol["branches"]), "updates": 120, "training_games_per_arm": 2560}


def load_protocol(root):
    protocol = read_json(root / "manifest.json")
    if source_hashes() != protocol["source_hashes"]:
        raise ValueError("Study implementation changed; preserve this experiment and prepare an explicit new branch")
    for parent in protocol["parents"].values():
        if sha256(parent["path"]) != parent["sha256"]:
            raise ValueError("Frozen parent changed")
    if "recovery" in protocol:
        recovery = protocol["recovery"]
        for path, digest in recovery["evidence"].items():
            if sha256(path) != digest:
                raise ValueError(f"Recovery evidence changed: {path}")
    return protocol


def recovery_payload(original, *, recipe, sources, source_digest, previous_path, previous_hash):
    """Rebind execution provenance while preserving an audited training prefix."""
    allowed = {
        "save_dir", "log_dir", "stop_request_path", "ppo_run_uuid", "ppo_source_sha256",
        "ppo_recipe_id", "total_updates",
    }
    before = original["cohort_ppo_config"]
    if set(before) != set(recipe):
        raise ValueError("Recovery recipe schema changed")
    changed = {key for key in before if before[key] != recipe[key]}
    if changed - allowed:
        raise ValueError(f"Recovery changes the learning recipe: {sorted(changed - allowed)}")
    if recipe["ppo_source_sha256"] != source_digest:
        raise ValueError("Recovery source digest differs from recipe")
    config = config_from_dict(recipe)
    payload = dict(original)
    payload.update(
        cohort_ppo_config=copy.deepcopy(recipe),
        ppo_config_fields={**original["ppo_config_fields"], **copy.deepcopy(recipe)},
        ppo_run_provenance=_ppo_run_provenance(config),
        planned_updates=config.total_updates,
    )
    payload["cohort_branch"] = {
        **copy.deepcopy(original["cohort_branch"]),
        "kind": "long_horizon_logical_kl_recovery",
        "source_hashes": sources,
        "source_sha256": source_digest,
        "recovered_prefix": {
            "checkpoint": str(previous_path), "sha256": previous_hash,
            "update_count": original["update_count"],
            "source_sha256": original["cohort_branch"]["source_sha256"],
            "branch": copy.deepcopy(original["cohort_branch"]),
        },
        "recovery_overrides": {key: {"from": before[key], "to": recipe[key]} for key in sorted(changed)},
        "preserved_state": list(PRESERVED),
    }
    validate_cohort_resume(config, payload)
    _validate_state(payload["cohort_state"], config, payload["update_count"], config.total_updates)
    for key in PRESERVED:
        if not _same_state(original[key], payload[key]):
            raise ValueError(f"Recovery altered {key}")
    return payload


def prepare_recovery(root, previous):
    """Recover the diagnosed physical-160 guard interruption into a new root.

    This revision retains accepted old-source prefixes and their immutable
    trajectories. All future updates use logical KL, and development panels
    are rerun. Original failure evidence and exposed development remain public.
    """
    root, previous = Path(root).resolve(), Path(previous).resolve()
    if root.exists():
        raise FileExistsError("Recovery requires a new directory")
    old = read_json(previous / "manifest.json")
    archive_metadata = read_json(previous / "analysis/core_sources_at_kl_failure.json")
    archive_path = previous / "analysis/core_sources_at_kl_failure.tar.gz"
    if (
        archive_metadata["source_hashes"] != old["source_hashes"]
        or sha256(archive_path) != archive_metadata["archive_sha256"]
    ):
        raise ValueError("Original source archive differs from the frozen study")
    sources = source_hashes()
    changed_sources = sorted(k for k in sources.keys() | old["source_hashes"].keys()
                             if sources.get(k) != old["source_hashes"].get(k))
    if changed_sources != sorted([
        "src/pylatro_agent/training/ppo_config.py",
        "src/pylatro_agent/training/ppo_optimization.py",
        "tools/long_horizon_study.py",
    ]):
        raise ValueError(f"Unexpected recovery source changes: {changed_sources}")
    replay_path = previous / "analysis/legacy_seed1_logical_kl_replay/result.json"
    replay = read_json(replay_path)
    if (
        replay["old_source_sha256"] != old["source_sha256"]
        or not replay["complete_optimizer_exposure"] or replay["candidate_checkpoint_written"]
        or replay["thresholds"] != {"hard": .08, "soft": .015}
        or any(replay["source_hashes"][key] != sources[key] for key in changed_sources if key.startswith("src/"))
    ):
        raise ValueError("Recovery lacks a successful replay of this exact optimizer fix")
    audit_path = previous / "analysis/accepted_prefix_logical_kl_audit.json"
    audit = read_json(audit_path)
    if audit["study_sha256"] != sha256(previous / "manifest.json") or audit["updates"] != 50:
        raise ValueError("Recovery prefix audit belongs to another study")
    endpoint_audits = [previous / "analysis" / name for name in (
        "first_replica_training_snapshot.json", "second_mc_endpoint_verification.json",
        "second_gae_endpoint_verification.json",
    )]
    endpoints = {}
    for path in endpoint_audits:
        verified = read_json(path)
        if verified["study_sha256"] != audit["study_sha256"]:
            raise ValueError("Endpoint audit belongs to another study")
        if verified["replica"] == 0:
            endpoints.update({f"{arm}_seed0": row["checkpoint_sha256"] for arm, row in verified["arms"].items()})
        else:
            endpoints[f"{verified['arm']}_seed1"] = verified["checkpoint_sha256"]
    if read_json(previous / "training_status.json")["status"] != "failed":
        raise ValueError("Expected the diagnosed stopped training run")
    for stage in ("confirmation", "early_transfer"):
        if list((previous / "evaluations" / stage).rglob("outcomes.jsonl")):
            raise ValueError("Recovery must precede confirmation and early-transfer exposure")
    limit, replacement = replacement_budget(previous)
    replacement["reason"] = "Logical-KL recovery retaining 50 verified accepted updates and all prior compute charges"
    digest = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    protocol = copy.deepcopy(old)
    protocol.update(version=3, created_unix=time.time(), source_hashes=sources, source_sha256=digest,
                    replacement=replacement, branches={})
    evidence = {str(p): sha256(p) for p in (
        previous / "manifest.json", previous / "compute.json", archive_path,
        previous / "analysis/core_sources_at_kl_failure.json", replay_path, audit_path,
        previous / "legacy_gae_seed1/logs/failure.json",
        *endpoint_audits,
    )}
    for row in replay["inputs"].items():
        evidence[row[0]] = row[1]
    root.mkdir()
    preserved_updates, copied_files = {}, {}
    for replica in REPLICAS:
        for arm in ARMS:
            name = f"{arm}_seed{replica}"
            count = 0 if name == "legacy_gae_seed1" else 10
            records = audit["evidence"][name]
            if len(records) != count:
                raise ValueError("Unexpected accepted prefix length")
            directory = root / name
            directory.mkdir()
            recipe = copy.deepcopy(old["branches"][name]["recipe"])
            recipe.update(
                save_dir=str(directory / "checkpoints"), log_dir=str(directory / "logs"),
                stop_request_path=str(root / "STOP"), ppo_run_uuid=str(uuid.uuid4()),
                ppo_source_sha256=digest, ppo_recipe_id=f"long_horizon_logical_kl_v1_{name}", total_updates=142,
            )
            for endpoint in (0, count) if count else (0,):
                source = checkpoint(previous, arm, replica, endpoint)
                original = validate_checkpoint(source, old, arm, replica, endpoint)
                source_hash = sha256(source)
                if endpoint == 0 and source_hash != old["branches"][name]["sha256"]:
                    raise ValueError("Original prepared branch changed")
                if endpoint and source_hash != endpoints[name]:
                    raise ValueError("Previously audited endpoint checkpoint changed")
                destination = checkpoint(root, arm, replica, endpoint)
                destination.parent.mkdir(exist_ok=True)
                payload = recovery_payload(
                    original, recipe=recipe, sources=sources, source_digest=digest,
                    previous_path=source, previous_hash=source_hash,
                )
                torch.save(payload, destination)
                restored = load_checkpoint_payload(destination, "cpu")
                for key in PRESERVED:
                    if not _same_state(original[key], restored[key]):
                        raise ValueError(f"Saved recovery changed {name}/{key}")
                evidence[str(source)] = source_hash
                if endpoint == 0:
                    protocol["branches"][name] = {
                        "initial": str(destination), "sha256": sha256(destination), "recipe": recipe,
                    }
                del original, payload, restored
            observed_seeds = []
            for step, record in enumerate(records, 123):
                if record["update"] != step or record["optimizer_examples"] != 16384:
                    raise ValueError("Unmatched inherited optimizer exposure")
                for path, expected in record["artifacts"].items():
                    if sha256(path) != expected:
                        raise ValueError("Audited prefix metadata changed")
                source = previous / name / "logs/cohorts" / f"update_{step}"
                manifest = read_json(source / "manifest.json")
                update = read_json(source / "update.json")
                opt = update["optimizer"]
                if opt["kl_rollback"] or opt["stop_reason"] != "none" or max(opt["approx_kls"]) >= .08:
                    raise ValueError("Inherited update is not an accepted unchanged-guard prefix")
                for episode in manifest["episodes"]:
                    if sha256(source / episode["path"]) != episode["sha256"]:
                        raise ValueError("Inherited trajectory changed")
                    observed_seeds.append(episode["seed"])
                target = directory / "logs/cohorts" / source.name
                shutil.copytree(source, target)
                for path in source.rglob("*"):
                    if path.is_file():
                        copied = target / path.relative_to(source)
                        original_hash = sha256(path)
                        if sha256(copied) != original_hash:
                            raise ValueError("Copied prefix changed")
                        copied_files[str(copied)] = original_hash
            if observed_seeds != old["training_seeds"][str(replica)][:count * 64]:
                raise ValueError("Inherited training seeds differ from the original plan")
            preserved_updates[name] = count
    evidence.update(copied_files)
    protocol["recovery"] = {
        "previous_root": str(previous), "previous_source_sha256": old["source_sha256"],
        "changed_sources": changed_sources, "preserved_updates": preserved_updates,
        "evidence": evidence, "prior_compute": replacement,
        "prefix_semantics": "Copied updates retain their original physical-KL source/configuration; all were accepted",
        "future_semantics": "Hard KL uses sample-weighted logical optimizer groups; thresholds .08/.015 unchanged",
        "development": "Previously exposed development seeds are rerun; never counted as fresh confirmation",
        "selection_and_acceptance_unchanged": True,
        "confirmation_and_early_transfer_unexposed": True,
    }
    atomic_json(root / "manifest.json", protocol)
    atomic_json(root / "compute.json", {
        "limit_seconds": limit, "elapsed_seconds": 0, "jobs": [], "replacement": replacement,
        "scope": "Remaining original 12-hour allowance after all training failures and diagnostic replays",
        "authorization": "Active user goal authorizes model improvement and empirical generalization validation",
    })
    archive = root / "sources_at_recovery.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        for path, expected in sources.items():
            if sha256(REPO / path) != expected:
                raise ValueError("Source changed during preparation")
            bundle.add(REPO / path, arcname=path)
    atomic_json(root / "source_archive.json", {"source_sha256": digest, "archive_sha256": sha256(archive)})
    load_protocol(root)
    for replica in REPLICAS:
        for arm in ARMS:
            count = preserved_updates[f"{arm}_seed{replica}"]
            validate_checkpoint(checkpoint(root, arm, replica, count), protocol, arm, replica, count)
    return {"status": "prepared", "preserved_updates": 50, "remaining_updates": 70,
            "remaining_accelerator_seconds": limit, "source_sha256": digest}


def checkpoint(root, arm, replica, endpoint):
    directory = root / f"{arm}_seed{replica}"
    return directory / (f"checkpoints/ppo_update{122 + endpoint}.pt" if endpoint else "initial.pt")


def validate_checkpoint(path, protocol, arm, replica, endpoint):
    payload = load_checkpoint_payload(path, "cpu")
    recipe = protocol["branches"][f"{arm}_seed{replica}"]["recipe"]
    validate_cohort_resume(config_from_dict(recipe), payload)
    branch = payload["cohort_branch"]
    if (
        payload["update_count"] != 122 + endpoint
        or branch["arm"] != arm
        or branch["source_sha256"] != protocol["source_sha256"]
        or branch["parent_sha256"] != protocol["parents"][str(replica)]["sha256"]
    ):
        raise ValueError("Checkpoint does not match the declared treatment/endpoint")
    _validate_state(
        payload["cohort_state"], config_from_dict(recipe), payload["update_count"], payload["planned_updates"]
    )
    return payload


def train(root, arm, replica, endpoint):
    protocol = load_protocol(root)
    final = checkpoint(root, arm, replica, endpoint)
    if final.exists():
        validate_checkpoint(final, protocol, arm, replica, endpoint)
        return {"status": "already_complete"}
    directory = root / f"{arm}_seed{replica}"
    existing = list((directory / "checkpoints").glob("ppo_update*.pt"))
    previous = (
        max(existing, key=lambda p: int(p.stem.removeprefix("ppo_update"))) if existing else directory / "initial.pt"
    )
    payload = load_checkpoint_payload(previous, "cpu")
    current = payload["update_count"] - 122
    if current >= endpoint:
        raise ValueError("Refuse to train backwards from an unexpected checkpoint")
    validate_checkpoint(previous, protocol, arm, replica, current)
    config = config_from_dict(protocol["branches"][f"{arm}_seed{replica}"]["recipe"])
    config.total_updates = 122 + endpoint
    torch.set_num_threads(1)
    with accelerator_job(root, f"{arm}_seed{replica}_to{endpoint}", (endpoint - current) * 280):
        train_ppo(config, resume_path=str(previous))
    final_payload = validate_checkpoint(final, protocol, arm, replica, endpoint)
    return {"status": "complete", "update": final_payload["update_count"]}


def modes(protocol, stage):
    repeats = protocol["development_repeats"] if stage == "development" else protocol["confirmation_repeats"]
    return [(f"sample{r}", False, r) for r in repeats] + [("greedy", True, None)]


def evaluation_path(root, stage, endpoint, arm, replica, mode):
    return root / "evaluations" / stage / f"update{endpoint}" / f"{arm}_seed{replica}" / mode


def read_panel(directory, protocol, seeds, checkpoint_hash, repeat):
    manifest = read_json(directory / "manifest.json")
    if (
        manifest["status"] != "complete"
        or manifest["checkpoint_sha256"] != checkpoint_hash
        or manifest["source_sha256"] != protocol["source_sha256"]
    ):
        raise ValueError("Evaluation artifact is incomplete or mismatched")
    rows = [json.loads(line) for line in (directory / "outcomes.jsonl").read_text().splitlines()]
    validate_panel(rows, seeds, [repeat])
    if any(row.get("censored") or row.get("stalled") for row in rows):
        raise ValueError("Censored/stalled games cannot establish generalization")
    return rows


def verify_training(root, protocol, endpoint):
    evidence = {}
    for arm in ARMS:
        for replica in REPLICAS:
            name = f"{arm}_seed{replica}"
            recipe = protocol["branches"][name]["recipe"]
            initial = load_checkpoint_payload(checkpoint(root, arm, replica, 0), "cpu")
            if sha256(checkpoint(root, arm, replica, 0)) != protocol["branches"][name]["sha256"]:
                raise ValueError("Prepared branch changed")
            expected_seeds = protocol["training_seeds"][str(replica)][: endpoint * recipe["cohort_games"]]
            observed_seeds, artifacts = [], {}
            if (root / name / "logs/failure.json").exists():
                raise ValueError("Training failure needs investigation before interpreting this arm")
            for step in range(1, endpoint + 1):
                folder = root / name / "logs/cohorts" / f"update_{122 + step}"
                cohort = read_json(folder / "manifest.json")
                update = read_json(folder / "update.json")
                if not cohort["complete"] or cohort["teacher_labels"]:
                    raise ValueError("Incomplete cohort or unexpected teacher supervision")
                observed_seeds.extend(row["seed"] for row in cohort["episodes"])
                optimizer = update["optimizer"]
                if (
                    optimizer["kl_rollback"]
                    or optimizer["ppo_samples_processed"] != recipe["ppo_epochs"] * recipe["cohort_training_rows"]
                ):
                    raise ValueError("Unmatched optimizer exposure")
                if (
                    update["buffer"]["label_coverage"]["all"]["rows"]
                    != update["buffer"]["label_coverage"]["all"]["labels"]
                ):
                    raise ValueError("Missing completed outcome labels")
                if (
                    arm == "terminal_mc"
                    and update["buffer"]["credit_targets"]["return_target_mse_to_completed_return"] != 0
                ):
                    raise ValueError("Complete-episode targets differ from observed returns")
                for row in cohort["episodes"]:
                    if sha256(folder / row["path"]) != row["sha256"]:
                        raise ValueError("Saved training trajectory changed")
                for filename in ("manifest.json", "update.json"):
                    artifacts[str(folder / filename)] = sha256(folder / filename)
            if observed_seeds != expected_seeds:
                raise ValueError("Training did not consume its exact disjoint seed plan")
            final = validate_checkpoint(checkpoint(root, arm, replica, endpoint), protocol, arm, replica, endpoint)
            if final["cohort_state"]["seed_cursor"] - initial["cohort_state"]["seed_cursor"] != len(expected_seeds):
                raise ValueError("Seed cursor does not account for collected games")
            evidence[name] = {"games": len(observed_seeds), "artifacts": artifacts}
    return evidence


def selection(root, protocol):
    selected = read_json(root / "selection.json")
    if selected["study_sha256"] != sha256(root / "manifest.json"):
        raise ValueError("Selection belongs to another protocol")
    for path, digest in selected["checkpoint_hashes"].items():
        if sha256(path) != digest:
            raise ValueError("Selected checkpoint changed")
    return selected


def evaluate(root, arm, replica, endpoint, stage):
    protocol = load_protocol(root)
    if stage == "confirmation":
        selected = selection(root, protocol)
        if endpoint != selected["endpoint"] and arm != "baseline":
            raise ValueError("Confirmation requires the frozen endpoint")
    source_arm = "legacy_gae" if arm == "baseline" else arm
    path = checkpoint(root, source_arm, replica, 0 if arm == "baseline" else endpoint)
    validate_checkpoint(path, protocol, source_arm, replica, 0 if arm == "baseline" else endpoint)
    digest = sha256(path)
    seeds = protocol["panels"][stage]
    data = load_game_data()
    vocab = build_vocab(data)
    model, reward, _ = load_policy(path, vocab, "cpu")
    for mode, greedy, repeat in modes(protocol, stage):
        output = evaluation_path(root, stage, endpoint, arm, replica, mode)
        if output.exists():
            read_panel(output, protocol, seeds, digest, repeat)
            continue
        evaluate_artifact(
            model,
            reward,
            data,
            vocab,
            seeds,
            torch.device("cpu"),
            output,
            greedy=greedy,
            sampling_seed=repeat or 0,
            deck="b_blue",
            batch_size=32,
            workers=4,
            cpu_threads=4,
            record_return_forecasts=True,
            metadata={
                "checkpoint": str(path),
                "checkpoint_sha256": digest,
                "source_sha256": protocol["source_sha256"],
                "arm": arm,
                "replica": replica,
                "endpoint": endpoint,
                "pid": os.getpid(),
            },
        )
        read_panel(output, protocol, seeds, digest, repeat)
    return {"status": "complete", "arm": arm, "replica": replica, "endpoint": endpoint, "stage": stage}


def comparison(outcomes, left, right):
    result = {"replicas": {}, "combined": {}}
    for r in REPLICAS:
        result["replicas"][str(r)] = {
            mode: paired_statistics(outcomes[left][r][mode], outcomes[right][r][mode]) for mode in ("sampled", "greedy")
        }
    for mode in ("sampled", "greedy"):
        result["combined"][mode] = paired_statistics(
            combine_replicas({r: outcomes[left][r][mode] for r in REPLICAS}),
            combine_replicas({r: outcomes[right][r][mode] for r in REPLICAS}),
        )
    sampled, greedy = result["combined"]["sampled"], result["combined"]["greedy"]
    result["passes_gameplay_gate"] = (
        sampled["sampled_or_greedy_delta"] >= 0.05 - 1e-12
        and sampled["paired_ci95"][0] > 0
        and greedy["paired_ci95"][0] > -0.05
        and all(result["replicas"][str(r)]["sampled"]["sampled_or_greedy_delta"] > 0 for r in REPLICAS)
    )
    return result


def report(root, endpoint, stage):
    protocol = load_protocol(root)
    if stage == "confirmation" and selection(root, protocol)["endpoint"] != endpoint:
        raise ValueError("Report endpoint differs from frozen selection")
    outcomes, artifacts = {}, {}
    for arm in ("baseline", *ARMS):
        outcomes[arm] = {}
        for r in REPLICAS:
            outcomes[arm][r] = {"sampled": [], "greedy": []}
            arm_endpoint = 0 if arm == "baseline" else endpoint
            path = checkpoint(root, "legacy_gae" if arm == "baseline" else arm, r, arm_endpoint)
            validate_checkpoint(path, protocol, "legacy_gae" if arm == "baseline" else arm, r, arm_endpoint)
            for mode, greedy, repeat in modes(protocol, stage):
                directory = evaluation_path(root, stage, arm_endpoint, arm, r, mode)
                rows = read_panel(directory, protocol, protocol["panels"][stage], sha256(path), repeat)
                outcomes[arm][r]["greedy" if greedy else "sampled"].extend(rows)
                artifacts[str(directory / "outcomes.jsonl")] = sha256(directory / "outcomes.jsonl")
    summaries = {}
    for arm in outcomes:
        summaries[arm] = {}
        for mode in ("sampled", "greedy"):
            rows = combine_replicas({r: outcomes[arm][r][mode] for r in REPLICAS})
            summaries[arm][mode] = {**summarize(rows), "critic": critic_summary(rows)}
    comparisons = {
        f"{left}_vs_{right}": comparison(outcomes, left, right)
        for left, right in (
            ("terminal_mc", "terminal_gae"),
            ("terminal_mc", "baseline"),
            ("terminal_mc", "legacy_gae"),
            ("terminal_gae", "legacy_gae"),
        )
    }
    result = {
        "stage": stage,
        "endpoint": endpoint,
        "summaries": summaries,
        "comparisons": comparisons,
        "artifacts": artifacts,
        "study_sha256": sha256(root / "manifest.json"),
        "training_verification": verify_training(root, protocol, endpoint),
        "generalization_proven": False,
        "interpretation": (
            "Development does not validate gains; confirmation also needs the early-transfer intervention"
        ),
    }
    atomic_json(root / f"{stage}_update{endpoint}.json", result)
    return result


def select(root):
    protocol = load_protocol(root)
    if (root / "selection.json").exists():
        return selection(root, protocol)
    results = [report(root, endpoint, "development") for endpoint in ENDPOINTS]
    best = max(
        results,
        key=lambda value: (
            value["comparisons"]["terminal_mc_vs_terminal_gae"]["combined"]["sampled"]["sampled_or_greedy_delta"],
            -value["endpoint"],
        ),
    )
    endpoint = best["endpoint"]
    selected = {
        "endpoint": endpoint,
        "study_sha256": sha256(root / "manifest.json"),
        "created_unix": time.time(),
        "rule": protocol["selection_rule"],
        "development_artifacts": {
            str(root / f"development_update{e}.json"): sha256(root / f"development_update{e}.json") for e in ENDPOINTS
        },
        "checkpoint_hashes": {
            str(checkpoint(root, arm, r, e)): sha256(checkpoint(root, arm, r, e))
            for arm in ARMS
            for r in REPLICAS
            for e in (0, endpoint)
        },
        "confirmation_started": False,
    }
    atomic_json(root / "selection.json", selected)
    return selected


def confirm(root):
    protocol = load_protocol(root)
    endpoint = selection(root, protocol)["endpoint"]
    for arm in ("baseline", *ARMS):
        for r in REPLICAS:
            e = 0 if arm == "baseline" else endpoint
            run_child(
                root,
                ["evaluate", "--stage", "confirmation", "--arm", arm, "--replica", str(r), "--endpoint", str(e)],
                "confirmation_status.json",
                f"confirmation_{arm}_seed{r}_u{e}",
            )
    return report(root, endpoint, "confirmation")


def early_transfer(root):
    protocol = load_protocol(root)
    selected = selection(root, protocol)
    endpoint = selected["endpoint"]
    seeds = protocol["panels"]["early_transfer"]
    outcomes, artifacts, routes = {"early_mc": {}, "all_gae": {}}, {}, {}
    data = load_game_data()
    vocab = build_vocab(data)
    for replica in REPLICAS:
        paths = {arm: checkpoint(root, arm, replica, endpoint) for arm in ("terminal_mc", "terminal_gae")}
        models = {arm: load_policy(path, vocab, "cpu")[0] for arm, path in paths.items()}
        _, reward = restore_configuration(load_checkpoint_payload(paths["terminal_gae"], "cpu"))
        hybrid = EarlyAntePolicy(models["terminal_mc"], models["terminal_gae"], through_ante=3)
        hashes = {arm: sha256(path) for arm, path in paths.items()}
        for arm, model in (("early_mc", hybrid), ("all_gae", models["terminal_gae"])):
            outcomes[arm][replica] = {"sampled": []}
            for repeat in protocol["confirmation_repeats"]:
                output = root / "early_transfer" / f"{arm}_seed{replica}" / f"sample{repeat}"
                if not output.exists():
                    evaluate_artifact(
                        model,
                        reward,
                        data,
                        vocab,
                        seeds,
                        torch.device("cpu"),
                        output,
                        sampling_seed=repeat,
                        deck="b_blue",
                        batch_size=32,
                        workers=4,
                        cpu_threads=4,
                        record_return_forecasts=True,
                        metadata={
                            "arm": arm,
                            "replica": replica,
                            "source_sha256": protocol["source_sha256"],
                            "checkpoint_sha256": hashes["terminal_gae"],
                            "checkpoint_hashes": hashes,
                            "through_ante": 3,
                            "selection_sha256": sha256(root / "selection.json"),
                            "value_warning": "Component critic values do not predict the hybrid policy",
                        },
                    )
                    atomic_json(output / "routes.json", dict(hybrid.route_counts) if arm == "early_mc" else {})
                manifest = read_json(output / "manifest.json")
                if manifest["checkpoint_hashes"] != hashes or manifest["selection_sha256"] != sha256(
                    root / "selection.json"
                ):
                    raise ValueError("Early-transfer checkpoint/selection changed")
                rows = read_panel(output, protocol, seeds, hashes["terminal_gae"], repeat)
                outcomes[arm][replica]["sampled"].extend(rows)
                artifacts[str(output / "outcomes.jsonl")] = sha256(output / "outcomes.jsonl")
                routes[str(output)] = read_json(output / "routes.json")
    combined = {arm: combine_replicas({r: outcomes[arm][r]["sampled"] for r in REPLICAS}) for arm in outcomes}
    paired = paired_statistics(combined["early_mc"], combined["all_gae"])
    replica_deltas = {
        str(r): paired_statistics(outcomes["early_mc"][r]["sampled"], outcomes["all_gae"][r]["sampled"])
        for r in REPLICAS
    }
    result = {
        "paired": paired,
        "replicas": replica_deltas,
        "artifacts": artifacts,
        "routes": routes,
        "summaries": {arm: summarize(rows) for arm, rows in combined.items()},
        "passes_early_transfer_gate": paired["paired_ci95"][0] > 0
        and all(value["sampled_or_greedy_delta"] > 0 for value in replica_deltas.values()),
        "interpretation": "Only Ante 1-3 policy is changed. Both groups use the identical GAE policy in Antes 4-8.",
    }
    atomic_json(root / "early_transfer.json", result)
    return result


def finalize(root):
    protocol = load_protocol(root)
    endpoint = selection(root, protocol)["endpoint"]
    confirmation = report(root, endpoint, "confirmation")
    early = read_json(root / "early_transfer.json")
    for path, digest in early["artifacts"].items():
        if sha256(path) != digest:
            raise ValueError("Early-transfer evidence changed")
    # Regenerate the intervention statistics from completed per-game files.
    # early_transfer reuses hash-checked panels and never reruns completed games.
    early = early_transfer(root)
    gates = {
        "heldout_gain_vs_matched_gae": confirmation["comparisons"]["terminal_mc_vs_terminal_gae"][
            "passes_gameplay_gate"
        ],
        "heldout_gain_vs_frozen_parent": confirmation["comparisons"]["terminal_mc_vs_baseline"]["passes_gameplay_gate"],
        "early_decisions_improve_terminal_wins": early["passes_early_transfer_gate"],
    }
    result = {
        "generalization_proven": all(gates.values()),
        "gates": gates,
        "scope": protocol["interpretation"],
        "endpoint": endpoint,
        "confirmation": confirmation,
        "early_transfer": early,
        "limitation": (
            "Game-seed uncertainty conditions on two training replicas; does not prove transfer to other decks/tasks"
        ),
    }
    atomic_json(root / "final_results.json", result)
    lines = [
        f"Generalization evidence: {'passes' if result['generalization_proven'] else 'not established'}.\n",
        "| Policy | Sampled win rate | Greedy win rate |",
        "| --- | ---: | ---: |",
    ]
    for arm, values in confirmation["summaries"].items():
        lines.append(f"| {arm} | {values['sampled']['win_rate']:.2%} | {values['greedy']['win_rate']:.2%} |")
    lines.append("\nPaired differences below average both training replicas and sampling repeats within game seed.\n")
    for name in ("terminal_mc_vs_terminal_gae", "terminal_mc_vs_baseline"):
        paired = confirmation["comparisons"][name]["combined"]["sampled"]
        lo, hi = paired["paired_ci95"]
        lines.append(f"- {name}: {100 * paired['sampled_or_greedy_delta']:+.2f}pp [{100 * lo:+.2f}, {100 * hi:+.2f}].")
    lo, hi = early["paired"]["paired_ci95"]
    lines.append(
        f"- Early-policy intervention: {100 * early['paired']['sampled_or_greedy_delta']:+.2f}pp "
        f"[{100 * lo:+.2f}, {100 * hi:+.2f}]."
    )
    lines.append("\n" + result["scope"] + ". " + result["limitation"] + ".\n")
    (root / "report.md").write_text("\n".join(lines))
    return {"generalization_proven": result["generalization_proven"], "gates": gates}


def run_child(root, command, status, name):
    log_path = root / "jobs" / f"{name}.log"
    log_path.parent.mkdir(exist_ok=True)
    if log_path.exists():
        raise FileExistsError(f"Inspect prior job before resuming: {log_path}")
    with log_path.open("x") as stream:
        child = subprocess.Popen(
            [sys.executable, "-u", "-m", "tools.long_horizon_study", *command, "--root", str(root)],
            stdout=stream,
            stderr=subprocess.STDOUT,
            cwd=REPO,
        )
        record = {
            "status": "running",
            "pid": os.getpid(),
            "child_pid": child.pid,
            "job": name,
            "command": command,
            "started_unix": time.time(),
        }
        print(json.dumps(record), flush=True)
        while True:
            record["heartbeat_unix"] = time.time()
            atomic_json(root / status, record)
            try:
                code = child.wait(timeout=30)
                break
            except subprocess.TimeoutExpired:
                continue
        record.update(status="complete" if code == 0 else "failed", exit_code=code, finished_unix=time.time())
        atomic_json(root / status, record)
        if code:
            raise RuntimeError(f"{name} exited {code}; inspect {log_path}")


def train_all(root):
    load_protocol(root)
    for endpoint in ENDPOINTS:
        for replica in REPLICAS:
            for arm in ARMS:
                if (root / "STOP").exists():
                    atomic_json(root / "training_status.json", {"status": "stopped", "pid": os.getpid()})
                    return
                run_child(
                    root,
                    ["train", "--arm", arm, "--replica", str(replica), "--endpoint", str(endpoint)],
                    "training_status.json",
                    f"train_{arm}_seed{replica}_to{endpoint}",
                )
    atomic_json(root / "training_status.json", {"status": "complete", "all_endpoints": True, "pid": os.getpid()})


def evaluate_all(root):
    load_protocol(root)
    for endpoint in (0, *ENDPOINTS):
        for replica in REPLICAS:
            for arm in ("baseline",) if endpoint == 0 else ARMS:
                path = checkpoint(root, "legacy_gae" if arm == "baseline" else arm, replica, endpoint)
                while not path.exists():
                    status = read_json(root / "training_status.json")
                    if status["status"] in ("failed", "stopped") or status.get("all_endpoints"):
                        raise RuntimeError("Training ended without the required endpoint")
                    # A stale heartbeat alone is never treated as proof of a dead job.
                    os.kill(status["pid"], 0)
                    atomic_json(
                        root / "evaluation_status.json",
                        {
                            "status": "waiting_for_checkpoint",
                            "pid": os.getpid(),
                            "path": str(path),
                            "heartbeat_unix": time.time(),
                        },
                    )
                    time.sleep(10)
                run_child(
                    root,
                    ["evaluate", "--arm", arm, "--replica", str(replica), "--endpoint", str(endpoint)],
                    "evaluation_status.json",
                    f"development_{arm}_seed{replica}_u{endpoint}",
                )
        if endpoint:
            report(root, endpoint, "development")
    atomic_json(root / "evaluation_status.json", {"status": "complete", "all_endpoints": True, "pid": os.getpid()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "prepare",
            "prepare-recovery",
            "train",
            "evaluate",
            "report",
            "train-all",
            "evaluate-all",
            "select",
            "confirm",
            "early-transfer",
            "finalize",
        ),
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--arm", choices=(*ARMS, "baseline"), default="terminal_mc")
    parser.add_argument("--replica", type=int, choices=REPLICAS, default=0)
    parser.add_argument("--endpoint", type=int, choices=(0, *ENDPOINTS), default=20)
    parser.add_argument("--stage", choices=("development", "confirmation"), default="development")
    parser.add_argument("--micro-batch-size", type=int, help="Prepare only: explicit physical batch override")
    parser.add_argument("--training-seed-base", type=int, help="Prepare only: first replica's fresh seed range")
    parser.add_argument("--evaluation-seed-base", type=int, help="Prepare only: development seed range")
    parser.add_argument("--replaces", type=Path, help="Prepare only: carry the predecessor's unspent compute cap")
    parser.add_argument("--recover-from", type=Path, help="Prepare-recovery only: diagnosed physical-160 study")
    args = parser.parse_args()
    if args.command != "prepare" and any(
        value is not None for value in (
            args.micro_batch_size, args.training_seed_base, args.evaluation_seed_base, args.replaces,
        )
    ):
        parser.error("Batch, seed, and replacement overrides apply only when preparing a new study")
    root = args.root.resolve()
    if (args.command == "prepare-recovery") != (args.recover_from is not None):
        parser.error("prepare-recovery requires --recover-from, which is exclusive to that command")
    if args.command == "prepare":
        result = prepare(
            root, micro_batch_size=args.micro_batch_size,
            training_seed_base=args.training_seed_base if args.training_seed_base is not None else 300000000,
            evaluation_seed_base=args.evaluation_seed_base if args.evaluation_seed_base is not None else 230000,
            replaces=args.replaces,
        )
    elif args.command == "prepare-recovery":
        result = prepare_recovery(root, args.recover_from)
    elif args.command == "train":
        result = train(root, args.arm, args.replica, args.endpoint)
    elif args.command == "evaluate":
        result = evaluate(root, args.arm, args.replica, args.endpoint, args.stage)
    elif args.command == "report":
        result = report(root, args.endpoint, args.stage)
    elif args.command == "train-all":
        result = train_all(root)
    elif args.command == "select":
        result = select(root)
    elif args.command == "confirm":
        result = confirm(root)
    elif args.command == "early-transfer":
        result = early_transfer(root)
    elif args.command == "finalize":
        result = finalize(root)
    else:
        result = evaluate_all(root)
    if result:
        print(json.dumps(result if args.command != "report" else result["comparisons"], indent=2), flush=True)


if __name__ == "__main__":
    main()
