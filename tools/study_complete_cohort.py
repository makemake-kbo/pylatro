#!/usr/bin/env python3
"""Prepare, verify, and explicitly launch the complete-game outcome-label experiment."""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from pylatro import load_game_data
from pylatro_agent.checkpoint import load_checkpoint_payload
from pylatro_agent.cohort_study import (
    ARMS,
    ENDPOINTS,
    PRIOR_ROOT,
    REPLICAS,
    REPO,
    branch_checkpoint,
    compare_arms,
    config_from_dict,
    implementation_hashes,
    prepare_study,
    validate_panel,
    verify_sources,
)
from pylatro_agent.experiment_budget import accelerator_job
from pylatro_agent.learning_bottleneck import atomic_json, read_json
from pylatro_agent.outcome_study import critic_from_pooled, critic_metrics, pooled_critic_inputs
from pylatro_agent.training.cohort import CohortBuffer, policy_digest
from pylatro_agent.training.cohort_config import validate_cohort_resume
from pylatro_agent.training.cohort_io import load_cohort
from pylatro_agent.training.ppo import train_ppo
from pylatro_agent.training.ppo_evaluation import training_seed_reservations
from pylatro_agent.training.ppo_policy import _critic_predictions
from pylatro_agent.transfer import evaluate_artifact, load_policy, sha256
from pylatro_agent.value_head import outcome_nll
from pylatro_agent.vocab import build_vocab

DEFAULT_ROOT = REPO / "eval/complete_cohort_v1"


def load_protocol(root):
    protocol = read_json(Path(root) / "manifest.json")
    if protocol.get("version") != 1:
        raise ValueError("Unknown complete-cohort study protocol")
    verify_sources(protocol)
    for parent in protocol["parents"].values():
        if sha256(parent["path"]) != parent["sha256"] or sha256(parent["recipe_path"]) != parent["recipe_sha256"]:
            raise ValueError("Frozen parent checkpoint/recipe changed")
    return protocol


def verify_cpu(root):
    root = Path(root)
    destination = root / "cpu_verification"
    destination.mkdir(parents=True, exist_ok=False)
    command = [
        sys.executable,
        "-m",
        "pytest",
        "tests/test_complete_cohort.py",
        "tests/test_complete_cohort_training.py",
        "tests/test_complete_cohort_study.py",
        "tests/test_rollout_buffer.py",
        "tests/test_ppo_resume_checkpoint.py",
        "tests/test_outcome_ppo_pipeline.py",
        "tests/test_ppo_deck.py",
        "tests/test_outcome_branch.py",
        "tests/test_batched_eval.py",
        "--junitxml",
        str(destination / "junit.xml"),
        "--basetemp",
        str(destination / "artifacts"),
    ]
    started = time.monotonic()
    result = subprocess.run(command, cwd=REPO, text=True, capture_output=True, check=False)
    (destination / "pytest.log").write_text(result.stdout + result.stderr)
    artifact = {
        "command": command,
        "exit_code": result.returncode,
        "elapsed_seconds": time.monotonic() - started,
        "device": "cpu",
        "accelerator_training_seconds": 0,
        "source_hashes": implementation_hashes(),
        "status": "passed" if result.returncode == 0 else "failed",
        "log_sha256": sha256(destination / "pytest.log"),
    }
    atomic_json(destination / "manifest.json", artifact)
    if result.returncode:
        raise RuntimeError(f"CPU verification failed; see {destination / 'pytest.log'}")
    return artifact


def assign_budget(root, hours):
    if not math.isfinite(hours) or hours <= 0:
        raise ValueError("An explicit positive accelerator allowance is required")
    path = Path(root) / "compute.json"
    ledger = read_json(path)
    if ledger["limit_seconds"] or ledger["jobs"]:
        raise ValueError("Budget is already assigned; preserve its ledger instead of resetting it")
    ledger.update(limit_seconds=hours * 3600, authorization="Explicit budget CLI assignment")
    atomic_json(path, ledger)


def train_arm(root, protocol, arm, replica, endpoint, *, device, estimate_seconds):
    if not math.isfinite(estimate_seconds) or estimate_seconds <= 0:
        raise ValueError("Training requires a finite positive runtime estimate")
    if endpoint not in (10, 20):
        raise ValueError("Training must target one of the fixed pilot endpoints")
    root = Path(root)
    verification = read_json(root / "cpu_verification/manifest.json")
    if verification["status"] != "passed" or verification["source_hashes"] != protocol["source_hashes"]:
        raise ValueError("Current source requires passing CPU verification before a pilot")
    name = f"{arm}_seed{replica}"
    directory = root / name
    config = config_from_dict(protocol["recipes"][name]["config"])
    config.total_updates = 42 + endpoint
    config.device = device
    directory.mkdir(exist_ok=True)
    execution_path = directory / "execution.json"
    if execution_path.exists():
        if read_json(execution_path)["device"] != device:
            raise ValueError("Do not change the backend of an in-progress paired arm")
    else:
        atomic_json(
            execution_path,
            {"device": device, "device_override": device != "mps", "protocol_sha256": sha256(root / "manifest.json")},
        )
    latest = Path(config.save_dir) / "ppo_latest.pt"
    initial = directory / "initial.pt"
    context = accelerator_job(root, name, estimate_seconds) if device == "mps" else contextlib.nullcontext()
    started = time.monotonic()
    with context:
        if not initial.exists():
            parent = protocol["parents"][str(replica)]
            branch_checkpoint(parent["path"], initial, config, parent_sha256=parent["sha256"])
        # A committed numbered checkpoint may precede a crash while refreshing
        # the latest alias. Prefer that complete commit over the stale alias.
        numbered = list(Path(config.save_dir).glob("ppo_update*.pt"))
        resume = max(numbered, key=lambda p: int(p.stem.removeprefix("ppo_update"))) if numbered else initial
        train_ppo(config, resume_path=str(resume))
    atomic_json(
        directory / f"leg_to{endpoint}.json",
        {
            "elapsed_seconds": time.monotonic() - started,
            "device": device,
            "target_update": 42 + endpoint,
            "checkpoint_sha256": sha256(latest),
        },
    )


def checkpoint_for(root, protocol, arm, replica, endpoint):
    if endpoint == 0:
        return Path(protocol["parents"][str(replica)]["path"])
    if endpoint not in ENDPOINTS:
        raise ValueError("Unknown study endpoint")
    return Path(root) / f"{arm}_seed{replica}" / "checkpoints" / f"ppo_update{42 + endpoint}.pt"


def validate_endpoint_payload(payload, protocol, arm, replica, endpoint):
    if payload["update_count"] != 42 + endpoint:
        raise ValueError("Checkpoint is not the requested study endpoint")
    if endpoint:
        config = config_from_dict(protocol["recipes"][f"{arm}_seed{replica}"]["config"])
        config.device = payload["cohort_ppo_config"]["device"]  # Explicit execution backend override.
        validate_cohort_resume(config, payload)
        state = payload["cohort_state"]
        if (
            state["initial_update_count"] != 42
            or state["seed_cursor"] != endpoint * config.cohort_games
            or state["optimizer_examples"] != endpoint * config.cohort_training_rows * config.ppo_epochs
        ):
            raise ValueError("Endpoint does not have the predefined cohort and optimizer exposure")


def evaluate_endpoint(root, protocol, endpoint, *, stage="development"):
    root = Path(root)
    if endpoint not in ENDPOINTS:
        raise ValueError("Unknown study endpoint")
    seeds = protocol[f"{stage}_seeds"]
    selection = None
    if stage == "confirmation":
        selection = read_json(root / "candidate_selection.json")
        if endpoint != selection["endpoint"] or read_json(root / "compute.json")["limit_seconds"] <= 0:
            raise ValueError("Confirmation requires frozen candidates and an assigned study budget")
    torch.set_num_threads(4)
    data = load_game_data()
    vocab = build_vocab(data)
    for arm in ARMS:
        for replica in REPLICAS:
            checkpoint = checkpoint_for(root, protocol, arm, replica, endpoint)
            checkpoint_hash = sha256(checkpoint)
            if selection and checkpoint_hash != selection["checkpoint_hashes"][f"{arm}_seed{replica}"]:
                raise ValueError("Confirmation checkpoint differs from frozen selection")
            model, reward, _payload = load_policy(checkpoint, vocab, torch.device("cpu"))
            validate_endpoint_payload(_payload, protocol, arm, replica, endpoint)
            config = config_from_dict(protocol["recipes"][f"{arm}_seed{replica}"]["config"])
            model._training_reserved_seeds = training_seed_reservations(config)
            for mode, repeat in (("sample0", 0), ("sample1", 1), ("sample2", 2), ("greedy", None)):
                output = root / "evaluations" / stage / f"update{endpoint}" / f"{arm}_seed{replica}" / mode
                if output.exists():
                    manifest = read_json(output / "manifest.json")
                    if (
                        manifest["status"] != "complete"
                        or manifest["checkpoint_sha256"] != checkpoint_hash
                        or manifest["source_sha256"] != protocol["source_sha256"]
                    ):
                        raise ValueError(f"Incomplete or mismatched evaluation artifact: {output}")
                    validate_panel(read_rows(output), seeds, [repeat])
                    continue
                evaluate_artifact(
                    model,
                    reward,
                    data,
                    vocab,
                    seeds,
                    torch.device("cpu"),
                    output,
                    greedy=repeat is None,
                    sampling_seed=repeat or 0,
                    deck=config.deck_key,
                    temperature=config.rollout_temperature,
                    stake=config.stake,
                    win_ante=config.win_ante,
                    batch_size=32,
                    workers=4,
                    cpu_threads=4,
                    max_idle_steps=config.max_no_progress_steps,
                    record_return_forecasts=True,
                    metadata={
                        "checkpoint": str(checkpoint),
                        "checkpoint_sha256": checkpoint_hash,
                        "source_sha256": protocol["source_sha256"],
                        "study_sha256": sha256(root / "manifest.json"),
                        "reservation_basis": "Complete-cohort preparation seed audit",
                    },
                )
    return report_endpoint(root, protocol, endpoint, stage=stage)


def read_rows(directory):
    with (Path(directory) / "outcomes.jsonl").open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def report_endpoint(root, protocol, endpoint, *, stage="development"):
    root = Path(root)
    outcomes, artifacts = {}, {}
    for arm in ARMS:
        outcomes[arm] = {}
        for replica in REPLICAS:
            outcomes[arm][replica] = {"sampled": [], "greedy": []}
            expected_hash = sha256(checkpoint_for(root, protocol, arm, replica, endpoint))
            validate_endpoint_payload(
                load_checkpoint_payload(checkpoint_for(root, protocol, arm, replica, endpoint), "cpu"),
                protocol,
                arm,
                replica,
                endpoint,
            )
            for mode, key in (
                ("sample0", "sampled"),
                ("sample1", "sampled"),
                ("sample2", "sampled"),
                ("greedy", "greedy"),
            ):
                directory = root / "evaluations" / stage / f"update{endpoint}" / f"{arm}_seed{replica}" / mode
                manifest = read_json(directory / "manifest.json")
                if (
                    manifest["status"] != "complete"
                    or manifest["checkpoint_sha256"] != expected_hash
                    or manifest["source_sha256"] != protocol["source_sha256"]
                ):
                    raise ValueError("Report requires complete matching evaluation artifacts")
                outcomes[arm][replica][key].extend(read_rows(directory))
                artifacts[str(directory / "outcomes.jsonl")] = sha256(directory / "outcomes.jsonl")
    result = compare_arms(outcomes, protocol[f"{stage}_seeds"], confirmation=stage == "confirmation")
    failures = [
        str(path)
        for arm in ARMS
        for replica in REPLICAS
        for path in (root / f"{arm}_seed{replica}" / "logs").glob("failure.json")
    ]
    result["training_failures"] = failures
    result["acceptance"]["no_training_failures"] = not failures
    if failures:
        result["verdict"] = "inconclusive"
    result.update(
        endpoint=endpoint,
        stage=stage,
        outcome_artifacts=artifacts,
        study_sha256=sha256(root / "manifest.json"),
        source_sha256=protocol["source_sha256"],
    )
    if stage == "confirmation":
        selection = read_json(root / "candidate_selection.json")
        if selection["endpoint"] != endpoint or read_json(root / "compute.json")["limit_seconds"] <= 0:
            raise ValueError("Confirmation report lacks a frozen selection or study budget")
        for arm in ARMS:
            for replica in REPLICAS:
                if (
                    sha256(checkpoint_for(root, protocol, arm, replica, endpoint))
                    != selection["checkpoint_hashes"][f"{arm}_seed{replica}"]
                ):
                    raise ValueError("Confirmation report checkpoint is not frozen")
    atomic_json(root / f"{stage}_update{endpoint}.json", result)
    rows = []
    for mode in ("sampled", "greedy"):
        entry = result["combined"][mode]
        delta, interval = entry["paired"]["sampled_or_greedy_delta"], entry["paired"]["paired_ci95"]
        rows.append(
            f"| {mode} | {entry['complete_episode']['win_rate']:.2%} "
            f"({entry['complete_episode']['wins']}/{entry['complete_episode']['games']}) | "
            f"{entry['completion_window']['win_rate']:.2%} "
            f"({entry['completion_window']['wins']}/{entry['completion_window']['games']}) | {100 * delta:+.2f} "
            f"[{100 * interval[0]:+.2f}, {100 * interval[1]:+.2f}] |"
        )
    (root / f"{stage}_update{endpoint}.md").write_text(
        f"Complete-cohort outcome-label comparison: {result['verdict']}.\n\n"
        "| Policy | Complete labels (wins/games) | Completion-window labels (wins/games) | "
        "Paired change [95% CI], pp |\n"
        "| --- | --- | --- | --- |\n" + "\n".join(rows) + "\n\n"
        "Intervals resample game seeds after averaging sampling repeats and the two training replicas. "
        "They condition on these replicas. Calibration and implementation checks do not establish gameplay gains.\n"
    )
    return result


def freeze_selection(root, protocol):
    root = Path(root)
    if (root / "candidate_selection.json").exists():
        raise FileExistsError("Candidate selection is already frozen")
    if (root / "evaluations/confirmation").exists():
        raise ValueError("Cannot select a candidate after confirmation evaluation has started")
    # Recompute from both predefined endpoints. No choosing which result files
    # to include after observing confirmation.
    candidates = []
    for endpoint in (10, 20):
        report = report_endpoint(root, protocol, endpoint)
        sampled = report["combined"]["sampled"]["paired"]["sampled_or_greedy_delta"]
        greedy = report["combined"]["greedy"]["paired"]["sampled_or_greedy_delta"]
        if (
            sampled > 0
            and greedy > -0.05
            and report["acceptance"]["both_replicas_positive"]
            and report["acceptance"]["no_gameplay_failures"]
            and report["acceptance"]["no_training_failures"]
        ):
            candidates.append((sampled, greedy, endpoint))
    if not candidates:
        raise ValueError("No development candidate passes the predefined confirmation-entry gate")
    endpoint = max(candidates)[2]
    selection = {
        "endpoint": endpoint,
        "checkpoint_hashes": {
            f"{arm}_seed{replica}": sha256(checkpoint_for(root, protocol, arm, replica, endpoint))
            for arm in ARMS
            for replica in REPLICAS
        },
        "source_sha256": protocol["source_sha256"],
        "reason": "Largest paired sampled gain across both replicas; greedy gain then later endpoint break ties",
        "development_reports": {str(n): sha256(root / f"development_update{n}.json") for n in (10, 20)},
    }
    atomic_json(root / "candidate_selection.json", selection)
    return selection


def diagnose_labels(cohort_path, checkpoint, destination, *, updates=100):
    """Frozen-representation fitting isolates missingness; it does not train PPO."""
    destination = Path(destination)
    if destination.exists() or updates < 1:
        raise ValueError("Diagnostic requires a new output directory and positive update count")
    episodes, draw, manifest = load_cohort(cohort_path)
    if len(episodes) < 4:
        raise ValueError("Diagnostic requires at least four completed games for a game-level split")
    data = load_game_data()
    model, _reward, _payload = load_policy(checkpoint, build_vocab(data), torch.device("cpu"))
    if {episode.policy_sha256 for episode in episodes} != {policy_digest(model)}:
        raise ValueError("Diagnostic checkpoint must be the exact frozen collection policy")
    torch.set_num_threads(4)
    config = config_from_dict(manifest["ppo_config"])
    buffer = CohortBuffer(episodes, config, draw_indices=draw)
    pooled = []
    handle = model.value_head.register_forward_pre_hook(
        lambda _module, args, kwargs: pooled.append(pooled_critic_inputs(args, kwargs).detach().cpu()),
        with_kwargs=True,
    )
    try:
        with torch.no_grad():
            for start in range(0, buffer.size, 320):
                _critic_predictions(model, buffer.materialize(np.arange(start, min(start + 320, buffer.size)), "cpu"))
    finally:
        handle.remove()
    x = torch.cat(pooled)
    shuffled_games = np.random.default_rng(1729).permutation([episode.seed for episode in episodes])
    train_games = shuffled_games[: len(episodes) // 2]
    train_mask = np.isin(buffer.arrays["seed"], train_games)

    def dataset(mask):
        indices = np.flatnonzero(mask)
        return {
            "x": x[indices],
            "ante": torch.as_tensor(buffer.arrays["ante"][indices]),
            "target": torch.as_tensor(buffer.arrays["terminal_outcome_target"][indices]),
            "returns": torch.as_tensor(
                (buffer.arrays["terminal_outcome_target"][indices] == 8).astype(np.float32) * 10
            ),
            "step": torch.as_tensor(buffer.arrays["step"][indices]),
            "seed": torch.as_tensor(buffer.arrays["seed"][indices]),
        }

    training, validation = dataset(train_mask), dataset(~train_mask)
    available = buffer.arrays["completion_mask"][train_mask].astype(bool)
    destination.mkdir(parents=True)
    results = {
        "cohort_sha256": sha256(Path(cohort_path) / "manifest.json"),
        "checkpoint_sha256": sha256(checkpoint),
        "training_seeds": train_games.tolist(),
        "validation_seeds": shuffled_games[len(episodes) // 2 :].tolist(),
        "device": "cpu",
        "updates": updates,
        "actor_updated": False,
        "encoder_updated": False,
        "optimizer": {"name": "AdamW", "lr": 3e-4, "weight_decay": 0.01, "batch": 320},
        "interpretation": "Frozen-data label-selection diagnostic; validation games excluded from this fit. "
        "This is not an independent gameplay improvement test.",
        "replicas": {},
    }
    for replica in REPLICAS:
        rng = np.random.default_rng(1729 + replica)
        random_mask = np.zeros(len(available), bool)
        random_mask[rng.choice(len(available), size=int(available.sum()), replace=False)] = True
        order = rng.integers(len(available), size=(updates, 320))
        results["replicas"][str(replica)] = {"ordering_sha256": sha256_bytes(order), "arms": {}}
        for name, mask in (
            ("complete", np.ones(len(available), bool)),
            ("completion", available),
            ("random", random_mask),
        ):
            head = copy.deepcopy(model.value_head)
            for parameter in head.parameters():
                parameter.requires_grad_(False)
            parameters = [p for module in (head.outcome_proj, head.ante_survival) for p in module.parameters()]
            for parameter in parameters:
                parameter.requires_grad_(True)
            optimizer = torch.optim.AdamW(parameters, lr=3e-4, weight_decay=0.01)
            curve = []
            labels = 0
            for update in range(updates + 1):
                if update in {0, updates // 2, updates}:
                    with torch.no_grad():
                        metrics = critic_metrics(head, validation)
                    curve.append({"update": update, "labels_processed": labels, "validation": metrics})
                if update == updates:
                    break
                indices = order[update]
                values = critic_from_pooled(head, training["x"][indices], training["ante"][indices])
                loss = outcome_nll(
                    values["outcome_probabilities"], training["target"][indices], torch.as_tensor(mask[indices])
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
                optimizer.step()
                labels += int(mask[indices].sum())
            state_path = destination / f"{name}_seed{replica}.pt"
            torch.save(head.state_dict(), state_path)
            results["replicas"][str(replica)]["arms"][name] = {
                "curve": curve,
                "available_labels": int(mask.sum()),
                "physical_rows": int(order.size),
                "mask_sha256": sha256_bytes(mask),
                "state_sha256": sha256(state_path),
            }
    atomic_json(destination / "manifest.json", results)
    return results


def sha256_bytes(array):
    import hashlib

    return hashlib.sha256(np.asarray(array).tobytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Audit and freeze the protocol; do not start training")
    prepare.add_argument("--prior", type=Path, default=PRIOR_ROOT)
    commands.add_parser("verify-cpu", help="Run bounded tests and retain their smoke artifacts")
    budget = commands.add_parser("budget", help="Assign a newly authorized accelerator allowance")
    budget.add_argument("--hours", type=float, required=True)
    train = commands.add_parser("train", help="Explicitly launch one matched training arm")
    train.add_argument("--arm", choices=ARMS, required=True)
    train.add_argument("--replica", type=int, choices=REPLICAS, required=True)
    train.add_argument("--endpoint", type=int, choices=(10, 20), required=True)
    train.add_argument("--device", choices=("cpu", "mps"), default="mps")
    train.add_argument("--estimate-seconds", type=float, required=True)
    for command in ("evaluate", "report"):
        sub = commands.add_parser(command)
        sub.add_argument("--endpoint", type=int, choices=ENDPOINTS, required=True)
        sub.add_argument("--stage", choices=("development", "confirmation"), default="development")
    commands.add_parser("freeze", help="Freeze the best eligible development checkpoint pair")
    diagnose = commands.add_parser("diagnose", help="Fit fixed-data label controls on CPU")
    diagnose.add_argument("--cohort", type=Path, required=True)
    diagnose.add_argument("--checkpoint", type=Path, required=True)
    diagnose.add_argument("--output", type=Path, required=True)
    diagnose.add_argument("--updates", type=int, default=100)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        prepare_study(args.root, prior=args.prior)
        print(f"Prepared {args.root}; accelerator allowance is zero and no training started.")
        return
    protocol = load_protocol(args.root)
    if args.command == "verify-cpu":
        verify_cpu(args.root)
    elif args.command == "budget":
        assign_budget(args.root, args.hours)
    elif args.command == "train":
        train_arm(
            args.root,
            protocol,
            args.arm,
            args.replica,
            args.endpoint,
            device=args.device,
            estimate_seconds=args.estimate_seconds,
        )
    elif args.command == "evaluate":
        evaluate_endpoint(args.root, protocol, args.endpoint, stage=args.stage)
    elif args.command == "report":
        report_endpoint(args.root, protocol, args.endpoint, stage=args.stage)
    elif args.command == "freeze":
        freeze_selection(args.root, protocol)
    elif args.command == "diagnose":
        diagnose_labels(args.cohort, args.checkpoint, args.output, updates=args.updates)
    print(f"{args.command} complete: {args.root}")


if __name__ == "__main__":
    main()
