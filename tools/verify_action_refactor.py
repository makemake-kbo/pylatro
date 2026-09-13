#!/usr/bin/env python3
"""Audit preserved inputs, trained adapters, completed evaluations, and compute accounting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pylatro_agent.learning_bottleneck import atomic_json, read_json
from pylatro_agent.transfer import check_seed_separation, sha256, summarize


def verify(root, *, final=False):
    errors, verified = [], {}

    def require(condition, message):
        if not condition:
            errors.append(message)

    preserved = read_json(Path("eval/learning_bottleneck_20260913/original_hashes.json"))
    for name, expected in preserved.items():
        require(Path(name).exists() and sha256(Path(name)) == expected, f"Original artifact changed: {name}")
    verified["original_artifacts"] = len(preserved)
    manifest = read_json(root / "manifest.json")
    seeds = manifest["development_seeds"] + manifest["confirmation_seeds"]
    try:
        check_seed_separation(seeds, [Path(item["path"]) for item in manifest["seed_separation"]])
    except ValueError as error:
        errors.append(str(error))
    for item in manifest["seed_separation"]:
        require(sha256(item["path"]) == item["sha256"], f"Seed manifest changed: {item['path']}")
    data_hashes = read_json(root / "subset_manifest.json")["data"]
    shards = 0
    for split, expected in data_hashes.items():
        directory = Path("eval/learning_bottleneck_20260913/data") / split
        require(sha256(directory / "summary.json") == expected, f"Current-rule data summary changed: {split}")
        for name, digest in read_json(directory / "summary.json")["file_hashes"].items():
            require(sha256(directory / name) == digest, f"Current-rule data file changed: {split}/{name}")
            shards += 1
    verified["current_rule_data_files"] = shards
    for cache_name, prefix in (("cache_complete.json", ""), ("subset_cache_complete.json", "subset_")):
        cache = read_json(root / cache_name)
        for split, expected in cache["hashes"].items():
            path = root / f"{prefix}{split}.pt"
            require(sha256(path) == expected, f"Representation cache changed: {path}")
    source = manifest["checkpoint_sha256"]
    for seed in (0, 1):
        sale = [
            read_json(root / f"{variant}_seed{seed}" / "complete.json")
            for variant in ("pooled_sale_residual_v1", "entity_sale_residual_v1")
        ]
        require(sale[0]["ordering_sha256"] == sale[1]["ordering_sha256"], f"Sale training order differs: seed {seed}")
        require(sale[0]["cache_hashes"] == sale[1]["cache_hashes"], f"Sale cache differs: seed {seed}")
        for item in sale:
            require(item["source_checkpoint_sha256"] == source, f"Wrong sale source: seed {seed}")
            require(
                item["examples_processed"] == 640000 and item["updates"] == 2000,
                f"Wrong sale update count: seed {seed}",
            )
        hand = read_json(root / f"subset_residual_v1_seed{seed}" / "adapter_complete.json")
        require(hand["source_checkpoint_sha256"] == source, f"Wrong hand source: seed {seed}")
        require(
            hand["examples_processed"] == 640000 and hand["updates"] == 2000, f"Wrong hand update count: seed {seed}"
        )
        require(hand["accelerator_seconds"] == 0, f"Hand adapter training unexpectedly used GPU: seed {seed}")
    verified["training_trials"] = 6
    verified["training_examples_processed"] = 6 * 640000
    controls = root / "hand_exposure_control"
    verified["post_hoc_control_trials"] = 0
    verified["post_hoc_control_examples_processed"] = 0
    if (controls / "initial_policy_agreement.json").exists():
        for seed in (0, 1):
            directory = controls / f"pure_ar_seed{seed}"
            if not (directory / "complete.json").exists():
                require(not final, f"Post-hoc CPU control is incomplete: seed {seed}")
                continue
            control = read_json(directory / "complete.json")
            protocol = read_json(directory / "manifest.json")
            hand = read_json(root / f"subset_residual_v1_seed{seed}" / "adapter_complete.json")
            require(control["ordering_sha256"] == hand["ordering_sha256"], f"Hand exposure differs: seed {seed}")
            require(control["examples_processed"] == 640000, f"Control exposure differs: seed {seed}")
            require(control["accelerator_seconds"] == 0, f"Control used unexpected GPU time: seed {seed}")
            require(protocol["source_checkpoint_sha256"] == source, f"Control source differs: seed {seed}")
            require(protocol["cache_hashes"] == hand["cache_hashes"], f"Control cache differs: seed {seed}")
            require(not protocol["use_for_confirmation_selection"], "Post-hoc control entered candidate selection")
            require(sha256(directory / "manifest.json") == control["manifest_sha256"], "Control protocol changed")
            require(sha256(directory / "head_best.pt") == control["head_sha256"], "Trained control head changed")
            verified["post_hoc_control_trials"] += 1
            verified["post_hoc_control_examples_processed"] += control["examples_processed"]
    for path in root.glob("*_seed*/complete.json"):
        require(
            sha256(path.parent / "checkpoint.pt") == read_json(path)["checkpoint_sha256"],
            f"Exported checkpoint changed: {path.parent}",
        )
    for family in ("subset", "combined"):
        require(
            sha256(root / f"{family}_neutral_control.pt")
            == read_json(root / f"{family}_neutral_control.json")["checkpoint_sha256"],
            f"Neutral control changed: {family}",
        )
    evaluations, running = [], []
    # rglob does not traverse the symlinked checkpoint folders; each evaluation
    # namespace contains its own physical outcome files.
    for path in sorted(root.rglob("evaluations/**/manifest.json")):
        metadata = read_json(path)
        if "checkpoint_sha256" not in metadata:
            continue
        if metadata.get("status") != "complete":
            running.append(str(path))
            require(not final, f"Incomplete evaluation at final verification: {path}")
            continue
        require(
            sha256(Path(metadata["checkpoint"])) == metadata["checkpoint_sha256"],
            f"Evaluation checkpoint changed: {path}",
        )
        rows = [json.loads(line) for line in (path.parent / "outcomes.jsonl").read_text().splitlines()]
        require(
            len(rows) == len(metadata["seeds"]) and {row["seed"] for row in rows} == set(metadata["seeds"]),
            f"Missing or duplicate evaluation seeds: {path}",
        )
        require(
            all(not row.get("stalled") and not row.get("censored") for row in rows), f"Incomplete trajectory: {path}"
        )
        try:
            actual = summarize(rows)
            expected = read_json(path.parent / "summary.json")
            for key in ("games", "game_seeds", "wins", "win_rate", "mean_final_ante", "stalls", "ante_clear_rates"):
                require(actual[key] == expected[key], f"Metric denominator or outcome mismatch {key}: {path}")
        except ValueError as error:
            errors.append(f"{path}: {error}")
        evaluations.append(dict(path=str(path), games=len(rows), wins=sum(row["won"] for row in rows)))
    verified["completed_evaluations"] = evaluations
    verified["running_evaluations"] = running
    ledger = read_json(root / "compute.json")
    elapsed = sum(job["elapsed_seconds"] for job in ledger["jobs"])
    require(abs(elapsed - ledger["elapsed_seconds"]) < 1e-6, "Compute ledger total differs from job durations")
    require(elapsed <= ledger["limit_seconds"], "Accelerator allowance exceeded")
    if final:
        require(all(job["status"] != "running" for job in ledger["jobs"]), "Accelerator job still running")
    verified["accelerator_seconds_charged"] = elapsed
    verified["accelerator_limit_seconds"] = ledger["limit_seconds"]
    verified["final"] = final
    verified["passed"] = not errors
    verified["errors"] = errors
    atomic_json(root / ("verification_final.json" if final else "verification_progress.json"), verified)
    print({k: v for k, v in verified.items() if k not in ("completed_evaluations", "running_evaluations")})
    if errors:
        raise RuntimeError("Action-refactor verification failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("eval/action_refactor_20260913"))
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()
    verify(args.root, final=args.final)


if __name__ == "__main__":
    main()
