#!/usr/bin/env python3
"""Verify study integrity and recompute confirmation acceptance from complete raw panels."""

import argparse
import json
import math
import runpy
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from pylatro_agent.transfer import CONFIRMATION_SEEDS, CORRECTION_SEEDS, sha256, summarize


def read(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_new(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def finite(value):
    if isinstance(value, float):
        require(math.isfinite(value), "Non-finite outcome field")
    elif isinstance(value, dict):
        for item in value.values():
            finite(item)
    elif isinstance(value, list):
        for item in value:
            finite(item)


def panel(path, sampling_seed, checkpoint):
    manifest = read(path / "manifest.json")
    rows = [json.loads(line) for line in (path / "outcomes.jsonl").read_text().splitlines()]
    require(manifest["status"] == "complete", f"Incomplete panel: {path}")
    require(manifest["seeds"] == list(CONFIRMATION_SEEDS), "Wrong confirmation reservation/order")
    require(manifest["checkpoint_sha256"] == sha256(checkpoint), "Wrong confirmation checkpoint")
    require(manifest["greedy"] == (sampling_seed is None), "Wrong confirmation action-selection mode")
    require(manifest["sampling_seed"] == sampling_seed, "Wrong confirmation manifest sampling seed")
    require(manifest["kind"] == "fresh_start", "Confirmation must start fresh games")
    for key, expected in {
        "deck": "b_blue",
        "stake": 1,
        "win_ante": 8,
        "temperature": 1.0,
        "batch_size": 32,
        "workers": 4,
        "device": "mps",
    }.items():
        require(manifest[key] == expected, f"Wrong confirmation setting: {key}")
    require(manifest["agent_config"]["hand_ar_mixture_eps"] == 0.5, "Wrong confirmation mixture")
    require({row["seed"] for row in rows} == set(CONFIRMATION_SEEDS), "Missing confirmation seeds")
    require(all(row["sampling_seed"] == sampling_seed for row in rows), "Wrong sampling seed")
    require(summarize(rows)["games"] == 400, "Wrong confirmation denominator")
    finite(rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=Path("cache/datasets/5k-balatro-games"))
    parser.add_argument("--record-corrections", action="store_true", help="Record hashes before any trial starts")
    parser.add_argument(
        "--training-only", action="store_true", help="Audit completed training before confirmation exists"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    trials = args.root / "continuations"
    corrections = trials / "corrections"
    hash_file = trials / "correction_data_hashes.json"
    correction_manifest = read(corrections / "manifest.json")
    require(correction_manifest["status"] == "complete", "Correction collection is incomplete")
    if args.record_corrections:
        require(not list(trials.glob("*_seed*/manifest.json")), "Correction hashes must be recorded before training")
        write_new(hash_file, {str(path): sha256(path) for path in sorted(corrections.glob("*.parquet"))})
        print(f"Recorded correction hashes: {hash_file}")
        return
    require(args.output is not None, "Verification requires a new --output file")
    for path, expected in read(hash_file).items():
        require(sha256(path) == expected, f"Correction data changed after collection: {path}")
    generation = read(args.dataset / "metadata/summary.json")
    for path, expected in read(args.root / "diagnostics_v2/manifest.json")["data_hashes"].items():
        require(sha256(path) == expected, f"Original dataset file changed: {path}")
    for shard in generation["shards"]:
        require(sha256(args.dataset / shard["file"]) == shard["sha256"], "Original HF shard changed")
    for epoch in range(1, 11):
        manifest = read(args.root / "development" / f"epoch{epoch}_sample0_dev50" / "manifest.json")
        require(sha256(manifest["checkpoint"]) == manifest["checkpoint_sha256"], "Original checkpoint changed")
    selected_hash = sha256(read(args.root / "development/selection.json")["checkpoint"])
    prefixes = read(args.root / "current_prefix_reward_equivalence_v3_events.json")["results"]
    require(len(prefixes) == 20 and sum(row["steps"] for row in prefixes) == 476, "Incomplete reward-prefix audit")
    require(all(row["first_fast_env"] is None for row in prefixes), "Unresolved generator/environment reward mismatch")
    training = {}
    for arm in ("control", "actor_only", "correction"):
        for seed in (0, 1):
            manifest = read(trials / f"{arm}_seed{seed}/manifest.json")
            require(manifest["status"] == "complete", "Incomplete training arm")
            require(
                manifest["updates_completed"] == 1000 and manifest["examples_processed"] == 320000,
                "Wrong training update/example denominator",
            )
            require(
                manifest["physical_batch"] == 320 and manifest["precision"] == "fp32", "Wrong trial batch/precision"
            )
            require(manifest["source_checkpoint_sha256"] == selected_hash, "Unmatched starting checkpoint")
            require(manifest["training_seed"] == seed and manifest["arm"] == arm, "Wrong training replicate")
            require(manifest["effective_value_loss_coeff"] == (0.5 if arm == "control" else 0), "Wrong critic loss")
            require(manifest["effective_outcome_loss_coeff"] == (0.1 if arm == "control" else 0), "Wrong outcome loss")
            for key, expected in {
                "lr": 3e-5,
                "weight_decay": 0.01,
                "warmup_steps": 100,
                "batch_size": 320,
                "hand_ar_mixture_eps": 0.5,
                "action_entropy_coeff": 0.001,
                "outcome_weight_beta": 1.5,
                "bc_weight_min": 0.25,
                "bc_weight_max": 4.0,
                "device": "mps",
                "deck_key": "b_blue",
                "stake": 1,
            }.items():
                require(manifest["configuration"][key] == expected, f"Wrong training setting: {key}")
            require(manifest["hf_fraction"] == (0.5 if arm == "correction" else 1), "Wrong HF/correction mixture")
            learning = [
                json.loads(line)
                for line in (trials / f"{arm}_seed{seed}/learning.jsonl").read_text().splitlines()
            ]
            finite(learning)
            require(learning[-1]["update"] == 1000, "Learning curve is incomplete")
            training[arm, seed] = manifest
    for seed in (0, 1):
        require(
            training["control", seed]["hf_order_sha256"] == training["actor_only", seed]["hf_order_sha256"],
            "Control and actor-only HF ordering differed",
        )
    forbidden = {"won", "reward", "max_ante", "return_target", "terminal_outcome_target", "terminal_outcome_mask"}
    seen = set()
    for path in sorted(corrections.glob("*.parquet")):
        file = pq.ParquetFile(path)
        require(not forbidden.intersection(file.schema_arrow.names), "Teacher corrections contain outcome targets")
        for batch in file.iter_batches(
            batch_size=256,
            columns=[
                "obs.action_mask",
                "action",
                "executed_student_action",
                "seed",
                "step",
                "bc_weight",
                "student_checkpoint_sha256",
            ],
        ):
            columns = {name: batch.column(i) for i, name in enumerate(batch.schema.names)}
            masks = columns["obs"].field("action_mask").flatten().to_numpy().reshape(batch.num_rows, -1)
            for key in ("action", "executed_student_action"):
                actions = columns[key].to_numpy()
                require(np.all((actions >= 0) & (actions < masks.shape[1])), "Invalid correction action range")
                require(np.all(masks[np.arange(batch.num_rows), actions]), "Illegal recorded correction/student action")
            require(np.all(columns["bc_weight"].to_numpy() == 1), "Non-unit correction BC weight")
            require(set(columns["student_checkpoint_sha256"].to_pylist()) == {selected_hash}, "Wrong label provenance")
            for seed, step in zip(columns["seed"].to_pylist(), columns["step"].to_pylist(), strict=True):
                require(
                    seed in CORRECTION_SEEDS and (seed, step) not in seen, "Invalid or duplicate correction seed/step"
                )
                seen.add((seed, step))
    require(len(seen) == correction_manifest["examples"], "Missing correction examples")
    require({seed for seed, _ in seen} == set(CORRECTION_SEEDS), "Incomplete correction game panel")
    for seed in CORRECTION_SEEDS:
        steps = sorted(step for game, step in seen if game == seed)
        require(steps == list(range(len(steps))), "Missing correction decision steps")
    if args.training_only:
        write_new(
            args.output,
            {
                "status": "passed",
                "scope": "original artifact integrity, correction provenance, and six completed training runs only",
                "original_checkpoints_unchanged": 10,
                "original_hf_shards_unchanged": len(generation["shards"]),
                "verified_correction_rows": len(seen),
                "training_runs": 6,
                "examples_processed": 1920000,
                "confirmation_evaluated": False,
            },
        )
        print(f"Training verification passed: {args.output}")
        return
    selection = read(trials / "confirmation_selection.json")
    require(selection["selected_before_confirmation"], "Missing pre-confirmation selection")
    candidate = selection["arm"]
    results = {}
    for arm in (candidate, "control"):
        for seed in (0, 1):
            results[arm, seed] = {
                "sampled": [
                    row
                    for sample in (0, 1, 2)
                    for row in panel(
                        trials / f"confirmation_{arm}_seed{seed}_sample{sample}",
                        sample,
                        trials / f"{arm}_seed{seed}/checkpoint.pt",
                    )
                ],
                "greedy": panel(
                    trials / f"confirmation_{arm}_seed{seed}_greedy", None, trials / f"{arm}_seed{seed}/checkpoint.pt"
                ),
            }
    acceptance = runpy.run_path("tools/continue_transfer.py")["acceptance"]
    recomputed = acceptance(
        *[
            {seed: results[arm, seed][mode] for seed in (0, 1)}
            for mode in ("sampled", "greedy")
            for arm in (candidate, "control")
        ],
        confirmation=True,
    )
    saved = read(trials / "confirmation_result.json")
    require(
        recomputed["status"] == saved["status"] and recomputed["paired"] == saved["paired"],
        "Saved confirmation result disagrees with raw paired outcomes",
    )
    budget = read(args.root / "accelerator_budget.json")
    compute = read(trials / "compute.json")
    accounted = budget["diagnostic_accelerator_seconds"] + compute["total_elapsed_seconds"] + 360
    require(accounted <= 12 * 3600, "Accelerator budget exceeded")
    write_new(
        args.output,
        {
            "status": "passed",
            "confirmation": recomputed,
            "original_checkpoints_unchanged": 10,
            "original_hf_shards_unchanged": len(generation["shards"]),
            "verified_correction_rows": len(seen),
            "training_runs": 6,
            "examples_processed": 1920000,
            "confirmation_games": 6400,
            "accounted_accelerator_seconds_including_overhead_reserve": accounted,
        },
    )
    print(f"Verification passed: {args.output}")


if __name__ == "__main__":
    main()
