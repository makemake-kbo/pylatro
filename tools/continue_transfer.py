#!/usr/bin/env python3
"""Collect teacher corrections, run matched continuation arms, select on development, confirm once."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from pylatro import load_game_data
from pylatro_agent.training.model_generate import load_records
from pylatro_agent.transfer import (
    CONFIRMATION_SEEDS,
    CORRECTION_SEEDS,
    DEVELOPMENT_SEEDS,
    SAMPLING_SEEDS,
    check_seed_separation,
    evaluate_artifact,
    load_policy,
    paired_statistics,
    sha256,
    source_hashes,
    summarize,
    write_json,
)
from pylatro_agent.transfer_collection import collect_corrections
from pylatro_agent.transfer_training import train_continuation
from pylatro_agent.vocab import build_vocab


def acceptance(sampled_candidate, sampled_control, greedy_candidate, greedy_control, *, confirmation=False):
    """Average sampling and training replicates within each game before bootstrap."""
    panels = (sampled_candidate, sampled_control, greedy_candidate, greedy_control)
    if any(row.get("stalled") or row.get("censored") for panel in panels for rows in panel.values() for row in rows):
        raise ValueError("Cannot assess stalled or censored games as complete policy results")
    combined = {}
    replicate_deltas = []
    for mode, candidates, controls in [
        ("sampled", sampled_candidate, sampled_control),
        ("greedy", greedy_candidate, greedy_control),
    ]:
        a, b = [], []
        for seed in (0, 1):
            delta = paired_statistics(candidates[seed], controls[seed])
            if mode == "sampled":
                replicate_deltas.append(delta["sampled_or_greedy_delta"])
            a.extend(
                {**row, "sampling_seed": f"train{seed}/sample{row.get('sampling_seed')}"} for row in candidates[seed]
            )
            b.extend(
                {**row, "sampling_seed": f"train{seed}/sample{row.get('sampling_seed')}"} for row in controls[seed]
            )
        combined[mode] = paired_statistics(a, b)
    sampled = combined["sampled"]
    greedy = combined["greedy"]
    validated = (
        (
            sampled["sampled_or_greedy_delta"] >= 0.05
            or math.isclose(sampled["sampled_or_greedy_delta"], 0.05, rel_tol=0, abs_tol=1e-12)
        )
        and sampled["paired_ci95"][0] > 0
        and min(replicate_deltas) > 0
        and (
            greedy["paired_ci95"][0] >= -0.05 or math.isclose(greedy["paired_ci95"][0], -0.05, rel_tol=0, abs_tol=1e-12)
        )
    )
    status = (
        "validated"
        if validated
        else (
            "negative"
            if sampled["paired_ci95"][1] < 0
            else "promising"
            if sampled["sampled_or_greedy_delta"] > 0
            else "inconclusive"
        )
    )
    if not confirmation:
        status = "development_only"
    return {
        "status": status,
        "criteria_met": validated,
        "paired": combined,
        "sampled_training_replicate_deltas": replicate_deltas,
        "criteria": {
            "sampled_minimum_gain": 0.05,
            "sampled_ci_lower_positive": True,
            "both_training_replicates_positive": True,
            "greedy_ci_lower_minimum": -0.05,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--improvement-hours", type=float, default=6.0)
    parser.add_argument("--confirmation-hours", type=float, default=3.0)
    args = parser.parse_args()
    import signal

    def deadline(signum, frame):
        raise TimeoutError("Total improvement/confirmation budget exhausted")

    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(int(args.improvement_hours * 3600))
    selected = json.loads(args.selection.read_text())
    checkpoint = Path(selected["checkpoint"])
    args.output_dir.mkdir(parents=True, exist_ok=False)
    data = load_game_data()
    vocab = build_vocab(data)
    device = torch.device(args.device)
    start = time.monotonic()
    manifests = [
        args.dataset / "metadata/generation.json",
        args.dataset / "metadata/games.jsonl",
        Path("docs/heuristic/pretraining_v68_seeds.json"),
    ]
    separation = check_seed_separation(DEVELOPMENT_SEEDS + CONFIRMATION_SEEDS + CORRECTION_SEEDS, manifests)
    model, reward, _ = load_policy(checkpoint, vocab, device, mixture=0.5)
    provenance = {
        "selection_sha256": sha256(args.selection),
        "source_hashes": source_hashes(),
        "seed_separation": separation,
        "training_source": str(args.dataset),
        "data_hashes": {str(p): sha256(p) for p in sorted(args.dataset.rglob("*")) if p.is_file()},
    }
    write_json(args.output_dir / "provenance.json", provenance)
    # Agreement comparisons use the selected checkpoint on both state panels.
    from pylatro_agent.transfer_diagnostics import offline_predictions, selected_records

    diagnostic_manifest = args.output_dir.parent / "diagnostics_v2/manifest.json"
    diagnostic_seeds = json.loads(diagnostic_manifest.read_text())["offline_seeds"]
    hf_records = load_records(args.dataset, reward_config=reward)
    hf_selection = selected_records(hf_records, diagnostic_seeds)
    offline_output = args.output_dir / "selected_offline"
    offline_output.mkdir()
    offline_predictions(model, [row for seed in diagnostic_seeds for row in hf_selection[seed]], device, offline_output)
    del hf_selection, hf_records
    # Drop decoded diagnostic buffers before allocating the physical training batch.
    import gc

    import pyarrow as pa

    gc.collect()
    pa.default_memory_pool().release_unused()
    if device.type == "mps":
        torch.mps.empty_cache()
    corrections = collect_corrections(model, reward, data, vocab, checkpoint, device, args.output_dir / "corrections")
    del model
    if device.type == "mps":
        torch.mps.empty_cache()
    import subprocess
    import sys

    with (args.output_dir.parent / "interventions.log").open("x") as log:
        subprocess.run(
            [
                sys.executable,
                "-u",
                "tools/intervene_transfer.py",
                "--checkpoint",
                str(checkpoint),
                "--snapshots",
                str(args.output_dir / "corrections/snapshots.json"),
                "--output-dir",
                str(args.output_dir.parent / "interventions"),
                "--device",
                str(device),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=max(60, args.improvement_hours * 3600 - (time.monotonic() - start)),
        )
    records = load_records(args.dataset, reward_config=reward)
    checkpoints = {}
    ledger = []
    results = {}
    for seed in (0, 1):
        for arm in ("control", "actor_only", "correction"):
            remaining = args.improvement_hours * 3600 - (time.monotonic() - start)
            estimated_trial = max((r["elapsed_seconds"] for r in ledger), default=2400)
            if remaining < estimated_trial + 5400:
                raise TimeoutError("Cannot start a full continuation while preserving evaluation/confirmation reserve")
            began = time.monotonic()
            checkpoints[arm, seed] = train_continuation(
                checkpoint,
                records,
                vocab,
                device,
                args.output_dir / f"{arm}_seed{seed}",
                arm=arm,
                seed=seed,
                corrections=corrections if arm == "correction" else None,
                max_seconds=min(remaining - 5400, 3600),
            )
            ledger.append({"arm": arm, "seed": seed, "elapsed_seconds": time.monotonic() - began})
            write_json(
                args.output_dir / "compute.json",
                {"training_runs": ledger, "improvement_elapsed_seconds": time.monotonic() - start},
            )
            if device.type == "mps":
                torch.mps.empty_cache()

    def evaluate(arm, seed, panel, label):
        model, reward, _ = load_policy(checkpoints[arm, seed], vocab, device)
        sampled = []
        for sampling_seed in SAMPLING_SEEDS:
            sampled.extend(
                evaluate_artifact(
                    model,
                    reward,
                    data,
                    vocab,
                    panel,
                    device,
                    args.output_dir / f"{label}_{arm}_seed{seed}_sample{sampling_seed}",
                    sampling_seed=sampling_seed,
                    metadata={
                        **provenance,
                        "checkpoint": str(checkpoints[arm, seed]),
                        "checkpoint_sha256": sha256(checkpoints[arm, seed]),
                        "training_seed": seed,
                    },
                )
            )
        greedy = evaluate_artifact(
            model,
            reward,
            data,
            vocab,
            panel,
            device,
            args.output_dir / f"{label}_{arm}_seed{seed}_greedy",
            greedy=True,
            metadata={
                **provenance,
                "checkpoint": str(checkpoints[arm, seed]),
                "checkpoint_sha256": sha256(checkpoints[arm, seed]),
                "training_seed": seed,
            },
        )
        del model
        if device.type == "mps":
            torch.mps.empty_cache()
        return {"sampled": sampled, "greedy": greedy}

    for arm in ("control", "actor_only", "correction"):
        for seed in (0, 1):
            results[arm, seed] = evaluate(arm, seed, DEVELOPMENT_SEEDS, "development")
    candidates = []
    for arm in ("actor_only", "correction"):
        paired = acceptance(
            *[
                {seed: results[name, seed][mode] for seed in (0, 1)}
                for mode in ("sampled", "greedy")
                for name in (arm, "control")
            ]
        )
        candidates.append({"arm": arm, **paired})
    write_json(args.output_dir / "development_comparisons.json", candidates)
    write_json(
        args.output_dir / "correction_ablation.json",
        {
            "candidate_arm": "correction",
            "reference_arm": "actor_only",
            **acceptance(
                *[
                    {seed: results[name, seed][mode] for seed in (0, 1)}
                    for mode in ("sampled", "greedy")
                    for name in ("correction", "actor_only")
                ]
            ),
        },
    )
    candidate = max(candidates, key=lambda r: r["paired"]["sampled"]["sampled_or_greedy_delta"])["arm"]
    write_json(
        args.output_dir / "confirmation_selection.json",
        {
            "arm": candidate,
            "selected_before_confirmation": True,
            "rule": "highest paired sampled development gain over continuation control",
            "confirmation_seeds": list(CONFIRMATION_SEEDS),
            "sampling_seeds": list(SAMPLING_SEEDS),
        },
    )
    confirmation_started = time.monotonic()
    signal.alarm(int(args.confirmation_hours * 3600))
    confirmed = {}
    for arm in (candidate, "control"):
        for seed in (0, 1):
            if time.monotonic() - confirmation_started > args.confirmation_hours * 3600 - 900:
                raise TimeoutError("Confirmation budget exhausted; no acceptance claim allowed")
            confirmed[arm, seed] = evaluate(arm, seed, CONFIRMATION_SEEDS, "confirmation")
    result = acceptance(
        *[
            {seed: confirmed[name, seed][mode] for seed in (0, 1)}
            for mode in ("sampled", "greedy")
            for name in (candidate, "control")
        ],
        confirmation=True,
    )
    result["candidate_arm"] = candidate
    result["per_arm"] = {
        f"{arm}/seed{seed}/{mode}": summarize(confirmed[arm, seed][mode])
        for arm in (candidate, "control")
        for seed in (0, 1)
        for mode in ("sampled", "greedy")
    }
    write_json(args.output_dir / "confirmation_result.json", result)
    write_json(
        args.output_dir / "compute.json",
        {
            "training_runs": ledger,
            "confirmation_elapsed_seconds": time.monotonic() - confirmation_started,
            "total_elapsed_seconds": time.monotonic() - start,
        },
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
