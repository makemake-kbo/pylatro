#!/usr/bin/env python3
"""CPU-only post-hoc partition of the shop oracle; never selects a confirmation candidate."""

import argparse
import json
import multiprocessing as mp
import signal
import time
from dataclasses import asdict
from pathlib import Path

from pylatro_agent.checkpoint import load_checkpoint_payload
from pylatro_agent.learning_bottleneck import (
    CHECKPOINT,
    CONFIRM_SEEDS,
    DEV_SEEDS,
    TRAIN_SEEDS,
    VALIDATION_SEEDS,
    read_json,
)
from pylatro_agent.phase_oracle import ORACLE_SEEDS, worker_init
from pylatro_agent.sale_oracle import SALE_ARMS, run_game
from pylatro_agent.transfer import (
    check_seed_separation,
    paired_statistics,
    restore_configuration,
    sha256,
    source_hashes,
    summarize,
    write_json,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("eval/learning_bottleneck_20260913"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--hours", type=float, default=2)
    args = parser.parse_args()
    output = args.root / "sale_oracle_cpu"
    baseline = args.root / "phase_oracle_cpu"
    original = read_json(baseline / "manifest.json")
    if original["status"] != "complete" or original["checkpoint_sha256"] != sha256(CHECKPOINT):
        raise RuntimeError("Completed matched phase baseline is required")
    if output.exists():
        raise RuntimeError("Use an explicit inspected recovery; sale-oracle outputs are preserved")
    if set(ORACLE_SEEDS) & set(TRAIN_SEEDS + VALIDATION_SEEDS + DEV_SEEDS + CONFIRM_SEEDS):
        raise RuntimeError("Exploratory seeds overlap the main experiment")
    study = read_json(args.root / "manifest.json")
    manifests = [entry["path"] for entry in study["seed_separation"]]
    manifests += [args.root / "data/training/generation.json", args.root / "data/validation/generation.json"]
    separation = check_seed_separation(ORACLE_SEEDS, manifests)
    config, reward = restore_configuration(load_checkpoint_payload(CHECKPOINT, "cpu"), mixture=0.5)
    output.mkdir()
    (output / "games").mkdir()
    manifest = dict(
        status="running",
        kind="exploratory_sale_conditioned_oracle",
        device="cpu",
        batch_size=1,
        workers=args.workers,
        checkpoint=str(CHECKPOINT),
        checkpoint_sha256=sha256(CHECKPOINT),
        source_hashes=source_hashes(),
        agent_config=asdict(config),
        reward_config=asdict(reward),
        seeds=ORACLE_SEEDS,
        seed_separation=separation,
        never_train_on_seeds=True,
        sampling_seeds=[0, 1, 2],
        temperature=1.0,
        deck="b_blue",
        stake=1,
        win_ante=8,
        teacher=dict(shop_policy="search", grow_scalers=True, blind_rollout=None),
        intervention={
            "teacher_sale": "replace only when the isolated teacher proposes SHOP_SELL_JOKER",
            "teacher_other_shop": "replace only other teacher proposals in SHOP",
        },
        post_hoc=True,
        motivation="6.22% Joker-sale agreement in the completed HF audit; exploratory causal follow-up",
        baseline_manifest_sha256=sha256(baseline / "manifest.json"),
        baseline_results_sha256=sha256(baseline / "results.json"),
        sampling=(
            "per-game CPU RNG seed from NumPy SeedSequence([game_seed,sampling_seed,1729]); "
            "proposal drawn before replacement"
        ),
        interpretation=(
            "Teacher-proposal-conditioned interventions change subsequent visited states; "
            "do not isolate a missing input or validate a learned model"
        ),
        accelerator_seconds=0,
    )
    write_json(output / "manifest.json", manifest)
    tasks = []
    for seed in ORACLE_SEEDS:
        tasks.extend((arm, seed, sampling, str(output / "games")) for sampling in (0, 1, 2) for arm in SALE_ARMS)
    started = time.monotonic()

    def deadline(signum, frame):
        raise TimeoutError("Exploratory CPU wall-time limit reached; partial panels are not complete results")

    previous = signal.signal(signal.SIGALRM, deadline)
    signal.setitimer(signal.ITIMER_REAL, args.hours * 3600)
    rows = []
    try:
        with mp.get_context("spawn").Pool(args.workers, initializer=worker_init, initargs=(CHECKPOINT,)) as pool:
            for result in pool.imap_unordered(run_game, tasks, chunksize=1):
                rows.append(result)
                if len(rows) % 10 == 0:
                    print(
                        "phase_oracle",
                        len(rows),
                        len(tasks),
                        "games",
                        round(time.monotonic() - started, 1),
                        "seconds",
                        flush=True,
                    )
        groups = {
            arm: sorted([r for r in rows if r["arm"] == arm], key=lambda r: (r["seed"], r["sampling_seed"]))
            for arm in SALE_ARMS
        }
        references = {
            arm: sorted(
                [read_json(path) for path in (baseline / "games").glob(f"{arm}_*.json")],
                key=lambda r: (r["seed"], r["sampling_seed"]),
            )
            for arm in ("student", "shop")
        }
        for arm, values in references.items():
            if len(values) != 150 or {(r["seed"], r["sampling_seed"]) for r in values} != {
                (seed, sample) for seed in ORACLE_SEEDS for sample in (0, 1, 2)
            }:
                raise RuntimeError(f"Incomplete baseline panel: {arm}")
        summaries = {arm: summarize(values) for arm, values in groups.items()}
        comparisons = {
            arm: {reference: paired_statistics(groups[arm], values) for reference, values in references.items()}
            for arm in SALE_ARMS
        }
        write_json(
            output / "results.json",
            dict(
                summaries=summaries,
                comparisons=comparisons,
                note=(
                    "Post-hoc exploratory study: 50 game seeds are bootstrap units; no candidate selection. "
                    "Conditions depend on the teacher proposal at the currently visited state."
                ),
            ),
        )
        manifest["status"] = "complete"
    except BaseException as exc:
        manifest.update(status="failed", error=repr(exc))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        manifest.update(
            completed_games=len(rows), expected_games=len(tasks), elapsed_seconds=time.monotonic() - started
        )
        write_json(output / "manifest.json", manifest)
    print(json.dumps(read_json(output / "results.json")), flush=True)


if __name__ == "__main__":
    main()
