#!/usr/bin/env python3
"""CPU-only exploratory teacher phase replacements; never selects a confirmation candidate."""

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
from pylatro_agent.phase_oracle import ORACLE_ARMS, ORACLE_SEEDS, run_game, worker_init
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
    output = args.root / "phase_oracle_cpu"
    if output.exists():
        raise RuntimeError("Use an explicit inspected recovery; phase-oracle outputs are preserved")
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
        kind="exploratory_phase_oracle",
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
            "hand": "all CHOOSE_ACTION decisions, including consumables or reordering in that phase",
            "shop": "all SHOP decisions; booster-pack decisions remain with the student",
            "hand_shop": "both CHOOSE_ACTION and SHOP",
            "teacher_all": "every phase, deterministic once per game seed",
        },
        sampling=(
            "per-game CPU RNG seed from NumPy SeedSequence([game_seed,sampling_seed,1729]); "
            "proposal drawn before replacement"
        ),
        interpretation=(
            "Phase-policy interventions change subsequent visited states; "
            "do not isolate a missing input or validate a learned model"
        ),
        accelerator_seconds=0,
    )
    write_json(output / "manifest.json", manifest)
    tasks = []
    for seed in ORACLE_SEEDS:
        tasks.extend((arm, seed, sampling, str(output / "games")) for sampling in (0, 1, 2) for arm in ORACLE_ARMS[:-1])
        tasks.append(("teacher_all", seed, None, str(output / "games")))
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
        groups = {arm: [r for r in rows if r["arm"] == arm] for arm in ORACLE_ARMS}
        summaries = {arm: summarize(values) for arm, values in groups.items()}
        comparisons = {arm: paired_statistics(groups[arm], groups["student"]) for arm in ORACLE_ARMS[1:-1]}
        teacher_repeated = [
            dict(row, sampling_seed=sampling) for row in groups["teacher_all"] for sampling in (0, 1, 2)
        ]
        comparisons["teacher_all"] = paired_statistics(teacher_repeated, groups["student"])
        write_json(
            output / "results.json",
            dict(
                summaries=summaries,
                comparisons=comparisons,
                note=(
                    "50 game seeds are the bootstrap units. "
                    "Teacher-all outcomes are reused only to align sampling repeats."
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
