#!/usr/bin/env python3
"""Run matched PPO pilots sequentially, overlapping one CPU evaluation worker."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import sys
import time
from pathlib import Path

from pylatro_agent.learning_bottleneck import atomic_json, read_json
from pylatro_agent.outcome_study import ROOT
from pylatro_agent.transfer import sha256, source_hashes


def completed_checkpoint(directory):
    completed = []
    for path in directory.glob("leg_*/manifest.json"):
        metadata = read_json(path)
        if metadata["status"] != "complete":
            raise ValueError(f"Inspect unfinished leg before continuing: {path}")
        checkpoint = Path(metadata["checkpoint"])
        if sha256(checkpoint) != metadata["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint changed: {checkpoint}")
        completed.append((metadata["update_count"], checkpoint))
    return max(completed, default=(0, None), key=lambda item: item[0])


def checkpoint_at_update(directory, update):
    matches = []
    for path in directory.glob("leg_*/manifest.json"):
        metadata = read_json(path)
        if metadata["status"] == "complete" and metadata["update_count"] == update:
            checkpoint = Path(metadata["checkpoint"])
            if sha256(checkpoint) != metadata["checkpoint_sha256"]:
                raise ValueError("Branch parent checkpoint changed")
            matches.append(checkpoint)
    if len(matches) != 1:
        raise ValueError(f"Expected one immutable checkpoint at update {update}, found {len(matches)}")
    return matches[0]


def command_run(command, log):
    print(json.dumps(dict(command=command, log=str(log))), flush=True)
    started = time.monotonic()
    with log.open("x") as handle:
        subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True)
    return dict(command=command, log=str(log), elapsed_seconds=time.monotonic() - started)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--updates", type=int, default=20)
    parser.add_argument("--hours-per-leg", type=float, default=1.3)
    parser.add_argument("--recipes", nargs="+", choices=("gae100", "gae097", "gae100_lr3e5", "gae097_lr3e5",
                                                        "gae100_terminal", "gae097_terminal",
                                                        "gae100_terminal_lr3e5", "gae097_terminal_lr3e5"),
                        default=["gae100", "gae097"])
    parser.add_argument("--branch-base", choices=("gae100", "gae097", "gae100_terminal", "gae097_terminal"))
    parser.add_argument("--branch-update", type=int, default=20)
    parser.add_argument("--stage-name", required=True)
    args = parser.parse_args()
    if args.updates < 1 or args.hours_per_leg <= 0 or len(set(args.recipes)) != len(args.recipes):
        parser.error("Positive update and time limits, and distinct recipes, are required")
    directory = args.root / args.stage_name
    directory.mkdir(exist_ok=False)
    metadata = dict(
        status="running",
        target_updates=args.updates,
        recipes=args.recipes,
        training_replicates=[0, 1],
        development_games=200,
        source_hashes=source_hashes(),
        branch_base=args.branch_base,
        branch_update=args.branch_update if args.branch_base else None,
        jobs=[],
    )
    atomic_json(directory / "manifest.json", metadata)
    executable = [sys.executable, "-u"]
    reference = args.root / "evaluations/baseline_dev200"
    evaluations = []
    started = time.monotonic()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as worker:
            evaluations.append(worker.submit(
                command_run,
                [*executable, "tools/evaluate_outcome_ppo.py", "--root", str(args.root),
                              "--checkpoint", str(args.root / "critic_seed0/checkpoint.pt"),
                              "--name", "baseline_dev200", "--games", "200"],
                directory / "baseline_dev200.log",
            ))
            for seed in (0, 1):
                for recipe in args.recipes:
                    # Surface earlier evaluation failures before allocating another MPS leg.
                    for evaluation in evaluations:
                        if evaluation.done():
                            evaluation.result()
                    trial = args.root / "trials" / f"{recipe}_seed{seed}"
                    update, checkpoint = completed_checkpoint(trial)
                    branch = None
                    if update == 0 and args.branch_base and recipe != args.branch_base:
                        if recipe not in (f"{args.branch_base}_lr3e5", f"{args.branch_base}_terminal"):
                            raise ValueError("The declared branch must retain the selected GAE setting")
                        parent = args.root / "trials" / f"{args.branch_base}_seed{seed}"
                        branch = checkpoint_at_update(parent, args.branch_update)
                        update = args.branch_update
                    if update >= args.updates:
                        raise ValueError("Pilot target must extend the current checkpoint")
                    command = [*executable,
                        "tools/train_outcome_ppo.py", "--root", str(args.root),
                        "--recipe", recipe, "--seed", str(seed),
                        "--updates", str(args.updates - update), "--hours", str(args.hours_per_leg),
                    ]
                    if checkpoint:
                        command += ["--resume", str(checkpoint)]
                    elif branch:
                        command += ["--branch-from", str(branch)]
                    job = command_run(command, directory / f"{recipe}_seed{seed}_training.log")
                    actual, checkpoint = completed_checkpoint(trial)
                    if actual != args.updates:
                        raise RuntimeError(f"Time-limited leg ended at {actual}; matched target is {args.updates}")
                    job.update(checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint), updates=actual)
                    metadata["jobs"].append(job)
                    atomic_json(directory / "manifest.json", metadata)
                    name = f"{recipe}_seed{seed}_u{actual}_dev200"
                    evaluations.append(worker.submit(
                        command_run,
                        [*executable, "tools/evaluate_outcome_ppo.py", "--root", str(args.root),
                                      "--checkpoint", str(checkpoint), "--name", name, "--games", "200",
                                      "--paired", str(reference)],
                        directory / f"{name}.log",
                    ))
            metadata["evaluations"] = [future.result() for future in evaluations]
        metadata["status"] = "complete"
    except BaseException as exc:
        metadata.update(status="failed", error=repr(exc))
        raise
    finally:
        metadata["elapsed_seconds"] = time.monotonic() - started
        atomic_json(directory / "manifest.json", metadata)
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
