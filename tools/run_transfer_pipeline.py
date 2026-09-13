#!/usr/bin/env python3
"""Sequence authorized continuation work after a completed development screen."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--wait-hours", type=float, default=3.0)
    parser.add_argument(
        "--resume-measurements",
        action="store_true",
        help="Use completed regression/historical panels; continuations must not exist yet",
    )
    args = parser.parse_args()
    if args.resume_measurements and (args.root / "continuations").exists():
        raise ValueError("Measurement resume cannot replace existing continuation work")
    marker = args.root / "development/epoch10_greedy_demonstration200/manifest.json"
    start = time.monotonic()
    while True:
        try:
            if marker.exists() and json.loads(marker.read_text())["status"] == "complete":
                break
        except json.JSONDecodeError:
            pass  # A separately running older writer may still be replacing its manifest.
        if time.monotonic() - start > args.wait_hours * 3600:
            raise TimeoutError("Development screen did not complete; no training or confirmation was started")
        time.sleep(30)
    selection = args.root / "development/selection.json"
    if args.resume_measurements:
        required = [args.root / "regression_epoch10" / mode / "manifest.json" for mode in ("greedy", "sampled")] + [
            args.root / "historical_epoch10_mps" / label / "manifest.json"
            for label in ("sampled0", "sampled1", "sampled2", "greedy0")
        ]
        while True:
            try:
                statuses = [json.loads(path.read_text())["status"] if path.exists() else "missing" for path in required]
            except json.JSONDecodeError:
                statuses = ["writing"]
            if all(status == "complete" for status in statuses):
                break
            if "failed" in statuses:
                raise ValueError("A failed measurement must be repaired before continuation can resume")
            if time.monotonic() - start > args.wait_hours * 3600:
                raise TimeoutError("Pending measurement did not complete; no continuation was started")
            time.sleep(30)
    # Preserve the previous 10000-series measurements as explicit regression checks.
    from contextlib import nullcontext

    def validate_completed(path):
        from pylatro_agent.transfer import sha256, summarize

        manifest = json.loads((path / "manifest.json").read_text())
        rows = [json.loads(line) for line in (path / "outcomes.jsonl").read_text().splitlines()]
        expected_hash = sha256("checkpoints/supervised_5k_v68_20260911_181359/supervised_epoch10.pt")
        if (
            manifest["status"] != "complete"
            or manifest["checkpoint_sha256"] != expected_hash
            or {row["seed"] for row in rows} != set(manifest["seeds"])
            or summarize(rows)["games"] != len(manifest["seeds"])
        ):
            raise ValueError(f"Cannot resume incomplete or mismatched measurement: {path}")

    with (args.root / "regression_epoch10.log").open("x") if not args.resume_measurements else nullcontext() as log:
        for mode, sampling_seed, games in (("greedy", 0, 50), ("sampled", 31013, 25)):
            if args.resume_measurements:
                validate_completed(args.root / "regression_epoch10" / mode)
                continue
            subprocess.run(
                [
                    sys.executable,
                    "-u",
                    "tools/evaluate_checkpoint.py",
                    "--checkpoint",
                    "checkpoints/supervised_5k_v68_20260911_181359/supervised_epoch10.pt",
                    "--panel",
                    "regression",
                    "--games",
                    str(games),
                    "--mode",
                    mode,
                    "--sampling-seed",
                    str(sampling_seed),
                    "--max-idle-steps",
                    "32",
                    "--device",
                    "mps",
                    "--training-manifest",
                    str(args.dataset / "metadata/generation.json"),
                    "--training-manifest",
                    "docs/heuristic/pretraining_v68_seeds.json",
                    "--output-dir",
                    str(args.root / "regression_epoch10" / mode),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=600,
            )
    if args.resume_measurements:
        for label in ("sampled0", "sampled1", "sampled2", "greedy0"):
            validate_completed(args.root / "historical_epoch10_mps" / label)
    else:
        with (args.root / "historical_epoch10_mps.log").open("x") as log:
            subprocess.run(
                [
                    sys.executable,
                    "-u",
                    "tools/historical_transfer_evaluation.py",
                    "--reconstruction",
                    "/tmp/pylatro_transfer_generation_v2_20260912",
                    "--checkpoint",
                    "checkpoints/supervised_5k_v68_20260911_181359/supervised_epoch10.pt",
                    "--output-dir",
                    str(args.root / "historical_epoch10_mps"),
                    "--device",
                    "mps",
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=2400,
            )
    accelerator_seconds = 0.0
    for manifest_path in args.root.rglob("manifest.json"):
        manifest = json.loads(manifest_path.read_text())
        if str(manifest.get("device", "")).startswith("mps"):
            timing_path = manifest_path.parent / "timing.json"
            seconds = manifest.get("elapsed_seconds", 0.0)
            if not seconds and timing_path.exists():
                seconds = json.loads(timing_path.read_text())["elapsed_seconds"]
            accelerator_seconds += seconds
    # Use unused diagnostic allocation, retaining three full hours for confirmation
    # and six minutes for accelerator initialization and short probes.
    improvement_hours = 12.0 - 3.0 - accelerator_seconds / 3600.0 - 0.1
    if improvement_hours <= 0:
        raise TimeoutError("No improvement allocation remains with confirmation reserve intact")
    (args.root / "accelerator_budget.json").write_text(
        json.dumps(
            {
                "diagnostic_accelerator_seconds": accelerator_seconds,
                "improvement_hours": improvement_hours,
                "confirmation_hours": 3.0,
                "overhead_reserve_hours": 0.1,
                "limit_hours": 12.0,
            },
            indent=2,
        )
    )
    with (args.root / "continuations.log").open("x") as log:
        subprocess.run(
            [
                sys.executable,
                "-u",
                "tools/continue_transfer.py",
                "--selection",
                str(selection),
                "--dataset",
                str(args.dataset),
                "--output-dir",
                str(args.root / "continuations"),
                "--device",
                "mps",
                "--improvement-hours",
                str(improvement_hours),
                "--confirmation-hours",
                "3",
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=(improvement_hours + 3) * 3600,
        )


if __name__ == "__main__":
    main()
