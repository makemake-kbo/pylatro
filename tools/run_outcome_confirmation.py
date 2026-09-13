#!/usr/bin/env python3
"""Run the already frozen confirmation panels once, with matched trained controls."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from pylatro_agent.learning_bottleneck import atomic_json, read_json
from pylatro_agent.outcome_study import ROOT
from pylatro_agent.transfer import sha256, source_hashes

if __package__:
    from .report_outcome_ppo import frozen_checkpoints
else:
    from report_outcome_ppo import frozen_checkpoints


def confirmation_plan(root):
    selection = read_json(root / "candidate_selection.json")
    if sha256(root / "manifest.json") != selection["study_manifest_sha256"]:
        raise ValueError("The frozen study protocol changed")
    for record in (selection, selection.get("matched_training_control")):
        if record and sha256(record["development_comparison"]) != record["development_comparison_sha256"]:
            raise ValueError("A frozen development comparison changed")
    paths, _ = frozen_checkpoints(selection)
    device = selection["confirmation_device"]
    if device not in ("mps", "cpu"):
        raise ValueError("Unsupported frozen confirmation device")
    jobs = []
    for key, checkpoint in paths.items():
        if key == "control":
            name = "baseline_confirmation400"
        else:
            seed = int(key[-1])
            recipe = (selection["matched_training_control"]["recipe"]
                      if key.startswith("matched_control") else selection["recipe"])
            name = f"{recipe}_seed{seed}_u{selection['updates']}_confirmation400"
        jobs.append(dict(key=key, name=name, checkpoint=checkpoint, command=[
            sys.executable, "-u", "tools/evaluate_outcome_ppo.py", "--root", str(root),
            "--checkpoint", checkpoint, "--name", name, "--games", "400",
            "--stage", "confirmation", "--device", device,
        ]))
    # Every checkpoint sees exactly the same seeds, device and all four modes.
    if len({job["name"] for job in jobs}) != len(jobs):
        raise ValueError("Duplicate confirmation arm names")
    return selection, jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root
    selection, jobs = confirmation_plan(root)
    ledger = read_json(root / "compute.json")
    if any(job["status"] == "running" for job in ledger["jobs"]):
        raise RuntimeError("Finish active accelerator work before confirmation")
    estimate = len(jobs) * (60 + 400 * 4 * 1.5)
    remaining = ledger["limit_seconds"] - ledger["elapsed_seconds"]
    if selection["confirmation_device"] == "mps" and 1.2 * estimate + 5 > remaining:
        raise TimeoutError("The full frozen MPS confirmation suite does not fit with its 20% margin")
    for job in jobs:
        if (root / "evaluations" / job["name"]).exists():
            raise FileExistsError("Confirmation artifacts already exist; inspect them before any recovery")
    directory = root / "confirmation_stage"
    directory.mkdir()
    started = time.monotonic()
    manifest = dict(
        status="running", started_utc=datetime.now(UTC).isoformat(),
        selection_sha256=sha256(root / "candidate_selection.json"),
        device=selection["confirmation_device"], jobs=jobs, completed_jobs=[],
        physical_game_count=len(jobs) * 400 * 4,
        estimated_accelerator_seconds=estimate if selection["confirmation_device"] == "mps" else 0,
        compute_before=ledger, source_hashes=source_hashes(),
    )
    atomic_json(directory / "manifest.json", manifest)
    try:
        for job in jobs:
            # Revalidate immutable inputs before each arm, without using any outcomes to adapt the plan.
            confirmation_plan(root)
            if sha256(root / "candidate_selection.json") != manifest["selection_sha256"]:
                raise ValueError("Confirmation selection changed during execution")
            print(json.dumps(job), flush=True)
            job_started = time.monotonic()
            with (directory / f"{job['name']}.log").open("x") as handle:
                subprocess.run(job["command"], stdout=handle, stderr=subprocess.STDOUT, check=True)
            manifest["completed_jobs"].append(dict(
                name=job["name"], elapsed_seconds=time.monotonic() - job_started,
                comparison_sha256=sha256(root / "evaluations" / job["name"] / "comparison.json"),
            ))
            atomic_json(directory / "manifest.json", manifest)
        command = [sys.executable, "tools/report_outcome_ppo.py", "--root", str(root),
                   "--recipe", selection["recipe"], "--updates", str(selection["updates"]),
                   "--stage", "confirmation"]
        with (directory / "paired_report.log").open("x") as handle:
            subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True)
        manifest["status"] = "complete"
    except BaseException as exc:
        manifest.update(status="failed", error=repr(exc))
        raise
    finally:
        manifest.update(elapsed_seconds=time.monotonic() - started,
                        compute_after=read_json(root / "compute.json"))
        atomic_json(directory / "manifest.json", manifest)
    print("All frozen confirmation panels and paired reports are complete", flush=True)


if __name__ == "__main__":
    main()
