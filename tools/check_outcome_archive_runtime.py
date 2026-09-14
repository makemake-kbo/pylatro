#!/usr/bin/env python3
"""After the final audit, compare current and archived source on bounded CPU gameplay."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from pylatro_agent.learning_bottleneck import atomic_json, read_json
from pylatro_agent.outcome_study import ROOT
from pylatro_agent.transfer import sha256, source_hashes

if __package__:
    from .report_outcome_ppo import frozen_checkpoints
    from .verify_outcome_ppo import reference_source_gate
else:
    from report_outcome_ppo import frozen_checkpoints
    from verify_outcome_ppo import reference_source_gate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    reference = root / "reference_source"
    gate, _ = reference_source_gate(root, reference, final=True)
    if read_json(root / "finalization/manifest.json")["status"] != "complete":
        raise ValueError("Finish all frozen study evaluations and evidence export first")
    selection = read_json(root / "candidate_selection.json")
    checkpoints, hashes = frozen_checkpoints(selection)
    protocol = read_json(root / "manifest.json")
    seeds = protocol["diagnostic_seeds"][:2]
    manifests = [Path(item["path"]).resolve() for item in protocol["seed_separation"]]
    manifests += [root / "critic_training_seeds.json",
                  *sorted(root.glob("trials/*/tensorboard/training_episodes.jsonl"))]
    repository = Path(__file__).resolve().parents[1]
    current = {name: digest for name, digest in source_hashes().items() if name.startswith("src/")}
    archived = read_json(reference / "manifest.json")["source_hashes"]
    if set(current) != set(archived):
        raise ValueError("Current and archived core file sets differ")
    directory = root / "archive_runtime_check"
    directory.mkdir()
    report = dict(status="running", source_gate=gate, source_sha256=sha256(__file__),
                  checkpoint_sha256=hashes["seed0"], seeds=seeds, accelerator_seconds=0, jobs=[],
                  changed_core_files=[name for name in current if current[name] != archived[name]],
                  scope="Two diagnostic seeds, sampled seed zero and greedy on CPU under both source versions. "
                        "Compares every game and critic field except timing. This is an archive/runtime check, "
                        "not a strength estimate or proof of complete PPO-process resume determinism.")
    started = time.monotonic()
    try:
        outcomes = {}
        for version, code, expected in (("current", repository / "src", current),
                                         ("archived", reference / "src", archived)):
            environment = dict(os.environ, PYTHONPATH=str(code))
            for mode in ("sampled", "greedy"):
                destination = directory / version / mode
                command = [sys.executable, str(repository / "tools/evaluate_checkpoint.py"),
                           "--checkpoint", str(Path(checkpoints["seed0"]).resolve()), "--seeds", *map(str, seeds),
                           "--mode", mode, "--sampling-seed", "0", "--device", "cpu", "--deck", "b_blue",
                           "--hand-ar-mixture-eps", "0.5", "--rollout-temperature", "1.0", "--stake", "1",
                           "--win-ante", "8", "--eval-batch-size", "32", "--eval-workers", "4",
                           "--eval-cpu-threads", "4", "--output-dir", str(destination)]
                for path in manifests:
                    command += ["--training-manifest", str(path)]
                job = dict(version=version, mode=mode, pythonpath=str(code), command=command)
                report["jobs"].append(job)
                atomic_json(directory / "manifest.json", report)
                job_started = time.monotonic()
                with (directory / f"{version}_{mode}.log").open("x") as handle:
                    subprocess.run(command, env=environment, stdout=handle, stderr=subprocess.STDOUT, check=True)
                manifest = read_json(destination / "manifest.json")
                if {name: digest for name, digest in manifest["source_hashes"].items()
                        if name.startswith("src/")} != expected:
                    raise ValueError("Evaluation imported a different source tree")
                rows = [json.loads(line) for line in (destination / "outcomes.jsonl").read_text().splitlines()]
                if len(rows) != len(seeds) or {row["seed"] for row in rows} != set(seeds):
                    raise ValueError("Runtime check has missing or duplicate game seeds")
                timing = {"elapsed_seconds", "policy_seconds", "env_seconds", "env_wall_seconds"}
                outcomes[version, mode] = {row["seed"]: {key: value for key, value in row.items() if key not in timing}
                                           for row in rows}
                job.update(elapsed_seconds=time.monotonic() - job_started,
                           manifest_sha256=sha256(destination / "manifest.json"),
                           outcomes_sha256=sha256(destination / "outcomes.jsonl"))
        if any(outcomes["current", mode] != outcomes["archived", mode] for mode in ("sampled", "greedy")):
            raise ValueError("Current and archived source disagree on gameplay or critic outputs")
        report.update(status="complete", passed=True, physical_games=8, exact_non_timing_agreement=True)
    except BaseException as exc:
        report.update(status="failed", error=repr(exc))
        raise
    finally:
        report["elapsed_cpu_seconds"] = time.monotonic() - started
        atomic_json(directory / "manifest.json", report)
    print(json.dumps({key: value for key, value in report.items() if key != "jobs"}, indent=2))


if __name__ == "__main__":
    main()
