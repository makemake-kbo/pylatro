#!/usr/bin/env python3
"""Resume completed stages; serialize MPS jobs and retain every failure artifact."""

import argparse
import subprocess
import sys
import time
from pathlib import Path

from pylatro_agent.learning_bottleneck import ARMS, atomic_json, memorization_name, read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("eval/learning_bottleneck_20260913"))
    args = parser.parse_args()
    root = args.root
    root.mkdir(parents=True, exist_ok=True)

    def run(task, marker, *extra):
        if (root / marker).exists():
            print("completed", marker, flush=True)
            return
        name = "_".join([task, *extra]).replace("--", "")
        path = root / f"pipeline_{name}.log"
        attempt = 1
        while path.exists():
            attempt += 1
            path = root / f"pipeline_{name}.attempt{attempt}.log"
        atomic_json(
            root / "pipeline_status.json",
            dict(stage=name, status="running", log=str(path), updated_unix=time.time()),
        )
        with path.open("x") as stream:
            subprocess.run(
                [sys.executable, "-u", "tools/learning_bottleneck.py", task, "--root", str(root), *extra],
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
            )

    try:
        run("init", "manifest.json")
        # A manually started initial diagnostic may still be completing.
        while (root / "compute.json").exists() and any(
            j["status"] == "running" for j in read_json(root / "compute.json")["jobs"]
        ):
            time.sleep(10)
        run("audit", "audit/probe_ids.json")
        run("gradients", "gradients_v2.json")
        for initialization in ("epoch7", "random"):
            for mixture in (0.5, 1.0):
                run(
                    "memorize",
                    f"{memorization_name(initialization, mixture)}/complete.json",
                    "--initialization",
                    initialization,
                    "--mixture",
                    str(mixture),
                )
        needs_followup = False
        for initialization in ("epoch7", "random"):
            summary = read_json(root / memorization_name(initialization, 1.0) / "step300/summary.json")
            metrics = summary["strata"]["overall"]["metrics"]
            needs_followup |= metrics["accuracy"]["decision"] < 0.99 or metrics["probability"]["decision"] <= 0.95
        if needs_followup:
            run("expressivity", "cpu_mps_gradients.json")
        run("inference", "inference_comparisons.json")
        run("memorization_cross_eval", "memorization_cross_eval.json")
        atomic_json(root / "pipeline_status.json", dict(stage="waiting_for_cpu_generation", status="running"))
        while not (root / "generation_complete.json").exists():
            print("Waiting for complete current-rule training/validation data", flush=True)
            time.sleep(30)
        run("replay", "replay_complete.json")
        for seed in (0, 1):
            for arm in ARMS:
                run("train", f"training/{arm}_seed{seed}/provenance.json", "--arm", arm, "--seed", str(seed))
        run("development", "development_complete.json")
        run("confirmation", "confirmation_complete.json")
        atomic_json(root / "pipeline_status.json", dict(stage="experiments_complete", status="complete"))
    except BaseException as exc:
        atomic_json(root / "pipeline_status.json", dict(status="failed", error=repr(exc), updated_unix=time.time()))
        raise


if __name__ == "__main__":
    main()
