#!/usr/bin/env python3
"""Finish the declared selection and confirmation protocol after the longer comparison."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from pylatro_agent.learning_bottleneck import atomic_json, read_json
from pylatro_agent.outcome_study import ROOT
from pylatro_agent.transfer import sha256

if __package__:
    from .advance_outcome_study import select_development
    from .extend_outcome_pilots import verify_predecessor_reports
    from .report_outcome_ppo import compare_panel
else:
    from advance_outcome_study import select_development
    from extend_outcome_pilots import verify_predecessor_reports
    from report_outcome_ppo import compare_panel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root
    if (root / "candidate_selection.json").exists():
        raise FileExistsError("Inspect existing frozen confirmation inputs before any recovery")
    longer = read_json(root / "longer_continuation_selection.json")
    directory = root / "finalization"
    directory.mkdir()
    manifest = dict(
        status="running", started_utc=datetime.now(UTC).isoformat(), jobs=[],
        source_sha256=sha256(__file__), study_manifest_sha256=sha256(root / "manifest.json"),
        longer_selection_sha256=sha256(root / "longer_continuation_selection.json"),
        protocol="Wait for all declared longer development panels and their verified reports. Apply the already "
                 "declared greedy-guardrail then sampled-win-rate ranking across every completed replicated "
                 "development checkpoint. Freeze before confirmation. Include equally trained controls for a "
                 "selected treatment. Run the selected-checkpoint backend probe, full CPU confirmation, separate "
                 "teacher reference, final source audit and portable evidence export. No additional training, "
                 "source maintenance or candidate adaptation to confirmation results.",
    )

    def save():
        atomic_json(directory / "manifest.json", manifest)

    def run(label, tool, *arguments):
        command = [sys.executable, "-u", f"tools/{tool}.py", "--root", str(root), *map(str, arguments)]
        job = dict(label=label, command=command, status="running")
        manifest["jobs"].append(job)
        save()
        print(f"{datetime.now(UTC).isoformat()} {label}", flush=True)
        started = time.monotonic()
        try:
            with (directory / f"{label}.log").open("x") as handle:
                subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True)
            job["status"] = "complete"
        except BaseException:
            job["status"] = "failed"
            raise
        finally:
            job["elapsed_seconds"] = time.monotonic() - started
            save()

    save()
    try:
        if longer["additional_updates"]:
            while True:
                stage = read_json(root / "longer_continuation/manifest.json")
                if stage["status"] == "complete":
                    break
                if stage["status"] != "running":
                    raise RuntimeError("The longer comparison requires inspected recovery")
                print(f"{datetime.now(UTC).isoformat()} Waiting for the full longer comparison", flush=True)
                time.sleep(45)
            expected = [root / f"development_{recipe}_u{longer['target_update']}_comparison.json"
                        for recipe in longer["recipes"]]
            deadline = time.monotonic() + 300
            while not all(path.exists() for path in expected):
                if time.monotonic() >= deadline:
                    raise TimeoutError("The completed longer comparison lacks its owner's reports")
                time.sleep(5)
        if sha256(root / "longer_continuation_selection.json") != manifest["longer_selection_sha256"]:
            raise ValueError("The declared longer comparison changed")
        if sha256(root / "manifest.json") != manifest["study_manifest_sha256"]:
            raise ValueError("The study protocol changed")
        if any(job["status"] == "running" for job in read_json(root / "compute.json")["jobs"]):
            raise RuntimeError("Selection and backend verification require the completed training block")
        run("training_verification", "verify_outcome_ppo")
        paths = sorted(root.glob("development_*_u*_comparison.json"))
        comparisons = [compare_panel(root, read_json(path)["recipe"], read_json(path)["updates"], "development")
                       for path in paths]
        verify_predecessor_reports(paths, comparisons)
        chosen, guardrail = select_development(comparisons)
        manifest["considered_development_reports"] = {str(path): sha256(path) for path in paths}
        manifest["chosen"] = dict(recipe=chosen["recipe"], updates=chosen["updates"], greedy_guardrail=guardrail)
        save()
        reason = ("Highest mean sampled development win rate across both replicas among completed checkpoints "
                  "passing the prespecified greedy guardrail, with greedy rate as tie-break. All declared "
                  "longer comparisons are complete; no confirmation outcomes were inspected." if guardrail else
                  "No completed checkpoint passes the greedy guardrail. Retain the strongest sampled candidate "
                  "under the declared diagnostic fallback; confirmation must still meet every acceptance criterion.")
        arguments = ["--recipe", chosen["recipe"], "--updates", chosen["updates"], "--freeze",
                     "--selection-reason", reason, "--confirmation-device", "cpu"]
        if "_" in chosen["recipe"]:
            arguments += ["--matched-control-recipe", chosen["recipe"].split("_", 1)[0]]
        run("freeze_selection", "report_outcome_ppo", *arguments)
        selection = read_json(root / "candidate_selection.json")
        run("backend_verification", "check_subset_backend", "--checkpoint", selection["checkpoints"]["seed0"],
            "--observation-probe", root / "backend_observations.pt")
        run("confirmation", "run_outcome_confirmation")
        run("teacher_reference", "benchmark_outcome_teacher")
        run("final_source_verification", "verify_outcome_ppo", "--final")
        run("portable_evidence", "export_outcome_confirmation")
        manifest["status"] = "complete"
        print("Frozen confirmation, teacher reference, source audit and portable evidence are complete", flush=True)
    except BaseException as exc:
        manifest.update(status="failed", error=repr(exc))
        raise
    finally:
        manifest["finished_utc"] = datetime.now(UTC).isoformat()
        save()


if __name__ == "__main__":
    main()
