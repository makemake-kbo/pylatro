#!/usr/bin/env python3
"""Wait for full development panels, freeze one candidate, and run CPU confirmation."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from evaluate_refactor_panels import checkpoints

from pylatro_agent.learning_bottleneck import atomic_json, read_json
from pylatro_agent.transfer import sha256


def selection_rule(report):
    results = report["results"]
    paired = results["sampled"]["paired"]
    eligible = (
        paired["sampled_or_greedy_delta"] > 0
        and paired["paired_ci95"][0] > 0
        and all(results["per_replicate"][str(seed)]["sampled"]["sampled_or_greedy_delta"] > 0 for seed in (0, 1))
        and results["greedy"]["paired"]["paired_ci95"][0] >= -0.05
        and report["acceptance"]["no_stalled_or_censored_games"]
    )
    return eligible, (results["sampled"]["candidate"]["win_rate"], results["greedy"]["candidate"]["win_rate"])


def prepare_confirmation(root, selection):
    destination = root / "confirmation_cpu"
    if destination.exists():
        if read_json(destination / "candidate_selection.json") != selection:
            raise ValueError("Existing confirmation namespace has a different selection")
        return destination
    destination.mkdir()
    for name in ("manifest.json", "cache_complete.json", "candidate_selection.json"):
        shutil.copyfile(root / name, destination / name)
    _, candidates = checkpoints(root, selection["family"])
    references = [p.parent.name for p in candidates] + ["training.pt", "validation.pt"]
    if selection["family"] != "sale":
        references.append(f"{selection['family']}_neutral_control.pt")
    for name in references:
        (destination / name).symlink_to(Path("..") / name, target_is_directory=(root / name).is_dir())
    atomic_json(
        destination / "parent_study.json",
        dict(parent=str(root), purpose="Locked CPU confirmation namespace", linked_references=references),
    )
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("eval/action_refactor_20260913"))
    args = parser.parse_args()
    root = args.root
    paths = {
        "sale": root / "cpu_development_sale_200_comparisons.json",
        "combined": root / "combined_cpu_expansion" / "cpu_development_combined_200_comparisons.json",
    }
    rule = dict(
        source_panels={key: str(path) for key, path in paths.items()},
        candidates=["sale", "combined"],
        eligibility=(
            "Positive sampled paired CI and both training differences; "
            "greedy lower CI at least -5pp; no incomplete games"
        ),
        ranking=(
            "Highest mean sampled win rate across both training replicates; "
            "then greedy win rate; then fewer changed heads"
        ),
        confirmation=(
            "400 untouched game seeds, three sampling seeds, both training replicates, shared matched control, CPU"
        ),
        final_acceptance=(
            "At least +5pp sampled, positive paired 95% CI, both replicates positive, "
            "greedy lower CI >=-5pp, no failures"
        ),
    )
    rule_path = root / "confirmation_selection_rule.json"
    if rule_path.exists():
        if read_json(rule_path) != rule:
            raise ValueError("The recorded selection rule changed")
    else:
        atomic_json(rule_path, rule)
    deadline = time.monotonic() + 8 * 3600
    while not all(path.exists() for path in paths.values()):
        if time.monotonic() >= deadline:
            raise TimeoutError("Development panels did not finish within the bounded wait")
        for log in (root / "sale_development_cpu.log", root / "combined_development_cpu.log"):
            if log.exists() and "Traceback (most recent call last)" in log.read_text():
                raise RuntimeError(f"Development process failed; inspect {log}")
        atomic_json(
            root / "confirmation_status.json",
            dict(status="waiting_for_development", pending=[name for name, path in paths.items() if not path.exists()]),
        )
        time.sleep(15)
    reports = {key: read_json(path) for key, path in paths.items()}
    eligible = []
    for family, report in reports.items():
        if report["protocol"]["stage"] != "development" or report["protocol"]["device"] != "cpu":
            raise ValueError("Selection requires matched CPU development panels")
        if report["protocol"]["seeds"] != read_json(root / "manifest.json")["development_seeds"]:
            raise ValueError("Selection requires the entire reserved development panel")
        control, candidates = checkpoints(root, family)
        expected_hashes = {
            "control": sha256(control),
            **{f"seed{seed}": sha256(path) for seed, path in enumerate(candidates)},
        }
        if report["protocol"]["checkpoint_hashes"] != expected_hashes or report["protocol"]["family"] != family:
            raise ValueError("A selection candidate differs from the checkpoint evaluated on development")
        accepted, rank = selection_rule(report)
        if accepted:
            eligible.append((*rank, family == "sale", family))
    if not eligible:
        atomic_json(root / "confirmation_status.json", dict(status="no_eligible_candidate", confirmation_touched=False))
        print(
            "No candidate meets the pre-recorded development screening rule; confirmation remains untouched", flush=True
        )
        return
    family = max(eligible)[-1]
    control, candidates = checkpoints(root, family)
    digests = {"control": sha256(control), **{f"seed{seed}": sha256(path) for seed, path in enumerate(candidates)}}
    selection = dict(
        family=family,
        checkpoint_hashes=digests,
        selection_rule_sha256=sha256(rule_path),
        development_comparison_hashes={name: sha256(path) for name, path in paths.items()},
        source_panels={name: str(path) for name, path in paths.items()},
        status="frozen_before_confirmation",
    )
    selection_path = root / "candidate_selection.json"
    if selection_path.exists() and read_json(selection_path) != selection:
        raise ValueError("Frozen confirmation candidate cannot be changed")
    atomic_json(selection_path, selection)
    destination = prepare_confirmation(root, selection)
    atomic_json(root / "confirmation_status.json", dict(status="running", family=family, output_root=str(destination)))
    print("Frozen candidate:", json.dumps(selection), flush=True)
    subprocess.run(
        [
            sys.executable,
            "-u",
            "tools/evaluate_refactor_panels.py",
            "--root",
            str(destination),
            "--family",
            family,
            "--stage",
            "confirmation",
            "--device",
            "cpu",
        ],
        check=True,
    )
    comparison = destination / f"cpu_confirmation_{family}_400_comparisons.json"
    report = read_json(comparison)
    atomic_json(
        root / "confirmation_status.json",
        dict(
            status="complete",
            family=family,
            comparison=str(comparison),
            comparison_sha256=sha256(comparison),
            validated=report["validated"],
        ),
    )
    print("Confirmation completed; validated:", report["validated"], flush=True)


if __name__ == "__main__":
    main()
