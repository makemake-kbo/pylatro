#!/usr/bin/env python3
"""Verify the selected-policy phase intervention and its matched earlier diagnostic seeds."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

from pylatro_agent.learning_bottleneck import atomic_json, read_json
from pylatro_agent.outcome_study import ROOT
from pylatro_agent.phase_oracle import ORACLE_ARMS
from pylatro_agent.transfer import paired_statistics, sha256, source_hashes, summarize

if __package__:
    from .report_learning_bottleneck import phase_oracle_results
    from .verify_outcome_ppo import reference_source_gate
else:
    from report_learning_bottleneck import phase_oracle_results
    from verify_outcome_ppo import reference_source_gate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--reference-source", type=Path)
    args = parser.parse_args()
    root = args.root
    directory = root / "selected_phase_diagnostic"
    plan = read_json(directory / "manifest.json")
    study = read_json(root / "manifest.json")
    selection = read_json(root / "candidate_selection.json")
    if (sha256(root / "manifest.json") != plan["parent_study_sha256"]
            or sha256(root / "candidate_selection.json") != plan["candidate_selection_sha256"]
            or plan["checkpoint_sha256"] != selection["checkpoint_hashes"]["seed0"]
            or plan["seeds"] != study["diagnostic_seeds"][:25]):
        raise ValueError("Selected phase-diagnostic inputs changed")
    gate, _ = reference_source_gate(root, args.reference_source, final=True)
    expected = ({name: digest for name, digest in source_hashes().items() if name.startswith("src/")}
                if args.reference_source is None else
                read_json(args.reference_source / "manifest.json")["source_hashes"])
    results = {}
    records = {}
    historical_source_differences = {}
    evidence = {str(directory / "manifest.json"): sha256(directory / "manifest.json")}
    for name, location, seeds in (("before_ppo", root, study["diagnostic_seeds"]),
                                    ("selected_ppo", directory, plan["seeds"])):
        result = phase_oracle_results(location, verify_traces=True, expected_seeds=seeds, write_analysis=False)
        if result is None:
            raise ValueError("Complete both phase-diagnostic panels before reporting")
        manifest = result["manifest"]
        core = {path: digest for path, digest in manifest["source_hashes"].items() if path.startswith("src/")}
        if name == "selected_ppo" and core != expected:
            raise ValueError("Selected phase-diagnostic core source differs from the verified reference")
        if name == "before_ppo":
            historical_source_differences = {
                path: dict(before=core.get(path), selected=expected.get(path))
                for path in sorted(set(core) | set(expected)) if core.get(path) != expected.get(path)
            }
            for path in ("src/pylatro_agent/phase_oracle.py", "src/pylatro_agent/constants.py",
                         "src/pylatro_agent/transfer.py"):
                if core.get(path) != expected[path]:
                    raise ValueError("Historical trace interpretation or statistics source changed")
        if name == "selected_ppo" and manifest["checkpoint_sha256"] != plan["checkpoint_sha256"]:
            raise ValueError("The selected diagnostic used a different checkpoint")
        folder = location / "phase_oracle_cpu"
        paths = sorted((folder / "games").glob("*.json"))
        rows = [read_json(path) for path in paths]
        groups = {arm: [row for row in rows if row["arm"] == arm and row["seed"] in plan["seeds"]]
                  for arm in ORACLE_ARMS}
        records[name] = groups
        results[name] = {arm: summarize(values) for arm, values in groups.items()}
        for path in (folder / "manifest.json", folder / "results.json", *paths):
            evidence[str(path)] = sha256(path)
    before, after = records["before_ppo"], records["selected_ppo"]
    for old, new in zip(before["teacher_all"], after["teacher_all"], strict=True):
        excluded = {"elapsed_seconds", "trace_sha256"}
        if ({key: value for key, value in old.items() if key not in excluded}
                != {key: value for key, value in new.items() if key not in excluded}):
            raise ValueError(f"The matched teacher outcome differs on seed {old['seed']}")
        name = f"teacher_all_{old['seed']}_sampleNone.jsonl.gz"
        with (gzip.open(root / "phase_oracle_cpu/games" / name, "rt") as left,
              gzip.open(directory / "phase_oracle_cpu/games" / name, "rt") as right):
            if left.read() != right.read():
                raise ValueError(f"The matched teacher actions differ on seed {old['seed']}")
    teacher = [dict(row, sampling_seed=repeat) for row in after["teacher_all"] for repeat in (0, 1, 2)]
    comparisons = {arm: paired_statistics(after[arm], after["student"]) for arm in ORACLE_ARMS[1:-1]}
    comparisons["teacher_all"] = paired_statistics(teacher, after["student"])
    result = dict(
        passed=True, game_seeds=len(plan["seeds"]), selected_physical_games=plan["expected_physical_games"],
        summaries=results, phase_effects=comparisons,
        matched_teacher_actions_identical=True, source_verification=gate, source_sha256=sha256(__file__),
        historical_source_differences=historical_source_differences,
        evidence_sha256=evidence, accelerator_seconds=0,
        interpretation="Exploratory diagnosis on the first25 reserved diagnostic seeds and the fixed seed-zero "
                       "representative. Sampling repeats are averaged within game seed. No confirmation selection "
                       "or acceptance changes, and no PPO training uses these actions. The teacher receives full "
                       "simulator state and probes; phase interventions change subsequent visited states. "
                       "Earlier and current teacher actions/outcomes match exactly. The earlier diagnostic predates "
                       "four PPO training/resume source changes and uses its original reward configuration. "
                       "Its student is not used as a before/after control here; the independent400seed confirmation "
                       "provides that comparison. These small-panel results do not validate a learned change.",
    )
    path = directory / "comparison.json"
    if path.exists():
        previous = read_json(path)
        for key in ("summaries", "phase_effects", "historical_source_differences", "evidence_sha256"):
            if previous[key] != json.loads(json.dumps(result[key])):
                raise ValueError("Existing selected phase-diagnostic evidence changed")
    else:
        atomic_json(path, result)
    print(json.dumps({key: value for key, value in result.items() if key != "evidence_sha256"}, indent=2))


if __name__ == "__main__":
    main()
