#!/usr/bin/env python3
"""Export portable game rows and reproduce confirmation statistics without loading models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pylatro_agent.learning_bottleneck import atomic_json, read_json
from pylatro_agent.outcome_study import ROOT
from pylatro_agent.transfer import sha256

if __package__:
    from .benchmark_outcome_teacher import compare_teacher
    from .report_outcome_ppo import compare_matched_training, compare_replicates, verified_panel
    from .run_outcome_confirmation import confirmation_plan
else:
    from benchmark_outcome_teacher import compare_teacher
    from report_outcome_ppo import compare_matched_training, compare_replicates, verified_panel
    from run_outcome_confirmation import confirmation_plan


FIELDS = ("seed", "sampling_seed", "won", "max_ante", "final_ante", "steps",
          "stalled", "censored", "cleared_antes")


def reproduce(rows):
    groups, teacher = {}, []
    for row in rows:
        arm = row["arm"]
        if arm == "teacher":
            teacher.append(row)
            continue
        mode = "greedy" if row["sampling_seed"] is None else "sampled"
        groups.setdefault(arm, dict(sampled=[], greedy=[]))[mode].append(row)
    required = {"control", "seed0", "seed1"}
    if set(groups) not in (required, required | {"matched_control_seed0", "matched_control_seed1"}):
        raise ValueError("Expected one initial control, both candidates, and either both matched controls or neither")
    candidates = {seed: groups[f"seed{seed}"] for seed in (0, 1)}
    results = dict(primary=compare_replicates(candidates, groups["control"]))
    matched = {seed: groups[f"matched_control_seed{seed}"] for seed in (0, 1)
               if f"matched_control_seed{seed}" in groups}
    if matched:
        results["matched_training_control"] = compare_matched_training(candidates, matched)
    results["teacher"] = compare_teacher({seed: candidates[seed]["sampled"] for seed in (0, 1)}, teacher)
    return json.loads(json.dumps(results))


def verify_export(directory):
    metadata = read_json(directory / "outcome_confirmation_manifest.json")
    path = directory / "outcome_confirmation_games.jsonl"
    if sha256(path) != metadata["games_sha256"]:
        raise ValueError("Portable game evidence changed")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if len(rows) != metadata["physical_games"] or reproduce(rows) != metadata["results"]:
        raise ValueError("Portable confirmation statistics differ from the verified report")
    print(f"Reproduced all confirmation and teacher statistics from {len(rows)} physical game rows")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-directory", type=Path, default=Path("docs/transfer/data"))
    parser.add_argument("--verify-only", action="store_true",
                        help="Use only the portable files; no original logs or checkpoints are needed")
    args = parser.parse_args()
    if args.verify_only:
        verify_export(args.output_directory)
        return
    root = args.root
    if not read_json(root / "verification_final.json")["passed"]:
        raise ValueError("Complete the final source and training audit before exporting evidence")
    if read_json(root / "confirmation_stage/manifest.json")["status"] != "complete":
        raise ValueError("All frozen confirmation panels must finish")
    teacher_manifest = read_json(root / "teacher_confirmation400/manifest.json")
    if teacher_manifest["status"] != "complete":
        raise ValueError("Complete the separate teacher reference before exporting")
    selection, jobs = confirmation_plan(root)
    seeds = read_json(root / "manifest.json")["confirmation_seeds"]
    records, sources = [], {}
    for job in jobs:
        panels, evidence = verified_panel(root / "evaluations" / job["name"], seeds,
                                         selection["confirmation_device"])
        sources[job["key"]] = evidence
        # Preserve each raw panel's ordering, including the order used by the
        # deterministic finite-resample absolute-rate bootstrap.
        for values in panels.values():
            records.extend(dict(arm=job["key"], **{key: row[key] for key in FIELDS}) for row in values)
    teacher_path = Path(teacher_manifest["outcomes"])
    if sha256(teacher_path) != teacher_manifest["outcomes_sha256"]:
        raise ValueError("The teacher's raw evidence changed")
    teacher_rows = [json.loads(line) for line in teacher_path.read_text().splitlines()]
    records.extend(dict(arm="teacher", **{key: row[key] for key in FIELDS}) for row in teacher_rows)
    primary_path = root / f"confirmation_{selection['recipe']}_u{selection['updates']}_comparison.json"
    teacher_result = root / "teacher_confirmation400/comparison.json"
    expected = dict(primary=read_json(primary_path)["results"], teacher=read_json(teacher_result))
    supporting = [primary_path, teacher_result, root / "candidate_selection.json", root / "verification_final.json"]
    if selection.get("matched_training_control"):
        path = root / "confirmation_matched_control_comparison.json"
        expected["matched_training_control"] = read_json(path)["results"]
        supporting.append(path)
    if reproduce(records) != expected:
        raise ValueError("Exported fields do not reproduce the original paired reports")
    args.output_directory.mkdir(parents=True, exist_ok=True)
    games_path = args.output_directory / "outcome_confirmation_games.jsonl"
    with games_path.open("x") as handle:
        for row in records:
            handle.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
    atomic_json(args.output_directory / "outcome_confirmation_manifest.json", dict(
        physical_games=len(records), games_sha256=sha256(games_path), results=expected,
        selection=selection, source_panels=sources,
        teacher_source=dict(path=str(teacher_path), sha256=sha256(teacher_path)),
        supporting_sha256={str(path): sha256(path) for path in supporting},
        exporter_sha256=sha256(__file__), environment=read_json(root / "environment.json"),
        scope="Portable fields sufficient to reproduce all strength statistics. Full gameplay diagnostics and "
              "timings remain in the original immutable logs. Physical control and teacher rows appear once.",
    ))
    verify_export(args.output_directory)


if __name__ == "__main__":
    main()
