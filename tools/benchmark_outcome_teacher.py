#!/usr/bin/env python3
"""Measure the frozen teacher reference after candidate confirmation, without PPO labels."""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

from pylatro_agent.checkpoint import load_checkpoint_payload
from pylatro_agent.learning_bottleneck import atomic_json, read_json
from pylatro_agent.outcome_study import ROOT, outcome_reward
from pylatro_agent.transfer import (
    check_seed_separation,
    evaluate_teacher_artifact,
    paired_statistics,
    restore_configuration,
    sha256,
    source_hashes,
    summarize,
)

if __package__:
    from .report_outcome_ppo import frozen_checkpoints, verified_panel
else:
    from report_outcome_ppo import frozen_checkpoints, verified_panel


def compare_teacher(candidates, teacher):
    """Keep the deterministic teacher's physical denominator distinct from paired alignment rows."""
    if set(candidates) != {0, 1}:
        raise ValueError("Both fixed training replicas are required")
    teacher_summary = summarize(teacher)
    if any(row.get("sampling_seed") is not None for row in teacher):
        raise ValueError("The teacher reference must run once per game seed")
    pooled = [dict(row, sampling_seed=f"train{seed}/sample{row['sampling_seed']}")
              for seed in (0, 1) for row in candidates[seed]]
    reference = [dict(row, sampling_seed=f"train{seed}/sample{sampling}")
                 for seed in (0, 1) for sampling in (0, 1, 2) for row in teacher]
    paired = paired_statistics(pooled, reference)
    paired["aligned_rows_per_arm"] = paired.pop("games_per_arm")
    paired["physical_candidate_games"] = len(pooled)
    paired["physical_teacher_games"] = len(teacher)
    return dict(candidate=summarize(pooled), teacher=teacher_summary, paired=paired,
                interpretation="Descriptive frozen-candidate benchmark, not an acceptance or selection rule. "
                               "The deterministic teacher is repeated only to align labels for pairing. "
                               "Average all six student repeats within game seed before bootstrapping.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--teacher-generation", type=Path,
                        default=Path("cache/datasets/5k-balatro-games/metadata/generation.json"))
    args = parser.parse_args()
    root = args.root
    selection_path = root / "candidate_selection.json"
    selection, selection_hash = read_json(selection_path), sha256(selection_path)
    frozen_checkpoints(selection)
    if read_json(root / "confirmation_stage/manifest.json")["status"] != "complete":
        raise ValueError("Finish all frozen candidate and control panels before the teacher benchmark")
    if any(job["status"] == "running" for job in read_json(root / "compute.json")["jobs"]):
        raise ValueError("Finish accelerator work before this final CPU reference")
    protocol = read_json(root / "manifest.json")
    seeds = protocol["confirmation_seeds"]
    if len(seeds) != 400 or len(set(seeds)) != 400:
        raise ValueError("The teacher benchmark requires all 400 confirmation seeds")
    separation = check_seed_separation(
        seeds, [item["path"] for item in protocol["seed_separation"]]
        + list(root.glob("trials/*/tensorboard/training_episodes.jsonl")))
    generation = read_json(args.teacher_generation)
    required = dict(policy="V68", shop_policy="search", grow_scalers=True, blind_rollout=None,
                    deck="b_blue", stake=1, win_ante=8)
    if any(generation.get(key) != value for key, value in required.items()):
        raise ValueError("The original dataset's explicit teacher and game settings do not match")
    _, reward = restore_configuration(load_checkpoint_payload(selection["checkpoints"]["control"], "cpu"))
    if reward != outcome_reward():
        raise ValueError("The teacher environment must use the study's declared reward configuration")
    candidates, evidence = {}, {}
    for seed in (0, 1):
        name = f"{selection['recipe']}_seed{seed}_u{selection['updates']}_confirmation400"
        panel, evidence[seed] = verified_panel(root / "evaluations" / name, seeds,
                                             selection["confirmation_device"])
        candidates[seed] = panel["sampled"]
    directory = root / "teacher_confirmation400"
    directory.mkdir()
    manifest = dict(
        status="running", recorded_utc=datetime.now(UTC).isoformat(),
        selection_sha256=selection_hash, study_manifest_sha256=sha256(root / "manifest.json"),
        source_hashes=source_hashes(), device="cpu", workers=4, seeds=seeds,
        generation=dict(path=str(args.teacher_generation), sha256=sha256(args.teacher_generation)),
        seed_separation=separation,
        teacher=dict(version="V68", shop_policy="search", grow_scalers=True, blind_rollout=None),
        deck="b_blue", stake=1, win_ante=8, physical_games=400, sampling_seed=None,
        initialization_checkpoint=selection["checkpoints"]["control"],
        model_execution=False, accelerator_seconds=0,
        protocol="Reuse the shared strict teacher evaluator, with no student proposals or learning. "
                 "The checkpoint supplies the unchanged environment reward configuration; its actor is unused. "
                 "The teacher has full simulator access. Teacher-probe isolation is covered separately by regressions.",
        candidate_evidence=evidence,
    )
    atomic_json(directory / "manifest.json", manifest)
    rows, started = [], time.monotonic()
    try:
        evaluation = directory / "evaluation"
        rows = evaluate_teacher_artifact(
            seeds, reward, args.teacher_generation, evaluation, workers=4,
            metadata=dict(selection_sha256=selection_hash, seed_separation=separation,
                          accelerator_seconds=0, device="cpu", model_execution=False))
        if {row["seed"] for row in rows} != set(seeds) or len(rows) != 400:
            raise ValueError("The teacher benchmark has missing or duplicate seeds")
        if sha256(selection_path) != selection_hash:
            raise ValueError("The frozen candidate changed during the teacher benchmark")
        if any(row["stalled"] or row["censored"] for row in rows):
            raise ValueError("The teacher benchmark has incomplete games")
        outcomes = evaluation / "outcomes.jsonl"
        result = compare_teacher(candidates, rows)
        atomic_json(directory / "comparison.json", result)
        manifest.update(status="complete", outcomes=str(outcomes), outcomes_sha256=sha256(outcomes),
                        comparison_sha256=sha256(directory / "comparison.json"))
        print(json.dumps(result), flush=True)
    except BaseException as exc:
        manifest.update(status="failed", error=repr(exc))
        raise
    finally:
        manifest.update(completed_games=len(rows), elapsed_cpu_seconds=time.monotonic() - started)
        atomic_json(directory / "manifest.json", manifest)


if __name__ == "__main__":
    main()
