#!/usr/bin/env python3
"""Verify outcome-study inputs, episode provenance, learning curves and compute totals."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from pylatro_agent.learning_bottleneck import atomic_json, read_json
from pylatro_agent.outcome_study import ROOT, outcome_reward
from pylatro_agent.transfer import check_seed_separation, sha256

CURVE_TAGS = (
    "ppo/policy_loss", "ppo/approx_kl", "ppo/clip_fraction", "ppo/minibatch_fraction",
    "ppo/kl_rollback_event", "ppo/actual_lr", "ppo/explained_variance", "ppo/entropy_raw",
    "ppo/action_type_entropy_normalized", "critic/return_huber", "critic/outcome_nll",
    "critic/terminal_value_mean", "critic/return_residual_mean", "critic/expected_return_mean",
    "critic/rollout_labels/label_coverage", "performance/update_wall_seconds",
    "performance/optimization_wall_seconds", "performance/rollout_wall_seconds",
    "terminal_replay/samples", "terminal_replay/updates_applied",
)


def episode_summary(rows):
    return dict(games=len(rows), game_seeds=len({r["seed"] for r in rows}),
                wins=sum(r["won"] for r in rows),
                win_rate=sum(r["won"] for r in rows) / len(rows) if rows else None,
                states_in_completed_episodes=sum(r["steps"] for r in rows),
                mean_final_ante=sum(r["final_ante"] for r in rows) / len(rows) if rows else None)


def validate_training_rows(rows, reserved):
    for row in rows:
        if row["seed"] in reserved:
            raise ValueError(f"Reserved training-game seed: {row['seed']}")
        if row["reward"] != 10 * row["won"]:
            raise ValueError("Training reward differs from the pure outcome objective")
        if row["stalled"] or row["censored"] or row["archive_start"] or row["deck"] != "b_blue":
            raise ValueError("Incomplete, archived or wrong-deck training episode")


def verify(root, *, final=False):
    started = time.monotonic()
    protocol = read_json(root / "manifest.json")
    preserved = read_json(Path("eval/learning_bottleneck_20260913/original_hashes.json"))
    for path, digest in preserved.items():
        if sha256(path) != digest:
            raise ValueError(f"Original artifact changed: {path}")
    if sha256(protocol["checkpoint"]) != protocol["checkpoint_sha256"]:
        raise ValueError("Starting actor checkpoint changed")
    if protocol["reward_config"] != asdict(outcome_reward()):
        raise ValueError("The study reward objective changed")
    reserved = set(protocol["development_seeds"] + protocol["confirmation_seeds"]
                   + protocol["critic_validation_seeds"] + protocol["diagnostic_seeds"])
    check_seed_separation(sorted(reserved), [item["path"] for item in protocol["seed_separation"]])
    for item in protocol["seed_separation"]:
        if sha256(item["path"]) != item["sha256"]:
            raise ValueError("An earlier data manifest changed")
    earlier_panels = set(range(100000, 100200)) | set(range(110000, 110400))
    for path in (Path("eval/transfer_20260912/manifest.json"),
                 Path("eval/learning_bottleneck_20260913/manifest.json"),
                 Path("eval/action_refactor_20260913/manifest.json")):
        if path.exists():
            manifest = read_json(path)
            for key in ("development_seeds", "confirmation_seeds"):
                earlier_panels.update(manifest.get(key, []))

    collection = read_json(root / "critic_games/manifest.json")
    outcomes = root / "critic_games/outcomes.jsonl"
    if collection["status"] != "complete" or sha256(outcomes) != collection["outcomes_sha256"]:
        raise ValueError("Critic collection is incomplete or modified")
    games = [json.loads(line) for line in outcomes.read_text().splitlines()]
    expected_seeds = set(protocol["critic_training_seeds"] + protocol["critic_validation_seeds"])
    if len(games) != len(expected_seeds) or {r["seed"] for r in games} != expected_seeds:
        raise ValueError("Critic collection seed panel differs")
    for row in games:
        if sha256(row["path"]) != row["sha256"]:
            raise ValueError("A critic trajectory cache changed")
        expected_split = "training" if row["seed"] in protocol["critic_training_seeds"] else "validation"
        if row["split"] != expected_split:
            raise ValueError("Critic train/validation membership changed")
    critics = {}
    for seed in (0, 1):
        critic = read_json(root / f"critic_seed{seed}/manifest.json")
        if not critic["actor_unchanged"] or sha256(critic["checkpoint"]) != critic["checkpoint_sha256"]:
            raise ValueError("Critic preparation changed or modified the actor")
        critics[seed] = critic

    trials, first_rollouts = {}, {}
    for path in sorted(root.glob("trials/*/recipe.json")):
        recipe = read_json(path)
        config = recipe["ppo_config"]
        if recipe["study_manifest_sha256"] != sha256(root / "manifest.json"):
            raise ValueError("The pinned study protocol changed")
        if (config["reward_config"] != asdict(outcome_reward()) or config["gamma"] != 1
                or config["sil_coeff"] != 0 or config["return_path_coeff"] != 0
                or config["archive_config"] is not None):
            raise ValueError("Trial introduced an undeclared objective or imitation intervention")
        held_out = set(config["eval_seeds"] + config["training_excluded_seeds"])
        if not reserved <= held_out:
            raise ValueError("A trial omitted reserved game seeds")
        core = {name: digest for name, digest in recipe["source_hashes"].items() if name.startswith("src/")}
        # Verify the code used for this study. A later isolated maintenance fix
        # can be integrated only after this final study audit has been saved.
        if any(sha256(name) != digest for name, digest in core.items()):
            raise ValueError("Core code changed while a pinned study was active")
        legs = [read_json(p) for p in sorted(path.parent.glob("leg_*/manifest.json"))]
        complete = [leg for leg in legs if leg["status"] == "complete"]
        if final and len(complete) != len(legs):
            raise ValueError(f"Unfinished training leg: {path.parent}")
        for leg in complete:
            if leg["recipe_sha256"] != sha256(path) or sha256(leg["checkpoint"]) != leg["checkpoint_sha256"]:
                raise ValueError("A completed PPO leg or its recipe changed")
            if sha256(leg["source_checkpoint"]) != leg["source_checkpoint_sha256"]:
                raise ValueError("A PPO leg's initialization changed")
        rows_path = path.parent / "tensorboard/training_episodes.jsonl"
        rows = [json.loads(line) for line in rows_path.read_text().splitlines()] if rows_path.exists() else []
        validate_training_rows(rows, reserved)
        by_update = defaultdict(list)
        for row in rows:
            by_update[row["update"]].append(row)
        earlier_overlap = sorted({r["seed"] for r in rows} & earlier_panels)
        if earlier_overlap:
            raise ValueError(f"Observed PPO episodes overlap previous evaluation panels: {earlier_overlap}")
        curves = {}
        events = EventAccumulator(str(path.parent / "tensorboard"), size_guidance={"scalars": 0}).Reload()
        for tag in CURVE_TAGS:
            if tag not in events.Tags()["scalars"]:
                continue
            values = events.Scalars(tag)
            if any(not math.isfinite(value.value) for value in values):
                raise ValueError(f"Non-finite learning metric: {path.parent}/{tag}")
            if len({value.step for value in values}) != len(values):
                raise ValueError("Duplicate training update metrics")
            curves[tag] = [dict(update=v.step, value=v.value) for v in values]
        intervals = [(leg["start_update"], leg["update_count"]) for leg in complete]
        physical_updates = sum(end - start for start, end in intervals)
        trials[path.parent.name] = dict(
            recipe_sha256=sha256(path), recipe=recipe["recipe"], training_seed=recipe["training_seed"],
            completed_legs=complete, incomplete_legs=[leg for leg in legs if leg["status"] != "complete"],
            physical_completed_updates=physical_updates,
            physical_collected_steps=physical_updates * config["num_envs"] * config["rollout_length"],
            completed_episodes=episode_summary(rows), episode_file_sha256=sha256(rows_path) if rows else None,
            per_update={update: episode_summary(values) for update, values in sorted(by_update.items())},
            curves=curves, earlier_panel_overlap=earlier_overlap,
            repeated_training_seed_counts={str(seed): count for seed, count in Counter(r["seed"] for r in rows).items()
                                           if count > 1},
        )
        if recipe.get("branch") and by_update:
            first_update = min(by_update)
            logged_updates = {item["update"] for item in curves.get("ppo/policy_loss", [])}
            if first_update not in logged_updates and not final:
                first_rollouts[path.parent.name] = dict(update=first_update, status="still_collecting_or_updating")
                continue
            parent_trial = Path(recipe["branch"]["parent_checkpoint"]).parent.parent
            parent_rows = [json.loads(line) for line in
                           (parent_trial / "tensorboard/training_episodes.jsonl").read_text().splitlines()]
            reference = [row for row in parent_rows if row["update"] == first_update]
            def key(row):
                return row["environment"], row["seed"], row["steps"]

            matched = sorted(by_update[first_update], key=key) == sorted(reference, key=key)
            first_rollouts[path.parent.name] = dict(
                update=first_update, parent=str(parent_trial), completed_games=len(reference), identical=matched,
                scope="Completed-episode seeds/actions-counts/outcomes; does not claim unfinished trajectory equality",
            )
            if not matched:
                raise ValueError("Declared intervention does not reproduce its control's initial rollout outcomes")

    ledger = read_json(root / "compute.json")
    elapsed = sum(job["elapsed_seconds"] for job in ledger["jobs"])
    if abs(elapsed - ledger["elapsed_seconds"]) > 1e-6 or elapsed > ledger["limit_seconds"]:
        raise ValueError("Compute accounting differs or exceeds the allowance")
    if final and any(job["status"] == "running" for job in ledger["jobs"]):
        raise ValueError("An accelerator job is still running")
    result = dict(
        passed=True, final=final, original_artifacts_verified=len(preserved), critic_games=len(games),
        critic_states=sum(row["steps"] for row in games), critic_preparations=critics, trials=trials,
        first_rollout_matching=first_rollouts, compute=ledger, elapsed_cpu_seconds=time.monotonic() - started,
        scope="Training returns and seed provenance, immutable inputs/checkpoints, learning curves and compute. "
              "Fresh-panel outcomes and paired acceptance are verified separately by report_outcome_ppo.py. "
              "Completed-game training rates are descriptive, since policies change during collection.",
        source_sha256=sha256(__file__),
    )
    atomic_json(root / ("verification_final.json" if final else "verification_progress.json"), result)
    print(json.dumps(dict(passed=True, final=final, trials=len(trials), protected_artifacts=len(preserved),
                          compute_hours=elapsed / 3600, first_rollout_matching=first_rollouts)), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()
    verify(args.root, final=args.final)
