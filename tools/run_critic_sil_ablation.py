#!/usr/bin/env python3
"""Run and summarize a Scalar/HL-Gauss by SIL PPO ablation.

Every arm uses weights-only initialization from one checkpoint, reinitializes
its critic, and receives the same count-based critic warmup. Advantage clipping
is disabled so the core matrix changes only the critic type and SIL treatment.

Example:

    uv run --extra agent python tools/run_critic_sil_ablation.py run \
        --checkpoint checkpoints/ppo/ppo_best_eval.pt \
        --output-dir ablations/critic_sil \
        --updates 100 --seeds 0,1,2 --dry-run

    uv run --extra agent python tools/run_critic_sil_ablation.py summarize \
        --output-dir ablations/critic_sil
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shlex
import statistics
import subprocess
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 3
MANIFEST_NAME = "manifest.json"
SUMMARY_NAME = "summary.json"

METRIC_TAGS = (
    "eval/win_rate",
    "rollout/win_rate",
    "rollout/ep_reward_mean",
    "ppo/explained_variance",
    "ppo/value_mse",
    "sil/loss",
    "sil/loss_weighted",
    "sil/coeff",
    "sil/advantage_mean",
    "sil/advantage_p50",
    "sil/advantage_p80",
    "sil/advantage_p95",
    "sil/advantage_p99",
    "sil/gate_mean",
    "sil/gate_positive_fraction",
    "sil/gate_saturation_fraction",
    "sil/gate_open_threshold",
    "sil/gate_saturation_threshold",
    "sil/gate_weight_from_wins_fraction",
    "sil/noise_floor_rejected_fraction",
    "sil/logical_minibatches_attempted",
    "sil/logical_minibatches_applied",
    "sil/actor_grad_norm_weighted",
    "sil/ppo_actor_grad_norm",
    "sil/ppo_actor_grad_norm_ratio",
    "sil/ppo_actor_grad_cosine",
    "sil/buffer_episodes",
    "sil/buffer_transitions",
    "sil/buffer_win_fraction",
    "sil/episodes_added_total",
    "sil/stalled_episodes_dropped_total",
    "sil/overflow_episodes_dropped_total",
    "sil/samples",
    "sil/unique_episodes_sampled",
    "sil/max_samples_from_one_episode",
    "sil/sample_win_fraction",
)

RESERVED_SHARED_FLAGS = {
    "--additional-updates",
    "--advantage-clip-sigma",
    "--checkpoint-dir",
    "--critic-warmup-min-ev",
    "--critic-warmup-updates",
    "--hl-gauss",
    "--log-dir",
    "--pretrained",
    "--reinit-value-head",
    "--resume",
    "--seed",
    "--sil-advantage-floor",
    "--sil-buffer-episodes",
    "--sil-batch-size",
    "--sil-coeff",
    "--sil-coeff-final",
    "--sil-decay-fraction",
    "--sil-gate-open-percentile",
    "--sil-gate-saturation-percentile",
    "--sil-include-teacher-forced",
    "--sil-logical-minibatches-per-update",
    "--sil-objective",
    "--sil-samples-per-episode",
    "--updates",
}

XMULT_BASE_ARGS = (
    "--reward-v2",
    "--planet-match-shaping",
    "--build-curve-shaping",
    "--planet-unmatched-use-penalty-coeff",
    "0.4",
    "--planet-unmatched-claim-penalty-coeff",
    "0.4",
    "--heuristic-distill-coeff",
    "0",
    "--win-ante",
    "5",
    "--envs",
    "32",
    "--batch",
    "512",
    "--lr",
    "1.5e-5",
    "--entropy-coeff",
    "0.005",
    "--sil-min-episodes",
    "4",
    "--eval-games",
    "300",
    "--eval-interval",
    "15",
    "--min-minibatch-fraction",
    "0.5",
    "--target-kl-max",
    "0.25",
    "--log-interval",
    "1",
)


@dataclass(frozen=True)
class ArmSpec:
    arm_id: str
    critic: str
    sil_treatment: str  # one of: no_sil, advantage_sil, winning_bc


def _arm_specs() -> tuple[ArmSpec, ...]:
    """Three matched SIL arms per critic treatment."""
    specs: list[ArmSpec] = []
    for critic in ("scalar", "hl_gauss"):
        for treatment in ("no_sil", "advantage_sil", "winning_bc"):
            specs.append(ArmSpec(f"{critic}_{treatment}", critic, treatment))
    return tuple(specs)


ARM_SPECS = _arm_specs()

# Kept as a public alias because notebooks may have imported the first version.
CELL_SPECS = ARM_SPECS

# Treatments that turn SIL on (vs the no_sil control).
SIL_ON_TREATMENTS = ("advantage_sil", "winning_bc")


def _sil_treatment_flags(
    treatment: str,
    *,
    sil_coeff: float,
) -> list[str]:
    """Build the SIL flags for one matched treatment.

    Only the SIL treatment differs across arms; replay capacity, batch size,
    per-episode cap, coefficient schedule, and the logical-minibatch budget are
    held constant (shared via the runner, not here).
    """
    if treatment == "no_sil":
        return ["--sil-coeff", "0"]
    common = [
        "--sil-coeff", str(sil_coeff),
        "--sil-coeff-final", "0",
        "--sil-decay-fraction", "1.0",
        "--sil-logical-minibatches-per-update", "1",
    ]
    if treatment == "advantage_sil":
        return [*common,
            "--sil-objective", "advantage",
            "--sil-gate-open-percentile", "80",
            "--sil-gate-saturation-percentile", "95",
            "--sil-advantage-floor", "0.25",
        ]
    if treatment == "winning_bc":
        return [*common,
            "--sil-objective", "winning_bc",
        ]
    raise ValueError(f"unknown SIL treatment: {treatment}")


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_seeds(value: str) -> list[int]:
    try:
        seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--seeds must be comma-separated integers.") from exc
    if not seeds:
        raise argparse.ArgumentTypeError("--seeds must contain at least one integer.")
    if len(seeds) != len(set(seeds)):
        raise argparse.ArgumentTypeError("--seeds cannot contain duplicates.")
    if any(seed < 0 or seed > 2**32 - 1 for seed in seeds):
        raise argparse.ArgumentTypeError("Each seed must be between 0 and 4294967295.")
    return seeds


def _normalized_shared_args(args: Sequence[str]) -> list[str]:
    result = list(args)
    if result[:1] == ["--"]:
        result.pop(0)
    for token in result:
        flag = token.split("=", 1)[0]
        if flag in RESERVED_SHARED_FLAGS:
            raise ValueError(f"{flag} is controlled by the ablation runner and cannot be passed after --.")
    return result


def _build_command(
    *,
    python: Path,
    train_script: Path,
    checkpoint: Path,
    updates: int,
    checkpoint_dir: Path,
    log_dir: Path,
    arm: ArmSpec,
    seed: int,
    sil_coeff: float,
    sil_buffer_episodes: int,
    sil_batch_size: int,
    sil_samples_per_episode: int,
    critic_warmup_updates: int,
    base_args: Sequence[str],
    shared_args: Sequence[str],
) -> list[str]:
    command = [
        str(python),
        str(train_script),
        "ppo",
        "--seed",
        str(seed),
        "--pretrained",
        str(checkpoint),
        "--updates",
        str(updates),
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--log-dir",
        str(log_dir),
        "--reinit-value-head",
        "--critic-warmup-updates",
        str(critic_warmup_updates),
        "--critic-warmup-min-ev",
        "0.0",
        "--advantage-clip-sigma",
        "0.0",
    ]
    command.extend(base_args)
    if arm.critic == "hl_gauss":
        command.append("--hl-gauss")
    # Shared replay constants held identical across arms so only the SIL
    # treatment differs (no-ops for the no_sil control where sil-coeff is 0).
    command.extend([
        "--sil-buffer-episodes", str(sil_buffer_episodes),
        "--sil-batch-size", str(sil_batch_size),
        "--sil-samples-per-episode", str(sil_samples_per_episode),
    ])
    command.extend(
        _sil_treatment_flags(arm.sil_treatment, sil_coeff=sil_coeff)
    )
    command.extend(shared_args)
    return command


def create_manifest(args: argparse.Namespace, repo_root: Path) -> dict[str, Any]:
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise ValueError(f"Checkpoint does not exist or is not a file: {checkpoint}")
    if args.updates <= 0:
        raise ValueError("--updates must be greater than zero.")
    if args.sil_coeff <= 0:
        raise ValueError("--sil-coeff must be greater than zero for the SIL-on arms.")
    if args.critic_warmup_updates <= 0:
        raise ValueError("--critic-warmup-updates must be greater than zero.")

    output_dir = Path(args.output_dir).expanduser().resolve()
    manifest_path = output_dir / MANIFEST_NAME
    if manifest_path.exists():
        raise ValueError(
            f"A manifest already exists at {manifest_path}. Choose a new --output-dir to avoid mixing runs."
        )

    shared_args = _normalized_shared_args(args.shared_args)
    base_args = list(XMULT_BASE_ARGS) if args.preset == "xmult" else []
    train_script = repo_root / "train.py"
    if not train_script.is_file():
        raise ValueError(f"Required script not found: {train_script}")
    # Do not resolve the executable: virtualenv Python paths are commonly
    # symlinks, and resolving one can bypass the environment's packages.
    python = Path(args.python).expanduser()

    cells: list[dict[str, Any]] = []
    for seed in args.seeds:
        for arm in ARM_SPECS:
            cell_id = f"{arm.arm_id}_seed_{seed}"
            checkpoint_dir = output_dir / "checkpoints" / cell_id
            log_dir = output_dir / "runs" / cell_id
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            log_dir.mkdir(parents=True, exist_ok=True)
            command = _build_command(
                python=python,
                train_script=train_script,
                checkpoint=checkpoint,
                updates=args.updates,
                checkpoint_dir=checkpoint_dir,
                log_dir=log_dir,
                arm=arm,
                seed=seed,
                sil_coeff=args.sil_coeff,
                sil_buffer_episodes=args.sil_buffer_episodes,
                sil_batch_size=args.sil_batch_size,
                sil_samples_per_episode=args.sil_samples_per_episode,
                critic_warmup_updates=args.critic_warmup_updates,
                base_args=base_args,
                shared_args=shared_args,
            )
            cells.append(
                {
                    "id": cell_id,
                    "arm_id": arm.arm_id,
                    "seed": seed,
                    "critic": arm.critic,
                    "sil_treatment": arm.sil_treatment,
                    "sil_enabled": arm.sil_treatment in SIL_ON_TREATMENTS,
                    "sil_coeff": args.sil_coeff if arm.sil_treatment in SIL_ON_TREATMENTS else 0.0,
                    "checkpoint_dir": str(checkpoint_dir),
                    "log_dir": str(log_dir),
                    "train_log": str(output_dir / "job_logs" / f"{cell_id}.log"),
                    "command": command,
                    "command_shell": shlex.join(command),
                    "status": "planned",
                    "return_code": None,
                }
            )

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at": _utc_now(),
        "repo_root": str(repo_root),
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": _sha256(checkpoint),
        "output_dir": str(output_dir),
        "updates": args.updates,
        "seeds": args.seeds,
        "sil_coeff_on": args.sil_coeff,
        "sil_buffer_episodes": args.sil_buffer_episodes,
        "sil_batch_size": args.sil_batch_size,
        "sil_samples_per_episode": args.sil_samples_per_episode,
        "critic_warmup_updates": args.critic_warmup_updates,
        "critic_warmup_min_ev": 0.0,
        "advantage_clip_sigma": 0.0,
        "reinitialize_value_head": True,
        "preset": args.preset,
        "preset_train_args": base_args,
        "shared_train_args": shared_args,
        "execution": "sequential",
        "dry_run": bool(args.dry_run),
        "cells": cells,
    }
    _write_json(manifest_path, manifest)
    return manifest


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=False, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def _run_and_tee(command: Sequence[str], *, cwd: Path, log_path: Path, prefix: str) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        try:
            for line in process.stdout:
                log_handle.write(line)
                log_handle.flush()
                print(f"[{prefix}] {line}", end="", flush=True)
        except KeyboardInterrupt:
            process.terminate()
            process.wait()
            raise
        return process.wait()


def execute_manifest(manifest: dict[str, Any], *, keep_going: bool = False) -> int:
    output_dir = Path(manifest["output_dir"])
    manifest_path = output_dir / MANIFEST_NAME
    repo_root = Path(manifest["repo_root"])
    first_failure = 0
    for cell in manifest["cells"]:
        if cell.get("status") != "planned":
            continue
        cell["status"] = "running"
        cell["started_at"] = _utc_now()
        _write_json(manifest_path, manifest)
        print(f"\nStarting {cell['id']}")
        print(f"  {cell['command_shell']}")
        try:
            return_code = _run_and_tee(
                cell["command"], cwd=repo_root, log_path=Path(cell["train_log"]), prefix=cell["id"]
            )
        except KeyboardInterrupt:
            cell["status"] = "interrupted"
            cell["finished_at"] = _utc_now()
            _write_json(manifest_path, manifest)
            raise
        cell["return_code"] = return_code
        cell["status"] = "completed" if return_code == 0 else "failed"
        cell["finished_at"] = _utc_now()
        _write_json(manifest_path, manifest)
        if return_code != 0:
            first_failure = first_failure or return_code
            print(f"{cell['id']} failed with exit code {return_code}.", file=sys.stderr)
            if not keep_going:
                break
    return first_failure


def read_tensorboard_scalars(log_dir: Path, tags: Iterable[str]) -> dict[str, list[tuple[int, float]]]:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError as exc:
        raise RuntimeError(
            "TensorBoard is required to summarize runs. Use an environment with the agent extras installed."
        ) from exc
    accumulator = EventAccumulator(str(log_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    available = set(accumulator.Tags().get("scalars", []))
    return {
        tag: [(int(event.step), float(event.value)) for event in accumulator.Scalars(tag)]
        for tag in tags
        if tag in available
    }


def _metric_stats(events: Sequence[tuple[int, float]], tail: int) -> dict[str, Any] | None:
    finite = [(step, value) for step, value in events if math.isfinite(value)]
    if not finite:
        return None
    values = [value for _, value in finite]
    tail_values = values[-tail:]
    return {
        "count": len(values),
        "first_step": finite[0][0],
        "last_step": finite[-1][0],
        "first": values[0],
        "last": values[-1],
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
        "tail_count": len(tail_values),
        "tail_mean": sum(tail_values) / len(tail_values),
    }


PAIRED_EFFECT_NAMES = (
    "advantage_sil_minus_no_sil",
    "winning_bc_minus_no_sil",
    "advantage_sil_minus_winning_bc",
)


def _paired_effects(
    arms_by_id: dict[str, dict[str, Any]],
    metric: str,
    statistic: str,
    critic: str,
) -> dict[str, float] | None:
    """Three-way matched effects within one critic: treatment minus control.

    Reports ``advantage_sil - no_sil``, ``winning_bc - no_sil``, and
    ``advantage_sil - winning_bc`` for the given critic. Returns None unless
    all three arms of that critic have the requested metric/statistic.
    """
    def value(treatment: str) -> float | None:
        arm_id = f"{critic}_{treatment}"
        stats = arms_by_id.get(arm_id, {}).get("metrics", {}).get(metric)
        if stats is None or stats.get(statistic) is None:
            return None
        return float(stats[statistic])

    no_sil = value("no_sil")
    advantage = value("advantage_sil")
    winning_bc = value("winning_bc")
    if no_sil is None or advantage is None or winning_bc is None:
        return None
    return {
        f"{critic}_no_sil": no_sil,
        f"{critic}_advantage_sil": advantage,
        f"{critic}_winning_bc": winning_bc,
        "advantage_sil_minus_no_sil": advantage - no_sil,
        "winning_bc_minus_no_sil": winning_bc - no_sil,
        "advantage_sil_minus_winning_bc": advantage - winning_bc,
    }


def _aggregate(values: Sequence[float]) -> dict[str, Any] | None:
    finite = [float(value) for value in values if math.isfinite(value)]
    if not finite:
        return None
    return {
        "n": len(finite),
        "mean": statistics.fmean(finite),
        "std": statistics.pstdev(finite),
        "min": min(finite),
        "max": max(finite),
        "values": finite,
    }


MetricReader = Callable[[Path, Iterable[str]], dict[str, list[tuple[int, float]]]]


def build_summary(
    manifest: dict[str, Any], *, tail: int, metric_reader: MetricReader = read_tensorboard_scalars
) -> dict[str, Any]:
    if tail <= 0:
        raise ValueError("--tail must be greater than zero.")
    cell_summaries: list[dict[str, Any]] = []
    for cell in manifest["cells"]:
        raw_metrics = metric_reader(Path(cell["log_dir"]), METRIC_TAGS)
        metrics = {
            tag: stats
            for tag, events in raw_metrics.items()
            if (stats := _metric_stats(events, tail)) is not None
        }
        cell_summaries.append(
            {
                "id": cell["id"],
                "arm_id": cell["arm_id"],
                "seed": cell["seed"],
                "critic": cell["critic"],
                "sil_treatment": cell.get("sil_treatment"),
                "sil_enabled": cell.get("sil_enabled"),
                "status": cell.get("status"),
                "log_dir": cell["log_dir"],
                "metrics": metrics,
            }
        )

    arm_summaries: dict[str, Any] = {}
    for arm in ARM_SPECS:
        arm_cells = [cell for cell in cell_summaries if cell["arm_id"] == arm.arm_id]
        metric_summary: dict[str, Any] = {}
        for tag in METRIC_TAGS:
            statistic_summary: dict[str, Any] = {}
            for statistic_name in ("last", "tail_mean"):
                values = [
                    cell["metrics"][tag][statistic_name]
                    for cell in arm_cells
                    if tag in cell["metrics"]
                ]
                aggregate = _aggregate(values)
                if aggregate is not None:
                    aggregate["by_seed"] = {
                        str(cell["seed"]): cell["metrics"][tag][statistic_name]
                        for cell in arm_cells
                        if tag in cell["metrics"]
                    }
                    statistic_summary[statistic_name] = aggregate
            if statistic_summary:
                metric_summary[tag] = statistic_summary
        arm_summaries[arm.arm_id] = {
            "critic": arm.critic,
            "sil_treatment": arm.sil_treatment,
            "sil_enabled": arm.sil_treatment in SIL_ON_TREATMENTS,
            "metrics": metric_summary,
        }

    # Three-way matched paired effects per critic, for win rate and episode
    # reward (the acceptance criteria forbid treating reward gains without win
    # rate gains as SIL success, so both are reported).
    effects: dict[str, Any] = {}
    effect_metrics = ("eval/win_rate", "rollout/win_rate", "rollout/ep_reward_mean")
    for effect_metric in effect_metrics:
        metric_effects: dict[str, Any] = {}
        for statistic_name in ("last", "tail_mean"):
            per_seed: dict[str, Any] = {}
            for seed in manifest["seeds"]:
                seed_arms = {
                    cell["arm_id"]: cell for cell in cell_summaries if cell["seed"] == seed
                }
                for critic in ("scalar", "hl_gauss"):
                    result = _paired_effects(seed_arms, effect_metric, statistic_name, critic)
                    if result is not None:
                        per_seed[f"{critic}/seed_{seed}"] = result
            if per_seed:
                metric_effects[statistic_name] = {
                    "per_seed": per_seed,
                    "aggregate": {
                        name: _aggregate(
                            [seed_effects[name] for seed_effects in per_seed.values()]
                        )
                        for name in PAIRED_EFFECT_NAMES
                    },
                }
        if metric_effects:
            effects[effect_metric] = metric_effects

    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": _utc_now(),
        "manifest": str(Path(manifest["output_dir"]) / MANIFEST_NAME),
        "tail": tail,
        "seeds": manifest["seeds"],
        "cells": cell_summaries,
        "arms": arm_summaries,
        "paired_effects": effects,
    }


def _format_optional(value: float | None, *, percent: bool = False) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * value:.2f}%" if percent else f"{value:.4g}"


def _arm_metric_mean(arm: dict[str, Any], tag: str, statistic_name: str = "tail_mean") -> float | None:
    return arm.get("metrics", {}).get(tag, {}).get(statistic_name, {}).get("mean")


def print_summary(summary: dict[str, Any]) -> None:
    print(f"\nArm summary across {len(summary['seeds'])} seed(s)")
    print(
        f"{'arm':<26} {'eval tail':>10} {'rollout tail':>13} "
        f"{'ep reward':>10} {'SIL gate':>9} {'SIL ratio':>9}"
    )
    for arm_spec in ARM_SPECS:
        arm = summary["arms"][arm_spec.arm_id]
        print(
            f"{arm_spec.arm_id:<26} "
            f"{_format_optional(_arm_metric_mean(arm, 'eval/win_rate'), percent=True):>10} "
            f"{_format_optional(_arm_metric_mean(arm, 'rollout/win_rate'), percent=True):>13} "
            f"{_format_optional(_arm_metric_mean(arm, 'rollout/ep_reward_mean')):>10} "
            f"{_format_optional(_arm_metric_mean(arm, 'sil/gate_mean')):>9} "
            f"{_format_optional(_arm_metric_mean(arm, 'sil/ppo_actor_grad_norm_ratio')):>9}"
        )
    labels = (
        ("advantage_sil_minus_no_sil", "advantage_sil - no_sil"),
        ("winning_bc_minus_no_sil", "winning_bc - no_sil"),
        ("advantage_sil_minus_winning_bc", "advantage_sil - winning_bc"),
    )
    for effect_metric, metric_effects in summary.get("paired_effects", {}).items():
        for statistic_name, effects in metric_effects.items():
            print(f"\nPaired {effect_metric} effects ({statistic_name})")
            for name, label in labels:
                aggregate = effects["aggregate"].get(name)
                if aggregate is None:
                    continue
                pct = "%" if "win_rate" in effect_metric else ""
                value = aggregate["mean"] * (100.0 if pct else 1.0)
                print(
                    f"  {label + ':':<30} {value:+.2f}{pct} "
                    f"(sd {aggregate['std'] * (100.0 if pct else 1.0):.3f}, n={aggregate['n']})"
                )
    if not summary.get("paired_effects"):
        print(
            "\nPaired effects need eval/win_rate data for all three arms of at "
            "least one critic/seed pairing."
        )


def _load_manifest(output_dir: Path) -> dict[str, Any]:
    manifest_path = output_dir.expanduser().resolve() / MANIFEST_NAME
    if not manifest_path.is_file():
        raise ValueError(f"Manifest not found: {manifest_path}")
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported manifest schema {manifest.get('schema_version')}; expected {SCHEMA_VERSION}."
        )
    return manifest


def _add_run_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("run", help="Create and optionally execute the seeded ablation jobs.")
    parser.add_argument("--checkpoint", required=True, help="Common weights checkpoint for every job.")
    parser.add_argument("--output-dir", required=True, help="New directory for manifests, runs, and checkpoints.")
    parser.add_argument("--updates", type=int, required=True, help="PPO updates per job.")
    parser.add_argument(
        "--seeds", type=_parse_seeds, default=[0], help="Comma-separated training seeds (default: 0)."
    )
    parser.add_argument(
        "--sil-coeff",
        type=float,
        default=0.01,
        help="SIL coefficient for SIL-on arms, decaying to 0 (default: 0.01).",
    )
    parser.add_argument(
        "--sil-buffer-episodes",
        type=int,
        default=256,
        help="Replay capacity held constant across arms (default: 256).",
    )
    parser.add_argument(
        "--sil-batch-size",
        type=int,
        default=64,
        help="SIL batch size held constant across arms (default: 64).",
    )
    parser.add_argument(
        "--sil-samples-per-episode",
        type=int,
        default=8,
        help="Per-episode sample cap held constant across arms (default: 8).",
    )
    parser.add_argument(
        "--critic-warmup-updates",
        type=int,
        default=20,
        help="Count-based critic-only warmup applied to every job (default: 20).",
    )
    parser.add_argument(
        "--preset",
        choices=("xmult", "none"),
        default="xmult",
        help="Shared training profile. xmult reproduces the historical stalled run (default: xmult).",
    )
    parser.add_argument("--python", default=sys.executable, help="Python executable used to launch train.py.")
    parser.add_argument("--dry-run", action="store_true", help="Write the manifest and print commands only.")
    parser.add_argument("--keep-going", action="store_true", help="Continue with later jobs after a failure.")
    parser.add_argument(
        "shared_args", nargs=argparse.REMAINDER, help="Arguments after -- are appended to every train.py command."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Scalar/HL-Gauss by SIL PPO ablation.")
    subparsers = parser.add_subparsers(dest="action", required=True)
    _add_run_parser(subparsers)
    continuation = subparsers.add_parser(
        "continue",
        help="Execute only planned jobs in an existing manifest, skipping completed jobs.",
    )
    continuation.add_argument("--output-dir", required=True, help="Directory containing manifest.json.")
    continuation.add_argument("--keep-going", action="store_true", help="Continue after a job failure.")
    summarize = subparsers.add_parser("summarize", help="Summarize TensorBoard logs for an existing job set.")
    summarize.add_argument("--output-dir", required=True, help="Directory containing manifest.json.")
    summarize.add_argument("--tail", type=int, default=5, help="Eval points averaged for tail metrics (default: 5).")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    try:
        if args.action == "run":
            manifest = create_manifest(args, repo_root)
            print(f"Manifest: {Path(manifest['output_dir']) / MANIFEST_NAME}")
            for cell in manifest["cells"]:
                print(f"\n{cell['id']}\n  {cell['command_shell']}")
            if args.dry_run:
                return 0
            return_code = execute_manifest(manifest, keep_going=args.keep_going)
            if return_code != 0:
                return return_code
            summary = build_summary(manifest, tail=5)
            summary_path = Path(manifest["output_dir"]) / SUMMARY_NAME
            _write_json(summary_path, summary)
            print_summary(summary)
            print(f"\nSummary JSON: {summary_path}")
            return 0
        if args.action == "continue":
            manifest = _load_manifest(Path(args.output_dir))
            return_code = execute_manifest(manifest, keep_going=args.keep_going)
            if return_code != 0:
                return return_code
            summary = build_summary(manifest, tail=5)
            summary_path = Path(manifest["output_dir"]) / SUMMARY_NAME
            _write_json(summary_path, summary)
            print_summary(summary)
            print(f"\nSummary JSON: {summary_path}")
            return 0
        manifest = _load_manifest(Path(args.output_dir))
        summary = build_summary(manifest, tail=args.tail)
        summary_path = Path(manifest["output_dir"]) / SUMMARY_NAME
        _write_json(summary_path, summary)
        print_summary(summary)
        print(f"\nSummary JSON: {summary_path}")
        return 0
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
