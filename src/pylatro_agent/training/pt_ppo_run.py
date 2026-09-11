"""Single-GPU teacher pretraining followed by fresh-critic archive PPO.

Run under a process supervisor. Each stage gets a separate process, releasing
the in-memory teacher dataset and CUDA allocator before PPO starts. This is a
new-run launcher, not a resume command; existing manifests are never replaced.
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import os
import shutil
import subprocess
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def _positive_int(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--games", type=_positive_int, default=1000)
    parser.add_argument("--epochs", type=_positive_int, default=5)
    parser.add_argument("--updates", type=_positive_int, default=1000)
    parser.add_argument("--workers", type=_positive_int, default=8)
    parser.add_argument("--pt-batch", type=_positive_int, default=160)
    parser.add_argument("--envs", type=_positive_int, default=16)
    parser.add_argument("--rollout-length", type=_positive_int, default=256)
    parser.add_argument("--ppo-batch", type=_positive_int, default=320)
    parser.add_argument("--micro-batch", type=_positive_int, default=160)
    parser.add_argument("--eval-cpu-threads", type=int, default=0)
    parser.add_argument("--eval-workers", type=int, default=4)
    parser.add_argument("--eval-games", type=_positive_int, default=100)
    parser.add_argument("--eval-interval", type=_positive_int, default=10)
    parser.add_argument("--checkpoint-interval", type=_positive_int, default=100)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--stage", choices=("supervised", "ppo"), help=argparse.SUPPRESS)
    return parser


def _configs(args: argparse.Namespace):
    from ..agent import AgentConfig
    from ..archive import ArchiveConfig
    from ..reward import RewardConfig
    from .ppo_config import PPOConfig, _validate_ppo_config
    from .ppo_evaluation import evaluation_seed_list
    from .supervised import SupervisedConfig

    root = args.run_dir.resolve()
    agent = AgentConfig(precision=args.precision)
    supervised = SupervisedConfig(
        num_games=args.games,
        max_epochs=args.epochs,
        batch_size=args.pt_batch,
        num_workers=args.workers,
        chunk_size=args.games,
        device="cuda:0",
        save_dir=str(root / "checkpoints/supervised"),
        log_dir=str(root / "events/supervised"),
    )
    ppo = PPOConfig(
        seed=args.seed,
        precision=args.precision,
        device="cuda:0",
        num_envs=args.envs,
        rollout_length=args.rollout_length,
        total_updates=args.updates,
        total_timesteps=args.updates * args.envs * args.rollout_length,
        mini_batch_size=args.ppo_batch,
        micro_batch_size=args.micro_batch,
        ppo_epochs=4,
        lr=3e-6,
        gamma=supervised.gamma,
        win_ante=8,
        archive_config=ArchiveConfig(),
        reward_config=RewardConfig(objective="milestone"),
        eval_games=args.eval_games,
        eval_workers=args.eval_workers,
        eval_cpu_threads=args.eval_cpu_threads,
        eval_interval=args.eval_interval,
        checkpoint_interval=args.checkpoint_interval,
        log_interval=1,
        save_dir=str(root / "checkpoints/ppo"),
        log_dir=str(root / "events/ppo"),
        stop_request_path=str(root / "monitor_stop_request.json"),
    )
    _validate_ppo_config(ppo)
    supervised.excluded_seeds = tuple(evaluation_seed_list(ppo.eval_games, ppo.eval_seeds))
    return agent, supervised, ppo


def _status(root: Path, stage: str, state: str, **extra) -> None:
    temporary = root / "status.json.tmp"
    temporary.write_text(
        json.dumps(
            {
                "stage": stage,
                "state": state,
                "updated_utc": datetime.now(UTC).isoformat(),
                "pid": os.getpid(),
                **extra,
            },
            indent=2,
        )
        + "\n"
    )
    temporary.replace(root / "status.json")


def _run_stage(args: argparse.Namespace) -> None:
    import random

    import numpy as np
    import torch

    from .ppo import train_ppo
    from .supervised import train_supervised

    agent, supervised, ppo = _configs(args)
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Select exactly one GPU with CUDA_VISIBLE_DEVICES before launching")
    if args.precision == "bf16" and not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError("This run requires native CUDA BF16 support")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    logger.info("GPU: %s; compute=%s; parameters and Adam moments=FP32", torch.cuda.get_device_name(0), args.precision)
    if args.stage == "supervised":
        from .fast_generate import generate_training_data

        # Spawn workers explicitly: neither the CUDA context nor parent threads
        # are safe to fork. Keep the dataset in RAM, not duplicate disk pickles.
        records = generate_training_data(
            supervised.num_games,
            num_workers=supervised.num_workers,
            chunk_size=supervised.chunk_size,
            min_ante=supervised.min_ante,
            gamma=supervised.gamma,
            win_ante=supervised.win_ante,
            excluded_seeds=supervised.excluded_seeds,
        )
        if not records:
            raise RuntimeError("Teacher generation returned no records")
        train_supervised(supervised, agent_config=agent, records=records)
    else:
        checkpoint = Path(supervised.save_dir) / f"supervised_epoch{args.epochs}.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        train_ppo(ppo, agent_config=agent, actor_transfer_path=str(checkpoint))


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _parser().parse_args(argv)
    # Fresh executable per stage, and spawn-only simulator workers on CUDA.
    multiprocessing.set_start_method("spawn", force=True)
    if args.stage:
        _run_stage(args)
        return
    root = args.run_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    agent, supervised, ppo = _configs(args)
    manifest = {
        "agent": asdict(agent),
        "supervised": asdict(supervised),
        "ppo": asdict(ppo),
        "seed": args.seed,
        "weights_dtype": "float32",
        "optimizer_moments_dtype": "float32",
        "created_utc": datetime.now(UTC).isoformat(),
        "teacher_seed_note": "Worker-index seed streams; worker assignment is scheduling-dependent",
    }
    with (root / "manifest.json").open("x") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    forwarded_args = sys.argv[1:] if argv is None else argv
    for stage in ("supervised", "ppo"):
        try:
            if shutil.disk_usage(root).free < 3 * 1024**3:
                raise RuntimeError("Less than 3 GiB free for training checkpoints")
            _status(root, stage, "running")
            logger.info("Starting %s stage", stage)
            subprocess.run(
                [
                    sys.executable,
                    "-u",
                    "-m",
                    "pylatro_agent.training.pt_ppo_run",
                    *forwarded_args,
                    "--stage",
                    stage,
                ],
                check=True,
            )
        except BaseException as exc:
            _status(root, stage, "failed", error=str(exc))
            raise
        _status(root, stage, "complete")
    _status(root, "pipeline", "complete")


if __name__ == "__main__":
    main()
