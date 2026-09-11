#!/usr/bin/env python3
"""Evaluate a PPO/supervised checkpoint's win rate over N games.

Loads a v8 checkpoint (full PPO resume checkpoint or weights-only), rebuilds
the model from the embedded agent_config when available, and runs the same
greedy ``evaluate_model`` loop used during PPO eval. Checkpoints without an
embedded config may supply the architecture through the CLI flags, but their
weights must still match the v8 model exactly.

Example:

    uv run --extra agent python tools/evaluate_checkpoint.py \\
        --checkpoint checkpoints/ppo/ppo_update500.pt \\
        --games 500 --device mps --eval-device cpu \\
        --win-ante 4 --rollout-temperature 0.85
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("evaluate_checkpoint")


def _resolve_device(value: str | None) -> str:
    if value is not None:
        return value
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint's win rate over N games.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the checkpoint to evaluate.")
    parser.add_argument("--games", type=int, default=500, help="Number of eval games (default: 500).")
    parser.add_argument(
        "--eval-cpu-threads", type=int, default=0,
        help="CPU evaluation inference threads (0: choose by model width and CPU affinity).",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=32,
        help="Games advanced in lockstep per policy forward pass (default: 32).",
    )
    parser.add_argument("--eval-workers", type=int, default=4, help="CPU engine workers; 0 runs in-process.")
    parser.add_argument("--device", type=str, default=None, help="Model load device (default: auto-detect).")
    parser.add_argument(
        "--eval-device",
        type=str,
        default=None,
        help=(
            "Device override for the eval games themselves (cpu/mps/cuda). Useful to "
            "move eval forward-pass allocations off the GPU. Defaults to --device."
        ),
    )
    parser.add_argument("--win-ante", type=int, default=None, help="Victory ante cap (default: engine default 8).")
    parser.add_argument(
        "--max-idle-steps",
        type=int,
        default=256,
        help="Per-env stall limit during eval (default: 256).",
    )
    parser.add_argument(
        "--rollout-temperature",
        type=float,
        default=1.0,
        help="Softmax temperature applied at the eval forward pass (default: 1.0).",
    )
    parser.add_argument(
        "--stake",
        type=int,
        default=1,
        help="Stake (difficulty tier) the eval envs run at (default: 1).",
    )
    # Architecture overrides (used when the checkpoint lacks an agent_config).
    parser.add_argument("--d-model", type=int, default=384, help="Model dimension (default: 384).")
    parser.add_argument("--n-layers", type=int, default=12, help="Transformer layers (default: 12).")
    parser.add_argument("--n-heads", type=int, default=8, help="Transformer heads (default: 8).")
    parser.add_argument("--d-ff", type=int, default=1536, help="Transformer feed-forward dim (default: 1536).")
    parser.add_argument(
        "--fixed-seeds",
        action="store_true",
        help=(
            "Evaluate on the versioned EVAL_SEEDS_V1 list (400 fixed seeds) and "
            "persist per-seed outcomes to a JSON/CSV next to the checkpoint. This "
            "enables paired comparisons against any other policy evaluated on the "
            "same seeds, reducing the variance of the win-rate difference by an "
            "order of magnitude vs unpaired eval."
        ),
    )
    parser.add_argument(
        "--heuristic-baseline",
        action="store_true",
        help=(
            "Evaluate the heuristic teacher on the fixed seed list (requires "
            "--fixed-seeds). Produces the baseline every learned checkpoint must beat."
        ),
    )
    parser.add_argument(
        "--paired-outcomes",
        type=str,
        default=None,
        help=(
            "Path to a per-seed outcomes JSON (produced by a prior --fixed-seeds run) "
            "to compare against. Reports the win-rate delta with a McNemar test and a "
            "paired-bootstrap 95%% confidence interval. Requires --fixed-seeds."
        ),
    )
    parser.add_argument(
        "--outcomes-path",
        type=str,
        default=None,
        help=(
            "Where to persist the per-seed outcomes JSON when --fixed-seeds is set. "
            "Defaults to <checkpoint>.outcomes.json next to the checkpoint."
        ),
    )
    args = parser.parse_args()

    if args.paired_outcomes is not None and not args.fixed_seeds:
        parser.error("--paired-outcomes requires --fixed-seeds.")
    if args.heuristic_baseline and not args.fixed_seeds:
        parser.error("--heuristic-baseline requires --fixed-seeds.")

    load_device = _resolve_device(args.device)
    eval_device = _resolve_device(args.eval_device or args.device)

    from pylatro import load_game_data

    data = load_game_data()

    # === Fixed-seed paired-eval path ===
    if args.fixed_seeds:
        from pylatro_agent.eval import (
            EVAL_SEEDS_V1,
            evaluate_heuristic_on_seeds,
            evaluate_on_seeds,
            load_outcomes,
            paired_win_rate_delta,
            save_outcomes,
        )
        from pylatro_agent.vocab import build_vocab

        vocab = build_vocab(data)
        seeds = EVAL_SEEDS_V1[: args.games]
        logger.info(
            "Fixed-seed eval: %d seeds (EVAL_SEEDS_V1), win_ante=%s, stake=%d, temperature=%.3f",
            len(seeds),
            args.win_ante,
            args.stake,
            args.rollout_temperature,
        )
        start = time.time()

        if args.heuristic_baseline:
            outcomes = evaluate_heuristic_on_seeds(
                data,
                vocab,
                seeds,
                max_no_progress_steps=args.max_idle_steps,
                win_ante=args.win_ante,
                stake=args.stake,
            )
            label = "heuristic"
        else:
            import torch

            from pylatro_agent.agent import AgentConfig, BalatroAgent
            from pylatro_agent.checkpoint import load_checkpoint_payload

            logger.info("Loading checkpoint: %s", args.checkpoint)
            payload = load_checkpoint_payload(args.checkpoint, load_device)
            saved_agent_config = payload.get("agent_config") if isinstance(payload, dict) else None
            agent_config = AgentConfig(
                d_model=args.d_model,
                n_layers=args.n_layers,
                n_heads=args.n_heads,
                d_ff=args.d_ff,
            )
            if saved_agent_config:
                with contextlib.suppress(TypeError):
                    agent_config = AgentConfig(**saved_agent_config)
            device = torch.device(eval_device)
            model = BalatroAgent(agent_config, vocab).to(device)
            model_state = model.state_dict()
            state_dict = payload["state_dict"] if isinstance(payload, dict) else payload
            if any(k.startswith("module.") for k in state_dict) and not any(
                k.startswith("module.") for k in model_state
            ):
                state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
            try:
                model.load_state_dict(state_dict, strict=True)
            except RuntimeError as exc:
                raise RuntimeError(
                    f"Checkpoint {args.checkpoint} does not exactly match the v8 architecture. "
                    "Fresh v8 supervised training is required."
                ) from exc
            label = args.checkpoint
            outcomes = evaluate_on_seeds(
                model,
                data,
                vocab,
                seeds,
                device,
                max_no_progress_steps=args.max_idle_steps,
                win_ante=args.win_ante,
                temperature=args.rollout_temperature,
                stake=args.stake,
                batch_size=args.eval_batch_size,
                eval_workers=args.eval_workers,
                eval_cpu_threads=args.eval_cpu_threads,
            )

        elapsed = time.time() - start
        win_rate = sum(o.won_int for o in outcomes) / max(len(outcomes), 1)

        outcomes_path = args.outcomes_path or f"{args.checkpoint}.outcomes.json"
        if args.heuristic_baseline:
            outcomes_path = args.outcomes_path or "heuristic.outcomes.json"
        saved = save_outcomes(outcomes, outcomes_path)
        logger.info("Persisted %d per-seed outcomes to %s", len(outcomes), saved)

        print(
            f"\nlabel={label}\n"
            f"seeds={len(outcomes)}  win_rate={win_rate:.4f}  "
            f"wins={sum(o.won_int for o in outcomes)}/{len(outcomes)}  "
            f"elapsed={elapsed:.1f}s  device={eval_device}\n"
            f"win_ante={args.win_ante}  temperature={args.rollout_temperature}  stake={args.stake}\n"
            f"outcomes={outcomes_path}"
        )

        if args.paired_outcomes is not None:
            other = load_outcomes(args.paired_outcomes)
            delta = paired_win_rate_delta(outcomes, other)
            winner = "A(checkpoint)" if delta.delta >= 0 else "B(reference)"
            print(
                f"\n--- Paired comparison (this vs {args.paired_outcomes}) ---\n"
                f"win_rate this   = {delta.win_rate_a:.4f}\n"
                f"win_rate ref    = {delta.win_rate_b:.4f}\n"
                f"delta           = {delta.delta:+.4f}  (favoring {winner})\n"
                f"n_shared_seeds  = {delta.n_seeds}\n"
                f"discordant      = b(A-only-win)={delta.discordant_b}  c(B-only-win)={delta.discordant_c}\n"
                f"McNemar p-value = {delta.mcnemar_pvalue:.5f}\n"
                f"bootstrap mean  = {delta.bootstrap_mean:+.4f}\n"
                f"95% CI          = [{delta.bootstrap_ci_low:+.4f}, {delta.bootstrap_ci_high:+.4f}]\n"
                f"CI excludes 0   = {delta.excludes_zero_at(0.95)}"
            )
        return

    # === Unpaired path ===

    from pylatro_agent.agent import AgentConfig, BalatroAgent
    from pylatro_agent.checkpoint import load_checkpoint_payload
    from pylatro_agent.training.ppo import evaluate_model
    from pylatro_agent.vocab import build_vocab

    logger.info("Loading checkpoint: %s", args.checkpoint)
    payload = load_checkpoint_payload(args.checkpoint, load_device)
    saved_agent_config = payload.get("agent_config") if isinstance(payload, dict) else None

    agent_config = AgentConfig(
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
    )
    if saved_agent_config:
        try:
            agent_config = AgentConfig(**saved_agent_config)
            logger.info(
                "Using agent_config from checkpoint: d_model=%d n_layers=%d n_heads=%d d_ff=%d",
                agent_config.d_model,
                agent_config.n_layers,
                agent_config.n_heads,
                agent_config.d_ff,
            )
        except TypeError:
            logger.warning("Checkpoint agent_config was malformed; falling back to CLI architecture flags.")
    else:
        logger.info(
            "Checkpoint has no embedded agent_config; using CLI architecture "
            "(d_model=%d n_layers=%d n_heads=%d d_ff=%d).",
            agent_config.d_model,
            agent_config.n_layers,
            agent_config.n_heads,
            agent_config.d_ff,
        )

    if isinstance(payload, dict) and payload.get("update_count") is not None:
        logger.info(
            "Checkpoint metadata: update_count=%s total_steps=%s best_eval_win_rate=%s",
            payload.get("update_count"),
            payload.get("total_steps"),
            payload.get("best_eval_win_rate"),
        )

    data = load_game_data()
    vocab = build_vocab(data)

    import torch

    device = torch.device(eval_device)
    model = BalatroAgent(agent_config, vocab).to(device)
    model_state = model.state_dict()
    state_dict = payload["state_dict"] if isinstance(payload, dict) else payload
    # Strip a possible DataParallel prefix mismatch.
    if any(k.startswith("module.") for k in state_dict) and not any(k.startswith("module.") for k in model_state):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Checkpoint {args.checkpoint} does not exactly match the v8 architecture. "
            "Fresh v8 supervised training is required."
        ) from exc

    logger.info(
        "Running %d eval games on %s (win_ante=%s, temperature=%.3f, stake=%d)...",
        args.games,
        eval_device,
        args.win_ante,
        args.rollout_temperature,
        args.stake,
    )
    start = time.time()
    win_rate = evaluate_model(
        model,
        data,
        vocab,
        args.games,
        device,
        max_no_progress_steps=args.max_idle_steps,
        win_ante=args.win_ante,
        temperature=args.rollout_temperature,
        stake=args.stake,
        eval_batch_size=args.eval_batch_size,
        eval_workers=args.eval_workers,
        eval_cpu_threads=args.eval_cpu_threads,
    )
    elapsed = time.time() - start

    print(
        f"\ncheckpoint={args.checkpoint}\n"
        f"games={args.games}  win_rate={win_rate:.4f}  "
        f"wins={round(win_rate * args.games)}/{args.games}  "
        f"elapsed={elapsed:.1f}s  device={eval_device}\n"
        f"win_ante={args.win_ante}  temperature={args.rollout_temperature}  stake={args.stake}"
    )


if __name__ == "__main__":
    main()
