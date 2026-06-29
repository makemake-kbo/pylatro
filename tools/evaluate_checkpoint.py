#!/usr/bin/env python3
"""Evaluate a PPO/supervised checkpoint's win rate over N games.

Loads a checkpoint (full PPO resume checkpoint or weights-only), rebuilds the
model from the embedded agent_config when available (falling back to CLI
``--d-model``/``--n-layers``/``--n-heads``/``--d-ff`` for legacy files), and
runs the same greedy ``evaluate_model`` loop used during PPO eval.

Example:

    uv run --extra agent python tools/evaluate_checkpoint.py \\
        --checkpoint checkpoints/ppo/ppo_update500.pt \\
        --games 500 --device mps --eval-device cpu \\
        --win-ante 4 --rollout-temperature 0.85
"""

from __future__ import annotations

import argparse
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
    args = parser.parse_args()

    load_device = _resolve_device(args.device)
    eval_device = _resolve_device(args.eval_device or args.device)

    from pylatro import load_game_data
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
    compatible = {
        k: v for k, v in state_dict.items() if k in model_state and model_state[k].shape == v.shape
    }
    model.load_state_dict(compatible, strict=False)
    skipped = sorted(set(state_dict) - set(compatible))
    if skipped:
        logger.warning("Skipped %d incompatible tensors during load (e.g. %s).", len(skipped), skipped[:3])

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
