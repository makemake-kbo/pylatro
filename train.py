#!/usr/bin/env python3
"""Training entrypoint for the Balatro agent.

Usage:
    uv run python train.py supervised [--games 1000] [--epochs 5] [--device mps]
    uv run python train.py ppo [--pretrained PATH] [--steps 1000000] [--device mps]
    uv run python train.py self_play [--pretrained PATH] [--device mps]
    uv run python train.py pretrain_from_checkpoint \\
        --inference-checkpoint PATH [--min-ante 5] [--games 1000] [--data-path PATH]
"""

from __future__ import annotations

import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

from pylatro_agent._cython_check import check_cython_freshness  # noqa: E402

check_cython_freshness()


def main():
    parser = argparse.ArgumentParser(description="Train the Balatro agent")
    parser.add_argument(
        "phase",
        choices=["supervised", "ppo", "self_play", "pretrain_from_checkpoint"],
    )
    parser.add_argument("--games", type=int, default=1000, help="Heuristic games for supervised (default: 1000)")
    parser.add_argument("--epochs", type=int, default=5, help="Supervised epochs (default: 5)")
    parser.add_argument(
        "--supervised-entropy-coeff",
        type=float,
        default=0.001,
        help="Entropy bonus coefficient for supervised behavior cloning (default: 0.001)",
    )
    parser.add_argument(
        "--hand-ar-mixture-eps",
        type=float,
        default=None,
        help=(
            "Mixture weight on the autoregressive hand/discard head (Phase 1). 0.0 "
            "selects candidate-only support; the default depends on "
            "phase: 0.5 for supervised (trains the AR head on every label), 0.1 for PPO "
            "(targeted exploration). Pass explicitly to override."
        ),
    )
    parser.add_argument(
        "--outcome-weight-beta",
        type=float,
        default=1.5,
        help=(
            "Phase 5: AWR-style outcome weight beta for BC. w = exp(beta * normalized_outcome) "
            "clamped to [0.25, 4]. Tilts imitation toward successful trajectories while "
            "preserving full state coverage. 0.0 = uniform weights (no tilt). Default: 1.5."
        ),
    )
    parser.add_argument(
        "--min-ante",
        type=int,
        default=1,
        help=(
            "Minimum ante a heuristic game must reach to be kept for supervised pretraining. "
            "Phase 5: default changed from 5 to 1 (no filter) — hard outcome filtering creates "
            "survivorship bias. Outcome weighting (--outcome-weight-beta) replaces the filter."
        ),
    )
    parser.add_argument("--steps", type=int, default=1_000_000, help="PPO total timesteps (default: 1000000)")
    parser.add_argument("--envs", type=int, default=8, help="Parallel envs for PPO (default: 8)")
    parser.add_argument(
        "--rollout-length",
        type=int,
        default=256,
        help="PPO rollout length per env before each update (default: 256)",
    )
    parser.add_argument("--batch", type=int, default=128, help="Batch size (default: 128)")
    parser.add_argument("--ppo-epochs", type=int, default=4, help="PPO epochs per update (default: 4)")
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="Path to pretrained checkpoint (weights-only init; optimizer/counters start fresh)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help=(
            "Strictly resume a PPO run: restores model weights, optimizer (Adam moments), "
            "update counter, total_steps, entropy controller, RNG state, and return normalization. "
            "Requires a full PPO checkpoint (checkpoint_format == 'ppo_full'). Use --pretrained "
            "for weights-only initialization instead."
        ),
    )
    parser.add_argument(
        "--additional-updates",
        type=int,
        default=None,
        help=(
            "With --resume: run this many more PPO updates after the checkpoint's saved update "
            "count. The target update count becomes saved_update_count + N. Requires --resume."
        ),
    )
    parser.add_argument(
        "--updates",
        type=int,
        default=None,
        help=(
            "Run exactly N PPO updates from scratch, converting to N * envs * rollout_length "
            "timesteps internally. Overrides --steps. Combine with --pretrained for weights-only "
            "fine-tuning of a fixed length, or with --resume to set the total target instead of --additional-updates."
        ),
    )
    parser.add_argument(
        "--reset-schedules",
        action="store_true",
        help=(
            "With --resume: re-anchor fraction-of-training anneal schedules to this leg's "
            "recomputed horizon instead of "
            "the horizon saved in the checkpoint. Already-decayed coefficients will climb back "
            "toward their start values. Default: keep the original horizon so schedules never rewind."
        ),
    )
    parser.add_argument("--device", type=str, default=None, help="Device: cpu, mps, cuda (default: auto-detect)")
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help=(
            "Training seed for model initialization, policy sampling, SIL replay, "
            "and environment seed streams (default: 0)."
        ),
    )
    parser.add_argument("--lr", type=float, default=5e-5, help="PPO learning rate (default: 5e-5)")
    parser.add_argument(
        "--clip-eps",
        type=float,
        default=0.1,
        help=(
            "PPO clipped-surrogate range (default: 0.1, tighter than the usual 0.2). "
            "Raise toward 0.2 to let updates move the policy more when approx_kl sits "
            "well below --target-kl."
        ),
    )
    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=0.97,
        help="GAE lambda for advantage estimation (default: 0.97).",
    )
    parser.add_argument(
        "--value-loss-coeff",
        type=float,
        default=0.25,
        help="Weight on the critic (value) loss (default: 0.25).",
    )
    parser.add_argument("--d-model", type=int, default=384, help="Model dimension (default: 384)")
    parser.add_argument("--n-layers", type=int, default=12, help="Transformer layers (default: 12)")
    parser.add_argument("--n-heads", type=int, default=8, help="Transformer attention heads (default: 8)")
    parser.add_argument("--d-ff", type=int, default=1536, help="Transformer feed-forward dimension (default: 1536)")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Base dir for checkpoints (default: checkpoints/<phase>)",
    )
    parser.add_argument("--workers", type=int, default=0, help="CPU workers for game generation (default: all cores)")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10000,
        help="Generate games in chunks of this size, writing each to disk to limit memory (default: 10000)",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Base dir for TensorBoard logs (default: runs/<phase>)",
    )
    parser.add_argument(
        "--sync-envs",
        action="store_true",
        help=(
            "Use SyncVectorEnv instead of AsyncVectorEnv for PPO rollouts. "
            "Async is faster but worker exceptions surface as EOFError/"
            "BrokenPipe in the parent with no traceback. Pass --sync-envs "
            "to reproduce env-worker crashes synchronously in the parent "
            "process for debugging. Default: async."
        ),
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="PPO console log interval in updates (default: 10)",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=50,
        help="PPO checkpoint interval in updates (default: 50)",
    )
    parser.add_argument("--eval-interval", type=int, default=5, help="PPO eval interval in updates (default: 5)")
    parser.add_argument(
        "--max-idle-steps",
        type=int,
        default=32,
        help="Terminate PPO episodes after this many consecutive no-progress steps (default: 32)",
    )
    parser.add_argument(
        "--target-entropy",
        type=float,
        default=0.15,
        help="PPO target normalized entropy ratio in [0, 1] when adaptive entropy is enabled (default: 0.15)",
    )
    parser.add_argument(
        "--win-ante",
        type=int,
        default=None,
        help=(
            "Curriculum: cap the run's victory ante below the engine default of 8. "
            "Heuristic-teacher win rates: ~39%% at ante 4, ~12%% at 5, ~2%% at 6, "
            "~1%% at 8. Lower values give PPO frequent positive-reward terminals "
            "to learn from; raise as the policy improves. Default: engine default (8)."
        ),
    )
    parser.add_argument(
        "--eval-games",
        type=int,
        default=50,
        help="Games per PPO eval pass (default: 50). Wins are rare; small samples are noisy.",
    )
    parser.add_argument(
        "--eval-regression-tolerance",
        type=float,
        default=None,
        help=(
            "Stop cleanly after repeated eval drops of at least this absolute win-rate "
            "amount from the best checkpoint (for example 0.10). Default: disabled."
        ),
    )
    parser.add_argument(
        "--eval-regression-patience",
        type=int,
        default=2,
        help="Consecutive material eval regressions before stopping (default: 2).",
    )
    parser.add_argument(
        "--eval-device",
        type=str,
        default=None,
        help=(
            "Optional device override for the PPO eval pass (cpu, mps, cuda). "
            "Useful when training on MPS: eval runs serially in-process and "
            "its forward-pass allocations compete with idle AsyncVectorEnv "
            "workers for unified memory. Setting --eval-device cpu moves "
            "those allocations off the GPU entirely at the cost of slower "
            "eval. Default: same as --device."
        ),
    )
    parser.add_argument(
        "--rollout-temperature",
        type=float,
        default=1.0,
        help=(
            "Softmax temperature applied at all PPO distribution sites (rollout sampling, "
            "training forward pass, value bootstraps). Set to 1.0 to disable sharpening "
            "(default: 1.0). The 0.7 crutch is obsolete post-Phase-1 BC."
        ),
    )
    parser.add_argument(
        "--danger-rollout-temperature",
        type=float,
        default=None,
        help=(
            "Optional lower temperature for active hand decisions in Ante 1 or "
            "when immediate death probability crosses --danger-death-probability-threshold. "
            "Shop, pack, and blind-select exploration keep --rollout-temperature. "
            "Use 0.85; default: disabled."
        ),
    )
    parser.add_argument(
        "--danger-death-probability-threshold",
        type=float,
        default=0.35,
        help="Danger threshold for state-dependent hand-decision sharpening (default: 0.35).",
    )
    parser.add_argument(
        "--danger-shop-leave-logit-penalty",
        type=float,
        default=0.0,
        help=(
            "Soft macro-logit penalty on leaving an unsafe shop, paired with half as much "
            "positive reroll bias. It never masks actions. Use 2.0; default: disabled."
        ),
    )
    parser.add_argument(
        "--entropy-coeff",
        type=float,
        default=0.01,
        help="PPO entropy coefficient (default: 0.01)",
    )
    parser.add_argument(
        "--target-kl",
        type=float,
        default=0.05,
        help=(
            "Stop remaining PPO epochs when full-rollout approximate KL exceeds "
            "this value; <=0 disables (default: 0.05)"
        ),
    )
    parser.add_argument(
        "--target-kl-p95",
        type=float,
        default=None,
        help=(
            "Optional hard guard: warn (and log ppo/stop_reason_kl_p95) when the 95th "
            "percentile minibatch approx_kl exceeds this. Does not stop training by "
            "itself; pair with --target-kl for early stopping. Default: disabled."
        ),
    )
    parser.add_argument(
        "--target-kl-max",
        type=float,
        default=None,
        help=(
            "Optional hard safety limit: reject the complete PPO update and restore "
            "model plus optimizer when any minibatch approximate KL exceeds this. "
            "Default: disabled."
        ),
    )
    parser.add_argument(
        "--min-minibatch-fraction",
        type=float,
        default=None,
        help=(
            "Minimum minibatch fraction that must be processed before target-KL may "
            "soft-stop remaining epochs. Use ~0.50. Default: disabled."
        ),
    )
    parser.add_argument(
        "--entropy-ema-beta",
        type=float,
        default=0.6,
        help="EMA smoothing for PPO entropy control signal (default: 0.6)",
    )
    parser.add_argument(
        "--alpha-lr",
        type=float,
        default=1e-2,
        help="Learning rate for adaptive entropy coefficient updates (default: 1e-2)",
    )
    parser.add_argument(
        "--action-type-entropy-scale",
        type=float,
        default=0.0,
        help="Extra PPO bonus scale for normalized entropy over action types (default: 0.0)",
    )
    parser.add_argument(
        "--adaptive-entropy",
        action="store_true",
        help="Enable adaptive entropy tuning. Disabled by default to preserve pretrained policies.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.997,
        help="PPO discount factor for returns and reward potential shaping (default: 0.997)",
    )
    parser.add_argument(
        "--dense-reward-scale",
        type=float,
        default=1.0,
        help=(
            "Multiplier for dense shaping rewards. Terminal win/loss reward is unchanged. "
            "Set < 1.0 (e.g. 0.25) to shrink shaping so the policy optimizes winning rather "
            "than farming bounded shaping. Default: 1.0 (no change to existing behavior)."
        ),
    )
    parser.add_argument(
        "--consumable-reward-scale",
        type=float,
        default=1.0,
        help="Additional scale for optional planet-alignment shaping. Default: 1.0.",
    )
    parser.add_argument(
        "--strategic-event-reward-scale",
        type=float,
        default=1.0,
        help=(
            "Independent scale for attributable Tarot, seal, Gold, and cash events. "
            "Positive strategic credit remains capped at 1.0 per transition. Default: 1.0."
        ),
    )
    parser.add_argument(
        "--planet-unmatched-use-penalty-coeff",
        type=float,
        default=0.0,
        help=(
            "Per-step penalty coefficient for using a planet whose hand is not a "
            "draw-reliable scoring plan. The penalty is weighted by the plan's "
            "reliability deficit and ante progress. Default: 0.0 (disabled)."
        ),
    )
    parser.add_argument(
        "--planet-unmatched-claim-penalty-coeff",
        type=float,
        default=0.0,
        help=(
            "Per-step penalty coefficient for claiming a planet whose hand is not a "
            "draw-reliable scoring plan. Best-available pack choices remain exempt. "
            "Default: 0.0 (disabled)."
        ),
    )
    parser.add_argument(
        "--planet-match-shaping",
        action="store_true",
        help=(
            "Enable plan-aware Planet shaping. Pair/High Card are scalable "
            "fallbacks; Flush and kind hands require matching deck concentration. "
            "Two Pair is low-priority and requires a dedicated synergy Joker; "
            "Full House is not a strategic Planet target."
        ),
    )
    parser.add_argument(
        "--score-build-potential",
        dest="score_build_potential",
        action="store_true",
        help=(
            "Enable contextual score/build potential shaping. "
            "It values draw-reliable hand plans, blind readiness, Tarot/economy "
            "conversion, Blue/Purple seals, score-improving shop options, Standard-pack "
            "seal searches, and capped recognized-scaler value. Default: disabled."
        ),
    )
    parser.add_argument(
        "--critic-warmup-updates",
        type=int,
        default=0,
        help=(
            "Phase 3.2: number of PPO updates to train only the critic (+survival head) "
            "with the policy frozen. After a reward-function change, advantages are "
            "garbage until the critic tracks the new return distribution; warming it up "
            "on-policy removes the window in which PPO earnestly optimizes noise. "
            "Pair with --reinit-value-head. Default: 0 (disabled)."
        ),
    )
    parser.add_argument(
        "--critic-warmup-lr",
        type=float,
        default=None,
        help=(
            "Learning rate used only while the policy is frozen and critic gradients are restricted "
            "to the value head. The optimizer switches back to --lr before the first actor update. "
            "Default: --lr."
        ),
    )
    parser.add_argument(
        "--critic-warmup-min-ev",
        type=float,
        default=0.7,
        help=(
            "Explained-variance threshold for unfreezing the policy after the minimum "
            "critic warmup count. The complete finite window's rolling mean must "
            "clear it. Pass 0 for a purely count-based warmup. Default: 0.7."
        ),
    )
    parser.add_argument(
        "--critic-warmup-ev-window",
        type=int,
        default=3,
        help=(
            "Consecutive finite rollout-EV samples whose rolling mean must clear "
            "--critic-warmup-min-ev before actor unfreeze. Must be >=2 for an "
            "EV-gated warmup. Default: 3."
        ),
    )
    parser.add_argument(
        "--critic-warmup-max-updates",
        type=int,
        default=None,
        help=(
            "Fail-closed warmup limit. If sustained EV is still unready, save "
            "ppo_warmup_unready.pt and stop with the actor frozen. Default: 4x "
            "--critic-warmup-updates."
        ),
    )
    parser.add_argument(
        "--actor-ramp-updates",
        type=int,
        default=0,
        help=(
            "Protected actor updates after warmup. PPO clip epsilon ramps to its "
            "configured value while critic gradients stay out of the shared policy "
            "trunk. Default: 0 (historical immediate transition)."
        ),
    )
    parser.add_argument(
        "--actor-ramp-start-clip-fraction",
        type=float,
        default=0.5,
        help=("Starting actor-ramp clip epsilon as a fraction of --clip-eps. Default: 0.5."),
    )
    parser.add_argument(
        "--ppo-run-uuid",
        default=None,
        help="Provenance UUID generated once by the production launcher and persisted on strict resume.",
    )
    parser.add_argument(
        "--ppo-source-sha256",
        default=None,
        help="SHA256 of the pinned weights-only source checkpoint for run provenance.",
    )
    parser.add_argument(
        "--ppo-recipe-id",
        default=None,
        help="Stable production recipe identity persisted in every PPO checkpoint.",
    )
    parser.add_argument(
        "--reinit-value-head",
        action="store_true",
        help=(
            "Phase 3.2/2.4: reinitialize the value head when loading --pretrained. "
            "Required after any reward-function change so the critic doesn't start from "
            "a stale return mapping."
        ),
    )
    parser.add_argument(
        "--hl-gauss",
        action="store_true",
        help=(
            "Use the HL-Gauss categorical value head (51 return atoms over "
            "[-8, 12], cross-entropy loss) instead of the scalar MSE head. "
            "Bounded critic gradients on near-terminal coin-flip states and a "
            "distributional representation of bimodal win/loss returns. The "
            "head shape differs from scalar checkpoints, so initialize from one "
            "with --pretrained and --reinit-value-head rather than --resume. Pair "
            "it with --critic-warmup-updates 15 --critic-warmup-min-ev 0."
        ),
    )
    parser.add_argument(
        "--advantage-clip-sigma",
        type=float,
        default=4.0,
        help=(
            "Clamp globally-normalized advantages to this many standard "
            "deviations (0 disables). Caps the heavy near-terminal advantage "
            "tail that otherwise dominates the policy gradient. Default: 4.0."
        ),
    )
    parser.add_argument(
        "--reset-best-eval",
        action="store_true",
        help=(
            "With --resume: discard the checkpoint's best_eval_win_rate so "
            "ppo_best_eval.pt selection restarts from scratch. Use when the eval "
            "task changes (e.g. a --win-ante bump), otherwise no best-eval "
            "checkpoint is written until the harder task beats the old record."
        ),
    )
    parser.add_argument(
        "--sil-coeff",
        type=float,
        default=0.0,
        help=(
            "Self-imitation auxiliary actor coefficient (experimental, ~0.01). "
            "A small, bounded, critic-gated SIL loss on the agent's own "
            "completed episodes (wins and ordinary losses), trained alongside "
            "PPO in the same backward/optimizer step. Decays to --sil-coeff-final "
            "over --sil-decay-fraction of training. 0 disables (the default) and "
            "preserves the exact no-SIL PPO path. The buffer is in-memory only "
            "and refills after a resume."
        ),
    )
    parser.add_argument(
        "--sil-buffer-episodes",
        type=int,
        default=256,
        help=(
            "Max completed episodes kept in the SIL replay buffer (FIFO). Wins "
            "and ordinary losses are both stored now, so the default is larger "
            "than the historical win-only 64 (default: 256)."
        ),
    )
    parser.add_argument(
        "--sil-batch-size",
        type=int,
        default=64,
        help="SIL transitions sampled per attempted logical optimizer group.",
    )
    parser.add_argument(
        "--sil-min-episodes",
        type=int,
        default=8,
        help="Skip the SIL pass until the replay buffer holds this many episodes.",
    )
    parser.add_argument(
        "--sil-objective",
        choices=("advantage", "winning_bc"),
        default="advantage",
        help=(
            "SIL objective. 'advantage' samples all valid completed episodes "
            "(wins and losses), computes a current-critic MC advantage, and "
            "applies the percentile gate. 'winning_bc' samples only winning "
            "episodes with a unit gate (plain behavior cloning of wins), a "
            "matched control. --sil-coeff 0 remains the exact no-SIL switch "
            "(default: advantage)."
        ),
    )
    parser.add_argument(
        "--sil-advantage-floor",
        type=float,
        default=0.25,
        help=(
            "Absolute advantage floor (raw reward units). Sub-floor positive "
            "advantages receive zero gate weight, and the percentile gate only "
            "opens when the open percentile exceeds this floor (default: 0.25)."
        ),
    )
    parser.add_argument(
        "--sil-gate-open-percentile",
        type=float,
        default=80.0,
        help="Percentile of eligible advantages at which the gate opens (default: 80).",
    )
    parser.add_argument(
        "--sil-gate-saturation-percentile",
        type=float,
        default=95.0,
        help="Percentile of eligible advantages at which the gate saturates (default: 95).",
    )
    parser.add_argument(
        "--sil-samples-per-episode",
        type=int,
        default=8,
        help=("Maximum transitions one episode may contribute to a SIL / calibration batch (default: 8)."),
    )
    parser.add_argument(
        "--sil-logical-minibatches-per-update",
        type=int,
        default=1,
        choices=(1, 2),
        help=(
            "Logical PPO optimizer minibatches per update that may attempt SIL "
            "(one accumulated optimizer step == one logical group). The budget "
            "is per update, not per PPO epoch (default: 1)."
        ),
    )
    parser.add_argument(
        "--sil-coeff-final",
        type=float,
        default=0.0,
        help="SIL coefficient decays linearly to this value (default: 0.0).",
    )
    parser.add_argument(
        "--sil-decay-fraction",
        type=float,
        default=1.0,
        help=(
            "Fraction of the schedule horizon over which the SIL coefficient "
            "decays to --sil-coeff-final (default: 1.0, the full run)."
        ),
    )
    parser.add_argument(
        "--sil-grad-diagnostics-interval",
        type=int,
        default=10,
        help=(
            "Interval (updates) at which the weighted SIL / PPO actor-gradient "
            "ratio and cosine are logged. Diagnostic only (default: 10)."
        ),
    )
    parser.add_argument(
        "--counterfactual-diagnostic-interval",
        type=int,
        default=0,
        help=(
            "Replay one copied pre-play state with a focal joker removed every Nth "
            "actual play to validate analytic joker marginals. 0 disables (default). "
            "The copied-state replay has measurable many-env throughput cost."
        ),
    )
    parser.add_argument(
        "--inference-checkpoint",
        type=str,
        default=None,
        help="Checkpoint used to generate games for pretrain_from_checkpoint",
    )
    parser.add_argument(
        "--inf-d-model",
        type=int,
        default=None,
        help="Model dim for the inference checkpoint (defaults to --d-model)",
    )
    parser.add_argument(
        "--inf-n-layers",
        type=int,
        default=None,
        help="Transformer layers for the inference checkpoint (defaults to --n-layers)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Load records from this path if present; otherwise save generated records here",
    )
    parser.add_argument(
        "--sample-temperature",
        type=float,
        default=1.0,
        help="Softmax temperature for sampled actions during generation (default: 1.0)",
    )
    parser.add_argument(
        "--max-no-progress-steps-gen",
        type=int,
        default=2000,
        help="Per-env stall limit during data generation (default: 2000)",
    )
    args = parser.parse_args()

    device = args.device
    if device is None:
        import torch

        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    logging.info(f"Using device: {device}")

    from pylatro_agent.agent import AgentConfig

    agent_config = AgentConfig(
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        # 0 uses the scalar head; 51 atoms over [-8, 12] uses HL-Gauss.
        # In supervised mode a categorical head still trains (MSE through the
        # histogram mean); the HL-Gauss cross-entropy loss is PPO-only.
        value_bins=51 if args.hl_gauss else 0,
        danger_shop_leave_logit_penalty=args.danger_shop_leave_logit_penalty,
    )

    checkpoint_dir = args.checkpoint_dir
    log_dir = args.log_dir

    if args.phase == "supervised":
        from pylatro_agent.training.supervised import SupervisedConfig, train_supervised

        sup_eps = args.hand_ar_mixture_eps if args.hand_ar_mixture_eps is not None else 0.5
        train_supervised(
            SupervisedConfig(
                num_games=args.games,
                batch_size=args.batch,
                max_epochs=args.epochs,
                action_entropy_coeff=args.supervised_entropy_coeff,
                num_workers=args.workers,
                min_ante=args.min_ante,
                chunk_size=args.chunk_size,
                log_interval=args.log_interval,
                device=device,
                save_dir=checkpoint_dir or "checkpoints/supervised",
                log_dir=log_dir or "runs/supervised",
                hand_ar_mixture_eps=sup_eps,
                outcome_weight_beta=args.outcome_weight_beta,
            ),
            agent_config=agent_config,
        )

    elif args.phase == "ppo":
        from pylatro_agent.reward import RewardConfig
        from pylatro_agent.training.ppo import PPOConfig, train_ppo

        if args.resume and args.pretrained:
            parser.error("Pass either --pretrained or --resume, not both.")
        if args.additional_updates is not None and not args.resume:
            parser.error("--additional-updates requires --resume PATH.")
        if args.reset_schedules and not args.resume:
            parser.error("--reset-schedules requires --resume PATH.")
        if args.resume and args.updates is not None and args.additional_updates is not None:
            parser.error(
                "--updates and --additional-updates are mutually exclusive with --resume "
                "(--updates is an absolute target, --additional-updates is relative)."
            )
        # --updates N overrides --steps and is converted internally to
        # N * envs * rollout_length timesteps via PPOConfig.total_updates.
        ppo_eps = args.hand_ar_mixture_eps if args.hand_ar_mixture_eps is not None else 0.1
        train_ppo(
            PPOConfig(
                seed=args.seed,
                num_envs=args.envs,
                rollout_length=args.rollout_length,
                total_timesteps=args.steps,
                total_updates=args.updates,
                ppo_epochs=args.ppo_epochs,
                mini_batch_size=args.batch,
                lr=args.lr,
                clip_epsilon=args.clip_eps,
                gae_lambda=args.gae_lambda,
                value_loss_coeff=args.value_loss_coeff,
                device=device,
                save_dir=checkpoint_dir or "checkpoints/ppo",
                log_dir=log_dir or "runs/ppo",
                log_interval=args.log_interval,
                checkpoint_interval=args.checkpoint_interval,
                eval_interval=args.eval_interval,
                max_no_progress_steps=args.max_idle_steps,
                entropy_coeff=args.entropy_coeff,
                target_kl=None if args.target_kl <= 0.0 else args.target_kl,
                target_kl_p95=args.target_kl_p95,
                target_kl_max=args.target_kl_max,
                min_minibatch_fraction=args.min_minibatch_fraction,
                adaptive_entropy=args.adaptive_entropy,
                target_entropy=args.target_entropy,
                entropy_ema_beta=args.entropy_ema_beta,
                alpha_lr=args.alpha_lr,
                action_type_entropy_scale=args.action_type_entropy_scale,
                gamma=args.gamma,
                async_envs=not args.sync_envs,
                reset_schedules=args.reset_schedules,
                rollout_temperature=args.rollout_temperature,
                danger_rollout_temperature=args.danger_rollout_temperature,
                danger_death_probability_threshold=args.danger_death_probability_threshold,
                win_ante=args.win_ante,
                eval_games=args.eval_games,
                eval_regression_tolerance=args.eval_regression_tolerance,
                eval_regression_patience=args.eval_regression_patience,
                eval_device=args.eval_device,
                hand_ar_mixture_eps=ppo_eps,
                critic_warmup_updates=args.critic_warmup_updates,
                critic_warmup_lr=args.critic_warmup_lr,
                critic_warmup_min_ev=args.critic_warmup_min_ev,
                critic_warmup_ev_window=args.critic_warmup_ev_window,
                critic_warmup_max_updates=args.critic_warmup_max_updates,
                actor_ramp_updates=args.actor_ramp_updates,
                actor_ramp_start_clip_fraction=args.actor_ramp_start_clip_fraction,
                ppo_run_uuid=args.ppo_run_uuid,
                ppo_source_sha256=args.ppo_source_sha256,
                ppo_recipe_id=args.ppo_recipe_id,
                reinit_value_head=args.reinit_value_head,
                reset_best_eval=args.reset_best_eval,
                sil_coeff=args.sil_coeff,
                sil_buffer_episodes=args.sil_buffer_episodes,
                sil_batch_size=args.sil_batch_size,
                sil_min_episodes=args.sil_min_episodes,
                sil_objective=args.sil_objective,
                sil_advantage_floor=args.sil_advantage_floor,
                sil_gate_open_percentile=args.sil_gate_open_percentile,
                sil_gate_saturation_percentile=args.sil_gate_saturation_percentile,
                sil_samples_per_episode=args.sil_samples_per_episode,
                sil_logical_minibatches_per_update=args.sil_logical_minibatches_per_update,
                sil_coeff_final=args.sil_coeff_final,
                sil_decay_fraction=args.sil_decay_fraction,
                sil_grad_diagnostics_interval=args.sil_grad_diagnostics_interval,
                advantage_clip_sigma=args.advantage_clip_sigma,
                counterfactual_diagnostic_interval=args.counterfactual_diagnostic_interval,
                reward_config=RewardConfig(
                    gamma=args.gamma,
                    potential_win_ante=args.win_ante or 8,
                    enable_planet_match_rewards=args.planet_match_shaping,
                    planet_unmatched_use_penalty_coeff=args.planet_unmatched_use_penalty_coeff,
                    planet_unmatched_claim_penalty_coeff=args.planet_unmatched_claim_penalty_coeff,
                    enable_score_build_potential=args.score_build_potential,
                    dense_reward_scale=args.dense_reward_scale,
                    consumable_reward_scale=args.consumable_reward_scale,
                    strategic_event_reward_scale=args.strategic_event_reward_scale,
                ),
            ),
            agent_config=agent_config,
            pretrained_path=args.pretrained,
            resume_path=args.resume,
            additional_updates=args.additional_updates,
        )

    elif args.phase == "self_play":
        from pylatro_agent.training.self_play import SelfPlayConfig, train_self_play

        train_self_play(
            SelfPlayConfig(
                ppo_timesteps_per_stage=args.steps,
                device=device,
                save_dir=checkpoint_dir or "checkpoints/self_play",
            ),
            agent_config=agent_config,
            pretrained_path=args.pretrained,
        )

    elif args.phase == "pretrain_from_checkpoint":
        if not args.inference_checkpoint:
            parser.error("pretrain_from_checkpoint requires --inference-checkpoint PATH")

        from pathlib import Path

        from pylatro import load_game_data
        from pylatro_agent.training.model_generate import (
            ModelGenerateConfig,
            generate_training_data_from_model,
            load_records,
            save_records,
        )
        from pylatro_agent.training.supervised import SupervisedConfig, train_supervised
        from pylatro_agent.vocab import build_vocab

        data_path = Path(args.data_path) if args.data_path else None
        if data_path is not None and data_path.exists():
            records = load_records(data_path)
        else:
            game_data = load_game_data()
            build_vocab(game_data)  # validate vocab builds
            inf_config = AgentConfig(
                d_model=args.inf_d_model if args.inf_d_model is not None else args.d_model,
                n_layers=args.inf_n_layers if args.inf_n_layers is not None else args.n_layers,
            )
            gen_log_dir = log_dir or "runs/pretrain_from_checkpoint"
            records = generate_training_data_from_model(
                ModelGenerateConfig(
                    checkpoint_path=args.inference_checkpoint,
                    num_games=args.games,
                    num_envs=args.envs,
                    min_ante=args.min_ante,
                    device=device,
                    sample_temperature=args.sample_temperature,
                    max_no_progress_steps=args.max_no_progress_steps_gen,
                    async_envs=not args.sync_envs,
                    log_dir=gen_log_dir,
                ),
                agent_config=inf_config,
                data=game_data,
            )
            if data_path is not None:
                save_records(records, data_path)

        train_supervised(
            SupervisedConfig(
                num_games=args.games,
                batch_size=args.batch,
                max_epochs=args.epochs,
                num_workers=args.workers,
                min_ante=args.min_ante,
                device=device,
                save_dir=checkpoint_dir or "checkpoints/pretrain_from_checkpoint",
                log_dir=log_dir or "runs/pretrain_from_checkpoint",
            ),
            agent_config=agent_config,
            records=records,
        )


if __name__ == "__main__":
    main()
