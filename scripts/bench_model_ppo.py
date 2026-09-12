#!/usr/bin/env python3
"""Bounded, reproducible model/PPO microbenchmarks; never starts a training run.

Use the same command before/after a change. CUDA timings include explicit
synchronization, while the optional profiler separates host and kernel cost.
The fixture mixes reachable blind-selection, hand-play, and shop observations;
the PPO buffer is synthetic and its timing is not an end-to-end learning rate.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from pylatro import add_joker, load_game_data
from pylatro_agent.agent import AgentConfig, BalatroAgent
from pylatro_agent.constants import ActionRange
from pylatro_agent.env import BalatroEnv
from pylatro_agent.reward import RewardConfig
from pylatro_agent.training.ppo_config import PPOConfig
from pylatro_agent.training.ppo_observations import _obs_dicts_to_batch, _ObsBuffer
from pylatro_agent.training.ppo_optimization import _make_policy_optimizer, _run_ppo_update
from pylatro_agent.training.ppo_policy import _critic_predictions
from pylatro_agent.training.rollout_buffer import RolloutBuffer
from pylatro_agent.vocab import build_vocab


def fixture_observations(data) -> list[dict]:
    env = BalatroEnv(data=data, seed=1701, enable_teacher=False, reward_config=RewardConfig(objective="milestone"))
    blind, _ = env.reset()
    add_joker(env.state, "j_green_joker")
    add_joker(env.state, "j_runner")
    hand, *_ = env.step(int(ActionRange.BLIND_PLAY))
    env._controller.round_score = env._controller.blind_target() - 1
    action = next(
        int(a)
        for a in np.flatnonzero(env.action_masks())
        if ActionRange.PLAY_SUBSET_START <= a <= ActionRange.PLAY_SUBSET_END
    )
    shop, *_ = env.step(action)
    env.close()
    return [blind, hand, shop]


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def measure(fn, *, device: torch.device, warmup: int, repeats: int) -> dict:
    for _ in range(warmup):
        fn()
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    durations = []
    for _ in range(repeats):
        synchronize(device)
        started = time.perf_counter()
        fn()
        synchronize(device)
        durations.append((time.perf_counter() - started) * 1000)
    result = {"median_ms": statistics.median(durations), "min_ms": min(durations), "samples_ms": durations}
    if device.type == "cuda":
        result["peak_allocated_mb"] = torch.cuda.max_memory_allocated(device) / 2**20
    return result


def check_precision(model, batch, actions, rollout_batch_size: int) -> dict:
    """Compare fixed weights/actions, not two diverging optimization histories."""
    original_precision = model.config.precision
    results = {}
    reference_grads = None
    reference_log_probs = None
    reference_values = None
    try:
        for precision in ("fp32", "bf16"):
            model.config.precision = precision
            model.zero_grad(set_to_none=True)
            # Mimic the smaller no-grad rollout batches, then score those same
            # actions in a larger gradient-enabled PPO microbatch.
            with torch.no_grad():
                rollout_log_probs = []
                for start in range(0, len(actions), rollout_batch_size):
                    rows = {key: value[start : start + rollout_batch_size] for key, value in batch.items()}
                    dist, _ = model(**rows)
                    rollout_log_probs.append(dist.log_prob(actions[start : start + rollout_batch_size]))
                old_log_probs = torch.cat(rollout_log_probs)
            dist, values = model(**batch)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy()
            loss = -log_probs.mean() - 0.01 * entropy.mean() + values["expected_return"].square().mean()
            assert log_probs.dtype == entropy.dtype == torch.float32
            assert all(value.dtype == torch.float32 for value in values.values())
            loss.backward()
            gradients = [p.grad.detach() for p in model.parameters() if p.grad is not None]
            finite = all(bool(torch.isfinite(g).all()) for g in gradients)
            delta = log_probs.detach() - old_log_probs
            row = {
                "finite_loss_and_gradients": bool(torch.isfinite(loss)) and finite,
                "rollout_ppo_log_prob_max_abs": delta.abs().max().item(),
                "rollout_ppo_spurious_kl": ((delta.exp() - 1) - delta).mean().item(),
            }
            assert row["finite_loss_and_gradients"], row
            assert row["rollout_ppo_spurious_kl"] < 1e-4, row
            if precision == "fp32":
                reference_grads = [g.clone() for g in gradients]
                reference_log_probs = log_probs.detach().clone()
                reference_values = values["expected_return"].detach().clone()
            else:
                dot = sum((g.float() * ref.float()).sum() for g, ref in zip(gradients, reference_grads, strict=True))
                norm = sum(g.float().square().sum() for g in gradients).sqrt()
                ref_norm = sum(g.float().square().sum() for g in reference_grads).sqrt()
                ratio_error = ((log_probs.detach() - reference_log_probs).exp() - 1).abs()
                row.update(
                    gradient_cosine_vs_fp32=(dot / (norm * ref_norm).clamp_min(1e-20)).item(),
                    ratio_error_p99_vs_fp32=torch.quantile(ratio_error, 0.99).item(),
                    ratio_error_max_vs_fp32=ratio_error.max().item(),
                    expected_return_max_abs_vs_fp32=(values["expected_return"].detach() - reference_values)
                    .abs()
                    .max()
                    .item(),
                )
                assert row["gradient_cosine_vs_fp32"] > 0.99, row
                assert row["ratio_error_p99_vs_fp32"] < 0.02, row
            results[precision] = row
    finally:
        model.config.precision = original_precision
        model.zero_grad(set_to_none=True)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument(
        "--check-precision", action="store_true", help="Validate FP32/BF16 outputs and gradients on CUDA"
    )
    parser.add_argument(
        "--case",
        choices=["all", "rollout_policy", "model_backward", "obs_staging", "batches", "ppo",
                 "critic_replay", "critic_reference"],
        default="all",
    )
    parser.add_argument("--envs", type=int, default=16)
    parser.add_argument("--micro-batch", type=int, default=160)
    parser.add_argument("--logical-batch", type=int, default=320)
    parser.add_argument("--samples", type=int, default=320)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--n-layers", type=int, default=12)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--d-ff", type=int, default=1536)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    for key in ("envs", "micro_batch", "logical_batch", "samples", "ppo_epochs", "threads", "repeats"):
        if getattr(args, key) < 1:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be nonnegative")
    torch.set_num_threads(args.threads)
    torch.manual_seed(1701)
    np.random.seed(1701)
    device = torch.device(args.device)
    if (args.precision == "bf16" or args.check_precision) and device.type != "cuda":
        parser.error("BF16 benchmarking/validation requires CUDA; CPU fallback is not a BF16 benchmark")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    data = load_game_data()
    obs = fixture_observations(data)
    config = AgentConfig(
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        hand_ar_mixture_eps=0.1,
        win_only_value=True,
        precision=args.precision,
    )
    model = BalatroAgent(config, build_vocab(data)).to(device).eval()
    rollout_obs = [obs[i % len(obs)] for i in range(args.envs)]
    vector_obs = {key: np.stack([row[key] for row in rollout_obs]) for key in obs[0]}
    obs_buf = _ObsBuffer(args.envs, device)
    rollout_batch = _obs_dicts_to_batch(rollout_obs, device)
    micro_batch = _obs_dicts_to_batch([obs[i % len(obs)] for i in range(args.micro_batch)], device)
    with torch.no_grad():
        dist, _ = model(**micro_batch)
        fixed_actions = dist.sample()

    def rollout_policy():
        with torch.no_grad():
            dist, values = model(**rollout_batch)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)
            log_probs.exp()  # Same selected-action telemetry as collect_rollout.
            return values["expected_return"]

    def model_backward():
        model.zero_grad(set_to_none=True)
        dist, values = model(**micro_batch)
        loss = -dist.log_prob(fixed_actions).mean() - 0.01 * dist.entropy().mean()
        loss = loss + values["expected_return"].square().mean()
        loss.backward()

    buffer = RolloutBuffer(num_envs=1, rollout_length=args.samples)
    # Preparation is outside timed regions. Use valid actions and behavior
    # probabilities from the current model; returns are a synthetic signal.
    for start in range(0, args.samples, args.micro_batch):
        rows = [obs[i % len(obs)] for i in range(start, min(start + args.micro_batch, args.samples))]
        batch = _obs_dicts_to_batch(rows, device)
        with torch.no_grad():
            dist, values = model(**batch)
            actions = dist.sample()
            log_probs = dist.log_prob(actions).cpu().numpy()
            expected_returns = values["expected_return"].cpu().numpy()
            actions = actions.cpu().numpy()
        for i, row in enumerate(rows):
            buffer.add_batch(
                start + i,
                {key: value[None] for key, value in row.items()},
                actions[i : i + 1],
                np.asarray([0.01], dtype=np.float32),
                expected_returns[i : i + 1],
                log_probs[i : i + 1],
                np.asarray([False]),
                np.asarray([False]),
            )
    buffer.compute_returns_and_advantages(np.zeros(1))
    buffer.normalize_advantages()
    buffer.terminal_outcome_targets[:] = 5
    buffer.terminal_outcome_masks[::2] = 1
    optimizer = _make_policy_optimizer(model.parameters(), 3e-6)
    ppo_config = PPOConfig(
        ppo_epochs=args.ppo_epochs,
        mini_batch_size=args.logical_batch,
        micro_batch_size=args.micro_batch,
        target_kl=None,
        target_kl_max=None,
        precision=args.precision,
    )

    def ppo_update():
        return _run_ppo_update(
            model,
            optimizer,
            buffer,
            0.01,
            ppo_config,
            max(1, (args.logical_batch + args.micro_batch - 1) // args.micro_batch),
            args.micro_batch,
            device,
            device.type == "cuda",
        )

    def critic_step(optimized):
        model.zero_grad(set_to_none=True)
        values = _critic_predictions(model, micro_batch) if optimized else model(**micro_batch)[1]
        loss = -values["outcome_probabilities"][:, 0].clamp_min(1e-7).log().mean()
        params = [*model.value_head.outcome_proj.parameters(), *model.value_head.ante_survival.parameters()]
        return torch.autograd.grad(loss, params)

    cases = {
        "critic_reference": lambda: critic_step(False),
        "critic_replay": lambda: critic_step(True),
        "obs_staging": lambda: obs_buf.update(vector_obs),
        "rollout_policy": rollout_policy,
        "model_backward": model_backward,
        "batches": lambda: buffer.get_batches(
            args.logical_batch, device, pin_memory=device.type == "cuda", micro_batch_size=args.micro_batch
        ),
        "ppo": ppo_update,
    }
    report = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": str(device),
        "threads": torch.get_num_threads(),
        "model": asdict(config),
        "envs": args.envs,
        "micro_batch": args.micro_batch,
        "logical_batch": args.logical_batch,
        "samples": args.samples,
        "ppo_epochs": args.ppo_epochs,
        "precision": args.precision,
        "seed": 1701,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "results": {},
    }
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        report.update(
            gpu=props.name, compute_capability=[props.major, props.minor], gpu_memory_mb=props.total_memory / 2**20
        )
        report["native_bf16_supported"] = torch.cuda.is_bf16_supported(including_emulation=False)
    if args.check_precision:
        report["precision_checks"] = check_precision(model, micro_batch, fixed_actions, args.envs)
    print(json.dumps({key: value for key, value in report.items() if key != "results"}), flush=True)
    for name, fn in cases.items():
        if args.case not in ("all", name):
            continue
        result = measure(fn, device=device, warmup=args.warmup, repeats=args.repeats)
        report["results"][name] = result
        print(json.dumps({name: result}), flush=True)
        if args.profile:
            activities = [torch.profiler.ProfilerActivity.CPU]
            if device.type == "cuda":
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            with torch.profiler.profile(activities=activities) as profile:
                fn()
                synchronize(device)
            print(profile.key_averages().table(sort_by="self_cpu_time_total", row_limit=18), flush=True)
            counts = {
                event.key: event.count
                for event in profile.key_averages()
                if any(term in event.key for term in ("Synchronize", "aten::item", "_local_scalar", "attention"))
            }
            result["profile_counts"] = counts
    if args.case in ("all", "ppo"):
        report["finite_parameters_and_optimizer"] = all(
            bool(torch.isfinite(value).all())
            for value in [
                *model.parameters(),
                *[value for state in optimizer.state.values() for value in state.values() if torch.is_tensor(value)],
            ]
        )
        assert report["finite_parameters_and_optimizer"], "Nonfinite PPO parameters or optimizer state"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
