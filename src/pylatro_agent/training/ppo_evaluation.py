"""Fresh-start, seed-controlled batched PPO evaluation."""

from __future__ import annotations

import json
import logging
import math
import time
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from ..action import decode_action
from ..env import BalatroEnv
from ..reward import RewardConfig
from ..survival import terminal_outcome_class
from .ppo_observations import (
    _obs_dicts_to_batch,
)
from .ppo_policy import (
    _grammar_distribution,
    _policy_temperature_for_scalars,
)

if TYPE_CHECKING:
    from pylatro import GameData

    from ..agent import BalatroAgent
    from ..vocab import Vocab

logger = logging.getLogger(__name__)


def evaluation_seed_list(num_games: int, seeds: list[int] | None = None) -> list[int]:
    result = list(seeds[:num_games]) if seeds is not None else [10000 + i for i in range(num_games)]
    if num_games < 1 or len(result) != num_games or len(set(result)) != len(result):
        raise ValueError("Evaluation requires num_games distinct seeds and num_games > 0")
    return result


def validate_resume_evaluation_seeds(config, saved_config: dict) -> None:
    """Do not silently turn possibly trained seeds into a new validation panel."""
    if "eval_games" not in saved_config:
        raise ValueError("Resume lacks evaluation seed reservation metadata; use explicit actor transfer")
    saved = evaluation_seed_list(saved_config["eval_games"], saved_config.get("eval_seeds"))
    active = evaluation_seed_list(config.eval_games, config.eval_seeds)
    if active != saved:
        raise ValueError("Evaluation seeds changed on resume; keep the reserved seed panel")


def summarize_evaluation(outcomes: list[dict]) -> dict[str, float]:
    """Episode-weighted fresh metrics, including censored-episode coverage."""
    if not outcomes:
        return {}

    def mean(key):
        return sum(float(row[key]) for row in outcomes) / len(outcomes)

    metrics = {
        "win_rate": mean("won"),
        "final_ante_mean": mean("final_ante"),
        "max_ante_mean": mean("max_ante"),
        "stall_rate": mean("stalled"),
    }
    for ante in range(1, 9):
        metrics[f"reach_ante{ante}"] = sum(row["max_ante"] >= ante for row in outcomes) / len(outcomes)
        metrics[f"clear_ante{ante}"] = sum(ante in row.get("cleared_antes", ()) for row in outcomes) / len(outcomes)
    for name in ("outcome_nll", "outcome_brier", "win_brier"):
        values = [row["critic"][name] for row in outcomes if name in row.get("critic", {})]
        if values:
            metrics[f"critic/{name}"] = sum(values) / len(values)
    metrics["critic/labeled_episode_fraction"] = sum(bool(row.get("critic")) for row in outcomes) / len(outcomes)
    return metrics


def _score_critic_forecasts(forecasts: list[dict], *, won: bool, final_ante: int, censored: bool) -> dict:
    if not forecasts or censored:
        return {}
    target = terminal_outcome_class(won=won, final_ante=final_ante)
    nll, brier, win_brier = 0.0, 0.0, 0.0
    for forecast in forecasts:
        probs = forecast["outcome_probabilities"]
        nll -= math.log(max(probs[target], 1e-12))
        brier += sum((p - float(index == target)) ** 2 for index, p in enumerate(probs))
        win_brier += (probs[-1] - float(won)) ** 2
    return {
        "outcome_nll": nll / len(forecasts),
        "outcome_brier": brier / len(forecasts),
        "win_brier": win_brier / len(forecasts),
        "states": len(forecasts),
    }


def _next_eval_regression_streak(
    *,
    win_rate: float,
    best_win_rate: float | None,
    current_streak: int,
    tolerance: float | None,
) -> int:
    """Advance or clear the consecutive material-eval-regression count."""

    if tolerance is None or best_win_rate is None:
        return 0
    regression = best_win_rate - win_rate
    # Eval rates are ratios of integer game counts, so a nominal boundary such
    # as 0.87 - 0.77 can land just below 0.10 in binary floating point. Treat
    # numerically equal boundaries as material regressions.
    if regression > tolerance or math.isclose(regression, tolerance, rel_tol=1e-9, abs_tol=1e-12):
        return current_streak + 1
    return 0


def evaluate_model(
    model: BalatroAgent,
    data: GameData,
    vocab: Vocab,
    num_games: int,
    device: torch.device,
    max_no_progress_steps: int = 256,
    win_ante: int | None = None,
    temperature: float = 1.0,
    stake: int = 1,
    seeds: list[int] | None = None,
    eval_batch_size: int = 32,
    reward_config: RewardConfig | None = None,
    results_path: str | Path | None = None,
    writer=None,
    update_count: int = 0,
    greedy: bool = True,
    sampled_games: int = 0,
    policy_config=None,
    sampling_seed: int = 31013,
    summary_out: dict | None = None,
) -> float:
    """Evaluate model win rate with greedy action selection over num_games.

    When ``seeds`` is provided it is used verbatim (and truncated to
    ``num_games``); otherwise the historical ``10000 + game_idx`` seeds are
    generated. Passing the versioned :data:`pylatro_agent.eval.EVAL_SEEDS_V1`
    list makes every checkpoint's eval reproducible and pairable across runs.

    Games are played in lockstep batches of ``eval_batch_size`` so one forward
    pass serves many games (see :func:`run_seed_evaluation`). Per-seed results
    are unchanged: the engine is deterministic given its seed and greedy action
    selection does not depend on what the other games in the batch are doing.
    """
    seed_list = evaluation_seed_list(num_games, seeds)
    base_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    reserved = getattr(base_model, "_training_reserved_seeds", None)
    started = time.perf_counter()
    with ExitStack() as stack:
        if not greedy:
            cuda_devices = (
                [device.index if device.index is not None else torch.cuda.current_device()]
                if device.type == "cuda"
                else []
            )
            stack.enter_context(torch.random.fork_rng(devices=cuda_devices))
            torch.random.default_generator.manual_seed(sampling_seed)
            if device.type == "cuda":
                with torch.cuda.device(device):
                    torch.cuda.manual_seed(sampling_seed)
            elif device.type == "mps":
                stack.callback(torch.mps.set_rng_state, torch.mps.get_rng_state())
                torch.mps.manual_seed(sampling_seed)
        output_file = None
        if results_path is not None:
            path = Path(results_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            output_file = stack.enter_context(path.open("a", buffering=1))

        def record(outcome):
            if output_file is not None:
                output_file.write(
                    json.dumps(
                        {
                            "update": update_count,
                            "policy": "fresh_greedy" if greedy else "fresh_sampled",
                            "sampling_seed": None if greedy else sampling_seed,
                            "stake": stake,
                            "source_training_seed_disjoint": (
                                str(outcome["seed"]) in {str(seed) for seed in reserved}
                                if reserved is not None
                                else None
                            ),
                            **outcome,
                        },
                        allow_nan=False,
                    )
                    + "\n"
                )

        outcomes = run_seed_evaluation(
            model,
            data,
            vocab,
            seed_list,
            device,
            max_no_progress_steps=max_no_progress_steps,
            win_ante=win_ante,
            temperature=temperature,
            stake=stake,
            greedy=greedy,
            batch_size=eval_batch_size,
            reward_config=reward_config,
            record_critic=True,
            on_result=record,
            policy_config=policy_config,
        )
    metrics = summarize_evaluation(outcomes)
    if summary_out is not None:
        summary_out.update(metrics)
    if writer is not None:
        prefix = "eval/" if greedy else "eval/sampled/"
        for name, value in metrics.items():
            writer.add_scalar(prefix + name, value, update_count)
        writer.add_scalar(prefix + "wall_seconds", time.perf_counter() - started, update_count)
        writer.add_scalar(
            prefix + "source_training_disjoint_verified",
            float(reserved is not None and set(seed_list) <= set(reserved)),
            update_count,
        )
        for name in ("policy_seconds", "env_seconds"):
            writer.add_scalar(prefix + name, sum(row.get(name, 0) for row in outcomes), update_count)
    if greedy and sampled_games > 0:
        evaluate_model(
            model,
            data,
            vocab,
            min(sampled_games, len(seed_list)),
            device,
            max_no_progress_steps=max_no_progress_steps,
            win_ante=win_ante,
            temperature=temperature,
            stake=stake,
            seeds=seed_list,
            eval_batch_size=eval_batch_size,
            reward_config=reward_config,
            results_path=results_path,
            writer=writer,
            update_count=update_count,
            greedy=False,
            policy_config=policy_config,
            sampling_seed=sampling_seed,
        )
    wins = sum(1 for outcome in outcomes if outcome["won"])
    return wins / max(len(outcomes), 1)


def evaluation_rank(win_rate: float, summary: dict) -> tuple[float, ...]:
    """Win rate remains primary; fresh progression breaks sparse zero-win ties."""
    return (
        win_rate,
        sum(summary.get(f"clear_ante{ante}", 0) for ante in range(1, 9)),
        sum(summary.get(f"reach_ante{ante}", 0) for ante in range(1, 9)),
        -summary.get("stall_rate", 0),
    )


def run_seed_evaluation(
    model: BalatroAgent,
    data: GameData,
    vocab: Vocab,
    seeds: list[int],
    device: torch.device,
    *,
    max_no_progress_steps: int = 256,
    win_ante: int | None = None,
    temperature: float = 1.0,
    stake: int = 1,
    greedy: bool = True,
    batch_size: int = 32,
    reward_config: RewardConfig | None = None,
    record_critic: bool = False,
    on_result=None,
    policy_config=None,
) -> list[dict]:
    """Play every seed and return its outcome, batching the policy forward pass.

    The retired implementation played games strictly serially with batch-size-1
    forwards. Eval then cost more wall clock than the training it was measuring
    (measured on the Ante-5 runs: 60-95 minutes per 400-game eval against ~54
    minutes of training per 25-update interval). Here up to ``batch_size`` games
    advance in lockstep and share one forward pass, so the same GPU/CPU work
    serves many games at once. Env stepping stays in-process and serial, so the
    speedup tracks the forward-pass share of per-step cost.

    Determinism: each seed constructs its own env and greedy selection is a
    per-row argmax, so per-seed outcomes do not depend on batch composition or
    on which slot a seed lands in. Sampled evaluation (``greedy=False``) draws
    from the global torch RNG and is reproducible only under a fixed seed and a
    fixed ``batch_size``.

    Memory safety: runs under ``torch.inference_mode`` (no autograd graph) and
    drains the MPS cache periodically so an eval-heavy run cannot jetsam idle
    AsyncVectorEnv workers.
    """
    if not seeds:
        return []
    if len(set(seeds)) != len(seeds):
        raise ValueError("Evaluation seeds must be distinct")
    slot_count = max(1, min(int(batch_size), len(seeds)))

    results_by_seed: dict[int, dict] = {}
    pending = list(seeds)
    # Each live slot is (seed, env, obs). Slots advance in lockstep; a finished
    # slot immediately picks up the next pending seed so the batch stays full.
    slots: list[tuple[int, BalatroEnv, dict]] = []
    records: dict[int, dict] = {}

    def _start(seed: int) -> tuple[int, BalatroEnv, dict]:
        env = BalatroEnv(
            seed=seed,
            data=data,
            vocab=vocab,
            stake=stake,
            max_steps=max_no_progress_steps,
            win_ante=win_ante,
            # Eval never reads teacher labels; the heuristic teacher would
            # otherwise run twice per step of every eval game.
            enable_teacher=False,
            # No evaluation reward is optimized. Use explicit training reward
            # semantics when supplied; avoid expensive shaped-reward defaults.
            reward_config=reward_config
            or (
                RewardConfig(objective="milestone")
                if (win_ante or 8) == 8
                else RewardConfig(enable_score_build_potential=False)
            ),
        )
        env_stack.callback(env.close)
        obs, _ = env.reset(seed=seed)
        records[seed] = {
            "max_ante": 1,
            "steps": 0,
            "action_counts": Counter(),
            "critic_forecasts": [],
            "policy_seconds": 0.0,
            "env_seconds": 0.0,
        }
        return seed, env, obs

    with ExitStack() as env_stack, torch.inference_mode():
        env_stack.callback(model.train, model.training)
        model.eval()
        while pending and len(slots) < slot_count:
            slots.append(_start(pending.pop(0)))

        steps_since_drain = 0
        last_progress = time.monotonic()
        while slots:
            policy_started = time.perf_counter()
            batch = _obs_dicts_to_batch([obs for _seed, _env, obs in slots], device)
            policy_temperature = (
                _policy_temperature_for_scalars(batch["scalars"], policy_config)
                if policy_config is not None
                else temperature
            )
            dist, _value = _grammar_distribution(model, batch, temperature=policy_temperature)
            actions = (dist.mode() if greedy else dist.sample()).cpu().numpy()
            critic_probs = _value["outcome_probabilities"].cpu().tolist() if record_critic else None
            policy_seconds = (time.perf_counter() - policy_started) / len(slots)
            # Drop per-step inference tensors immediately; otherwise they sit on
            # the MPS allocator until the whole eval finishes.
            del batch, dist

            next_slots: list[tuple[int, BalatroEnv, dict]] = []
            for slot_index, (seed, env, _obs) in enumerate(slots):
                record = records[seed]
                record["policy_seconds"] += policy_seconds
                record["steps"] += 1
                record["action_counts"][decode_action(int(actions[slot_index])).action_type.value] += 1
                if critic_probs is not None:
                    record["critic_forecasts"].append(
                        {
                            "ante": int(_obs["scalars"][2]),
                            "phase": str(env._sub_phase),
                            "outcome_probabilities": critic_probs[slot_index],
                        }
                    )
                env_started = time.perf_counter()
                obs, _reward, terminated, truncated, info = env.step(int(actions[slot_index]))
                record["env_seconds"] += time.perf_counter() - env_started
                record["max_ante"] = max(record["max_ante"], int(info.get("ante", 1) or 1))
                if terminated or truncated:
                    results_by_seed[seed] = {
                        "seed": seed,
                        "won": bool(info.get("won", False)),
                        **record,
                        "final_ante": int(info.get("ante", 1) or 1),
                        "round_score": int(info.get("round_score", 0) or 0),
                        "stalled": bool(info.get("stalled", False)),
                        "censored": bool(truncated),
                        "cash": float(info.get("dollars", 0)),
                        "terminal_blind": str(info.get("blind_on_deck", "")),
                        "boss_key": str(info.get("boss_key", "")),
                        "joker_keys": list(env.state.joker_keys),
                        "cleared_antes": sorted(
                            env._cleared_boss_antes | ({env.state.win_ante} if info.get("won") else set())
                        ),
                        "critic": _score_critic_forecasts(
                            record["critic_forecasts"],
                            won=bool(info.get("won")),
                            final_ante=int(info.get("ante", 1)),
                            censored=bool(truncated),
                        ),
                    }
                    if on_result is not None:
                        on_result(results_by_seed[seed])
                    env.close()
                    if pending:
                        next_slots.append(_start(pending.pop(0)))
                else:
                    next_slots.append((seed, env, obs))
            slots = next_slots

            if time.monotonic() - last_progress >= 30:
                logger.info(
                    "Evaluation progress: %d/%d games complete; %d batched steps",
                    len(results_by_seed),
                    len(seeds),
                    steps_since_drain + 1,
                )
                last_progress = time.monotonic()

            steps_since_drain += 1
            if device.type == "mps" and steps_since_drain >= 200:
                torch.mps.empty_cache()
                steps_since_drain = 0

    # Preserve caller seed order regardless of completion order.
    return [results_by_seed[seed] for seed in seeds if seed in results_by_seed]
