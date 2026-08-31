"""Fixed-seed evaluation harness with paired comparison.

Balatro seed variance is enormous: a policy's per-seed outcome is dominated by
deck order, boss blind, and shop rolls rather than by policy quality. Unpaired
eval (different seeds per policy) has +/-0.05-0.07 noise at p~=0.4 win rate, which
swamps the real signal. Evaluating two policies on the *same* fixed seed list
reduces the variance of the *difference* by an order of magnitude because both
policies face identical deck orders, boss blinds, and shop rolls.

This module provides:

* ``EVAL_SEEDS_V1``, a versioned, frozen list of 400 seeds, identical across
  every checkpoint and the heuristic baseline. Bump the version suffix only if
  the seed semantics change; never edit in place.
* :func:`evaluate_on_seeds`, greedy per-seed eval that captures per-seed
  outcomes (won, max_ante, final round score) for persistence and pairing.
* :func:`paired_win_rate_delta`, win-rate delta with a McNemar test and a
  paired-bootstrap confidence interval on the difference.

The per-seed outcomes are persisted to a JSON/CSV next to the checkpoint so a
later paired comparison only needs the two outcome files, not a re-run.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import torch

    from pylatro import GameData
    from pylatro_agent.vocab import Vocab

logger = logging.getLogger(__name__)

# Versioned fixed seed list. Never edit in place, create EVAL_SEEDS_V2 if the
# seed semantics must change, so historical paired comparisons stay reproducible.
EVAL_SEEDS_V1: list[int] = list(range(10_000, 10_400))


@dataclass(slots=True)
class SeedOutcome:
    """Outcome of a single greedy eval game on one fixed seed."""

    seed: int
    won: bool
    max_ante: int
    round_score: int

    @property
    def won_int(self) -> int:
        return int(self.won)


@dataclass(slots=True)
class PairedDelta:
    """Result of a paired win-rate comparison between two policies."""

    win_rate_a: float
    win_rate_b: float
    delta: float  # win_rate_a - win_rate_b
    n_seeds: int
    # Discordant counts for McNemar's test:
    #   b = seeds where A won but B lost
    #   c = seeds where A lost but B won
    discordant_b: int
    discordant_c: int
    mcnemar_pvalue: float
    bootstrap_mean: float
    bootstrap_ci_low: float
    bootstrap_ci_high: float
    bootstrap_resamples: int

    def excludes_zero_at(self, confidence: float = 0.95) -> bool:
        """True if the (1-confidence) bootstrap CI excludes zero."""
        return self.bootstrap_ci_low > 0.0 or self.bootstrap_ci_high < 0.0


def evaluate_on_seeds(
    model,
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
) -> list[SeedOutcome]:
    """Evaluate ``model`` on a fixed list of seeds, returning per-seed outcomes.

    Each seed is played with ``dist.mode()`` (greedy) action selection by
    default; ``greedy=False`` samples instead, which is the mode PPO actually
    trains. The engine is deterministic given the seed, so re-running the same
    model on the same seeds under greedy selection yields identical outcomes.

    Games are advanced in lockstep batches of ``batch_size`` sharing one policy
    forward pass; see :func:`pylatro_agent.training.ppo.run_seed_evaluation`.
    """
    from .training.ppo import run_seed_evaluation

    results = run_seed_evaluation(
        model,
        data,
        vocab,
        list(seeds),
        device,
        max_no_progress_steps=max_no_progress_steps,
        win_ante=win_ante,
        temperature=temperature,
        stake=stake,
        greedy=greedy,
        batch_size=batch_size,
    )
    return [
        SeedOutcome(
            seed=int(result["seed"]),
            won=bool(result["won"]),
            max_ante=int(result["max_ante"]),
            round_score=int(result["round_score"]),
        )
        for result in results
    ]


def evaluate_heuristic_on_seeds(
    data: GameData,
    vocab: Vocab,
    seeds: list[int],
    *,
    max_no_progress_steps: int = 256,
    win_ante: int | None = None,
    stake: int = 1,
) -> list[SeedOutcome]:
    """Evaluate the heuristic teacher on the same fixed seeds.

    The heuristic uses the raw engine mask (not the structured grammar), so it
    is evaluated directly against the BalatroEnv without a model. This gives a
    paired baseline against any learned policy on identical seeds.
    """
    from .env import BalatroEnv

    outcomes: list[SeedOutcome] = []
    for seed in seeds:
        env = BalatroEnv(
            seed=seed,
            data=data,
            vocab=vocab,
            stake=stake,
            max_steps=max_no_progress_steps,
            win_ante=win_ante,
        )
        env.reset()
        done = False
        info: dict = {}
        while not done:
            action = _heuristic_action(env)
            _obs, _reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        won = bool(info.get("won", False))
        max_ante = int(info.get("ante", 1))
        round_score = int(info.get("round_score", 0))
        outcomes.append(SeedOutcome(seed=seed, won=won, max_ante=max_ante, round_score=round_score))

    return outcomes


def _heuristic_action(env) -> int:
    """Pick the heuristic teacher's action for the current env state.

    Delegates to the env's own teacher plumbing (the same query PPO
    distillation uses), so the baseline is exactly the teacher the policy is
    trained against, not a reimplementation that can drift on mask or
    round-score details. gymnasium.Env.unwrapped is a property, not a method.
    """
    env_unwrapped = env.unwrapped
    action = env_unwrapped._current_teacher_action()
    if action >= 0:
        return action
    # No valid teacher action (shouldn't happen on the heuristic's own
    # trajectory); fall back to the first legal action so eval cannot deadlock.
    valid = np.flatnonzero(env_unwrapped.action_masks())
    if len(valid) == 0:
        raise RuntimeError("Heuristic eval: no valid actions available.")
    return int(valid[0])


def save_outcomes(outcomes: list[SeedOutcome], path: str | Path) -> Path:
    """Persist per-seed outcomes to JSON (and a sibling CSV for dashboards)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seeds_version": "v1",
        "n": len(outcomes),
        "win_rate": _mean([o.won_int for o in outcomes]),
        "outcomes": [asdict(o) for o in outcomes],
    }
    with open(path, "w") as f:
        json.dump(payload, f)
    csv_path = path.with_suffix(".csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["seed", "won", "max_ante", "round_score"])
        writer.writeheader()
        for o in outcomes:
            row = asdict(o)
            row["won"] = int(o.won)
            writer.writerow(row)
    return path


def load_outcomes(path: str | Path) -> list[SeedOutcome]:
    """Load per-seed outcomes saved by :func:`save_outcomes`."""
    path = Path(path)
    with open(path) as f:
        payload = json.load(f)
    return [
        SeedOutcome(
            seed=int(r["seed"]),
            won=bool(r["won"]),
            max_ante=int(r["max_ante"]),
            round_score=int(r["round_score"]),
        )
        for r in payload["outcomes"]
    ]


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def paired_win_rate_delta(
    outcomes_a: list[SeedOutcome],
    outcomes_b: list[SeedOutcome],
    *,
    bootstrap_resamples: int = 10_000,
    confidence: float = 0.95,
    rng: np.random.Generator | None = None,
) -> PairedDelta:
    """Win-rate delta with McNemar test + paired bootstrap CI.

    Both outcome lists must cover the *same* set of seeds. Seeds are aligned by
    value; a missing seed in either list is dropped from the comparison (with a
    warning) so a partial re-run cannot silently bias the delta.

    The bootstrap resamples the per-seed win-indicator *differences* (paired),
    which captures the variance reduction from identical deck orders. The McNemar
    test uses the exact binomial form for the discordant-pair count.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    by_a = {o.seed: o.won_int for o in outcomes_a}
    by_b = {o.seed: o.won_int for o in outcomes_b}
    common = sorted(set(by_a) & set(by_b))
    dropped = (set(by_a) ^ set(by_b))
    if dropped:
        logger.warning(
            "Paired comparison: %d seeds not shared between the two outcome sets; "
            "dropped from the comparison.",
            len(dropped),
        )

    n = len(common)
    if n == 0:
        raise ValueError("Paired comparison requires at least one shared seed.")

    wins_a = np.array([by_a[s] for s in common], dtype=np.float64)
    wins_b = np.array([by_b[s] for s in common], dtype=np.float64)
    wr_a = float(wins_a.mean())
    wr_b = float(wins_b.mean())
    delta = wr_a - wr_b

    # McNemar exact test on discordant pairs.
    b = int(np.sum((wins_a == 1) & (wins_b == 0)))  # A won, B lost
    c = int(np.sum((wins_a == 0) & (wins_b == 1)))  # A lost, B won
    mcnemar_pvalue = _mcnemar_exact_pvalue(b, c)

    # Paired bootstrap over per-seed differences.
    diffs = wins_a - wins_b
    boot_means = np.empty(bootstrap_resamples, dtype=np.float64)
    for i in range(bootstrap_resamples):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = diffs[idx].mean()
    alpha = 1.0 - confidence
    ci_low = float(np.percentile(boot_means, 100 * alpha / 2))
    ci_high = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))

    return PairedDelta(
        win_rate_a=wr_a,
        win_rate_b=wr_b,
        delta=delta,
        n_seeds=n,
        discordant_b=b,
        discordant_c=c,
        mcnemar_pvalue=mcnemar_pvalue,
        bootstrap_mean=float(boot_means.mean()),
        bootstrap_ci_low=ci_low,
        bootstrap_ci_high=ci_high,
        bootstrap_resamples=bootstrap_resamples,
    )


def _mcnemar_exact_pvalue(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the binomial over discordant pairs.

    Under the null (no difference between paired proportions) the discordant
    count splits evenly: b ~ Binomial(b+c, 0.5). The two-sided p-value sums the
    tail probabilities at least as extreme as the observed split.
    """
    from math import comb

    n_disc = b + c
    if n_disc == 0:
        return 1.0
    k = min(b, c)
    # P(X <= k or X >= n_disc - k) under Binomial(n_disc, 0.5)
    tail = 0.0
    for i in range(0, k + 1):
        tail += comb(n_disc, i) * (0.5 ** n_disc)
    # Two-sided: double the smaller tail (handles the symmetric upper tail).
    p = min(1.0, 2.0 * tail)
    return p


def evaluate_history_ablation(
    model,
    records: list[dict[str, Any]],
    device: torch.device,
    *,
    batch_size: int = 256,
) -> dict[str, dict[str, dict[str, float]]]:
    """Report held-out prediction/policy metrics with and without history.

    Results are split by empty/partial/full three-round context and by whether
    the visible context contains a tagged DNA, Dusk, or Burglar event.  The
    ``context_reliance`` section is ``masked - normal`` for every metric, so a
    positive loss delta measures actual reliance by the trained checkpoint.
    """
    import torch
    import torch.nn.functional as F

    from .action import ActionType, decode_action
    from .training.ppo import _grammar_distribution
    from .training.supervised import SupervisedConfig, _collate_batch

    if not records:
        return {"normal": {}, "masked": {}, "context_reliance": {}}

    groups = (
        "overall",
        "history_empty",
        "history_partial",
        "history_full",
        "mechanic_flagged",
        "ordinary_history",
        "play_actions",
        "discard_actions",
    )
    totals: dict[str, dict[str, dict[str, float]]] = {
        mode: {
            group: {
                "count": 0.0,
                "return_abs_error": 0.0,
                "outcome_count": 0.0,
                "outcome_brier": 0.0,
                "outcome_nll": 0.0,
                "derived_win_brier": 0.0,
                "policy_nll": 0.0,
            }
            for group in groups
        }
        for mode in ("normal", "masked")
    }
    config = SupervisedConfig()
    was_training = model.training
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(records), batch_size):
            chunk = records[start : start + batch_size]
            batch = _collate_batch(chunk, device, config=config)
            round_counts = batch["history_round_mask"].sum(dim=-1)
            mechanic = (batch["history_events"][..., 3] != 0).any(dim=(-1, -2))
            action_types = [decode_action(int(record["action"])).action_type for record in chunk]
            group_masks = {
                "overall": torch.ones(len(chunk), dtype=torch.bool, device=device),
                "history_empty": round_counts == 0,
                "history_partial": (round_counts > 0) & (round_counts < 3),
                "history_full": round_counts == 3,
                "mechanic_flagged": mechanic,
                "ordinary_history": (round_counts > 0) & ~mechanic,
                "play_actions": torch.tensor(
                    [action_type == ActionType.PLAY_SUBSET for action_type in action_types],
                    dtype=torch.bool,
                    device=device,
                ),
                "discard_actions": torch.tensor(
                    [action_type == ActionType.DISCARD_SUBSET for action_type in action_types],
                    dtype=torch.bool,
                    device=device,
                ),
            }
            for mode in ("normal", "masked"):
                model_batch = batch
                if mode == "masked":
                    model_batch = dict(batch)
                    for key in (
                        "history_events",
                        "history_event_features",
                        "history_cards",
                        "history_card_mask",
                        "history_jokers",
                        "history_joker_mask",
                        "history_event_mask",
                        "history_round_mask",
                        "history_omitted",
                    ):
                        model_batch[key] = torch.zeros_like(batch[key])
                distribution, values = _grammar_distribution(model, model_batch)
                policy_nll = -distribution.log_prob(batch["actions"])
                return_error = (values["expected_return"] - batch["value_target"]).abs()
                outcome_mask = batch["terminal_outcome_mask"]
                outcome_target = batch["terminal_outcome_target"]
                outcome_one_hot = F.one_hot(
                    outcome_target,
                    num_classes=values["outcome_probabilities"].shape[1],
                ).to(dtype=values["outcome_probabilities"].dtype)
                outcome_brier = (
                    values["outcome_probabilities"] - outcome_one_hot
                ).square().sum(dim=-1)
                outcome_nll = -values["outcome_probabilities"].gather(
                    1, outcome_target.unsqueeze(1)
                ).squeeze(1).clamp_min(1e-7).log()
                win_target = (
                    outcome_target == values["outcome_probabilities"].shape[1] - 1
                ).float()
                derived_win_brier = (values["win_prob"] - win_target).square()
                for group, mask in group_masks.items():
                    count = int(mask.sum().item())
                    if not count:
                        continue
                    target = totals[mode][group]
                    target["count"] += count
                    target["return_abs_error"] += float(return_error[mask].sum().item())
                    target["policy_nll"] += float(policy_nll[mask].sum().item())
                    group_outcome_mask = outcome_mask[mask]
                    outcome_count = float(group_outcome_mask.sum().item())
                    target["outcome_count"] += outcome_count
                    target["outcome_brier"] += float(
                        (outcome_brier[mask] * group_outcome_mask).sum().item()
                    )
                    target["outcome_nll"] += float(
                        (outcome_nll[mask] * group_outcome_mask).sum().item()
                    )
                    target["derived_win_brier"] += float(
                        (derived_win_brier[mask] * group_outcome_mask).sum().item()
                    )

    if was_training:
        model.train()

    report: dict[str, dict[str, dict[str, float]]] = {"normal": {}, "masked": {}}
    for mode in ("normal", "masked"):
        for group, raw in totals[mode].items():
            count = raw["count"]
            if not count:
                continue
            outcome_count = max(raw["outcome_count"], 1.0)
            report[mode][group] = {
                "count": count,
                "expected_return_mae": raw["return_abs_error"] / count,
                "outcome_brier": raw["outcome_brier"] / outcome_count,
                "outcome_nll": raw["outcome_nll"] / outcome_count,
                "derived_win_brier": raw["derived_win_brier"] / outcome_count,
                "policy_nll": raw["policy_nll"] / count,
            }
    reliance: dict[str, dict[str, float]] = {}
    for group in set(report["normal"]) & set(report["masked"]):
        reliance[group] = {
            metric: report["masked"][group][metric] - report["normal"][group][metric]
            for metric in report["normal"][group]
            if metric != "count"
        }
        reliance[group]["count"] = report["normal"][group]["count"]
    report["context_reliance"] = reliance
    return report
