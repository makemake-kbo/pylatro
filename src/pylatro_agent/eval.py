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
from typing import TYPE_CHECKING

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
) -> list[SeedOutcome]:
    """Greedy-evaluate ``model`` on a fixed list of seeds, returning per-seed outcomes.

    Each seed is played with ``dist.mode()`` (greedy) action selection. The
    engine is deterministic given the seed, so re-running the same model on the
    same seeds yields identical outcomes.

    Memory safety mirrors :func:`pylatro_agent.training.ppo.evaluate_model`:
    runs under ``torch.inference_mode`` and drains the MPS cache periodically.
    """
    import torch

    from .env import BalatroEnv
    from .training.ppo import _grammar_distribution, _single_obs_to_batch

    model.eval()
    outcomes: list[SeedOutcome] = []

    with torch.inference_mode():
        for i, seed in enumerate(seeds):
            if device.type == "mps" and i > 0 and i % 50 == 0:
                torch.mps.empty_cache()

            env = BalatroEnv(
                seed=seed,
                data=data,
                vocab=vocab,
                stake=stake,
                max_steps=max_no_progress_steps,
                win_ante=win_ante,
            )
            obs, _ = env.reset()
            done = False
            info: dict = {}
            while not done:
                batch = _single_obs_to_batch(obs, device)
                dist, _ = _grammar_distribution(model, batch, temperature=temperature)
                action = dist.mode().item()
                del batch, dist
                obs, _reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

            won = bool(info.get("won", False))
            max_ante = int(info.get("ante", 1))
            round_score = int(info.get("round_score", 0))
            outcomes.append(SeedOutcome(seed=seed, won=won, max_ante=max_ante, round_score=round_score))

    return outcomes


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
