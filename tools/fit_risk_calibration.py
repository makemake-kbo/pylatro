#!/usr/bin/env python3
"""Refit the analytic death-probability Platt calibration from real outcomes.

Input: one or more ``risk_forecasts.jsonl`` files written during PPO training
(``PPOConfig.risk_forecast_log``). Every line is one resolved shop-leave
forecast with the raw (pre-calibration) death probability and the realized
next-blind outcome, so the fit is over the exact population the calibrated
probability is consumed on.

Output: refit ``ANALYTIC_DEATH_LOGIT_SCALE`` / ``ANALYTIC_DEATH_LOGIT_BIAS``
constants for ``pylatro_agent/risk.py``, plus before/after Brier, log-loss,
AUC, and a reliability table. Changing the constants changes observation and
reward semantics - bump ``reward.REWARD_MODEL_VERSION`` when pasting them.

Usage:
    uv run python tools/fit_risk_calibration.py runs/<run>/risk_forecasts.jsonl
    uv run python tools/fit_risk_calibration.py --per-ante runs/*/risk_forecasts.jsonl
    uv run python tools/fit_risk_calibration.py --min-update 400 <jsonl...>
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

EPS = 1e-4


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _mean_log_loss(scale: float, bias: float, x: np.ndarray, y: np.ndarray) -> float:
    z = scale * x + bias
    # log(1+exp(-z)) computed stably, then the standard cross-entropy identity.
    return float(np.mean(np.logaddexp(0.0, z) - y * z))


def fit_platt(raw_logits: np.ndarray, outcomes: np.ndarray, iterations: int = 200) -> tuple[float, float]:
    """Fit ``sigmoid(scale * logit + bias)`` to binary outcomes.

    Damped Newton with a backtracking line search on the mean log-loss. Plain
    Newton is not safe here: once the current fit pushes every probability
    toward 0 or 1 the IRLS weights ``p(1-p)`` collapse, the Hessian becomes
    near-singular, and an undamped step explodes (observed: scale running to
    1e5 on well-conditioned synthetic data). Requiring each step to actually
    decrease the loss, and halving it until it does, keeps the fit bounded.
    """
    x = np.asarray(raw_logits, dtype=np.float64)
    y = np.asarray(outcomes, dtype=np.float64)
    scale, bias = 1.0, 0.0
    loss = _mean_log_loss(scale, bias, x, y)
    n = max(len(x), 1)

    for _ in range(iterations):
        z = scale * x + bias
        p = _sigmoid(z)
        w = p * (1.0 - p)
        g_scale = float(((p - y) * x).sum()) / n
        g_bias = float((p - y).sum()) / n
        # Ridge term keeps the Hessian invertible when the weights collapse.
        ridge = 1e-6
        h_ss = float((w * x * x).sum()) / n + ridge
        h_sb = float((w * x).sum()) / n
        h_bb = float(w.sum()) / n + ridge
        det = h_ss * h_bb - h_sb * h_sb
        if not np.isfinite(det) or abs(det) < 1e-15:
            break
        step_scale = (h_bb * g_scale - h_sb * g_bias) / det
        step_bias = (h_ss * g_bias - h_sb * g_scale) / det

        # Backtracking: accept the first damped step that lowers the loss.
        step = 1.0
        improved = False
        for _ in range(40):
            trial_scale = scale - step * step_scale
            trial_bias = bias - step * step_bias
            trial_loss = _mean_log_loss(trial_scale, trial_bias, x, y)
            if np.isfinite(trial_loss) and trial_loss <= loss:
                if abs(loss - trial_loss) < 1e-12:
                    scale, bias, loss = trial_scale, trial_bias, trial_loss
                    improved = False
                    break
                scale, bias, loss = trial_scale, trial_bias, trial_loss
                improved = True
                break
            step *= 0.5
        if not improved:
            break
    return scale, bias


def auc(predictions: np.ndarray, outcomes: np.ndarray) -> float | None:
    positive = outcomes > 0.5
    n_pos = int(positive.sum())
    n_neg = int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(predictions, kind="mergesort")
    ranks = np.empty(len(predictions))
    sorted_p = predictions[order]
    start = 0
    while start < len(predictions):
        end = start + 1
        while end < len(predictions) and sorted_p[end] == sorted_p[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    return (float(ranks[positive].sum()) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def report(name: str, predictions: np.ndarray, outcomes: np.ndarray) -> None:
    brier = float(np.mean((predictions - outcomes) ** 2))
    p = np.clip(predictions, EPS, 1.0 - EPS)
    logloss = float(-np.mean(outcomes * np.log(p) + (1.0 - outcomes) * np.log(1.0 - p)))
    base = float(outcomes.mean())
    climatology = base * (1.0 - base)
    skill = 1.0 - brier / climatology if climatology > 0 else float("nan")
    a = auc(predictions, outcomes)
    print(
        f"{name:26s} brier={brier:.4f} (skill {skill:+.3f})  logloss={logloss:.4f}  "
        f"auc={'--' if a is None else f'{a:.3f}'}  mean_pred={predictions.mean():.3f}  base_rate={base:.3f}"
    )


def reliability_table(predictions: np.ndarray, outcomes: np.ndarray, bins: int = 10) -> None:
    print("  bin        n   mean_pred  observed")
    edges = np.linspace(0.0, 1.0, bins + 1)
    for lo, hi in itertools.pairwise(edges):
        sel = (predictions >= lo) & (predictions < hi if hi < 1.0 else predictions <= hi)
        if not sel.any():
            continue
        print(
            f"  {lo:.1f}-{hi:.1f}  {int(sel.sum()):6d}   "
            f"{predictions[sel].mean():.3f}      {outcomes[sel].mean():.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", help="risk_forecasts.jsonl files")
    parser.add_argument("--per-ante", action="store_true", help="Also fit and report per shop ante.")
    parser.add_argument("--min-update", type=int, default=0, help="Ignore rows from updates before this.")
    parser.add_argument("--min-count", type=int, default=500, help="Minimum rows required to fit (default 500).")
    args = parser.parse_args()

    rows = []
    for path in args.paths:
        with open(Path(path)) as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if int(row.get("update", 0)) < args.min_update:
                    continue
                rows.append(row)
    if len(rows) < args.min_count:
        print(f"Only {len(rows)} usable rows (< --min-count {args.min_count}); refusing to fit noise.")
        sys.exit(1)

    raw_death = np.asarray([float(r["raw_death_pred"]) for r in rows])
    calibrated_death = np.asarray([float(r["calibrated_death_pred"]) for r in rows])
    outcome = np.asarray([float(r["next_blind_death"]) for r in rows])
    antes = np.asarray([int(r["shop_ante"]) for r in rows])
    raw_logits = _logit(raw_death)

    print(f"{len(rows)} resolved shop-leave forecasts from {len(args.paths)} file(s)\n")
    report("raw analytic", raw_death, outcome)
    report("current calibration", calibrated_death, outcome)

    scale, bias = fit_platt(raw_logits, outcome)
    refit = _sigmoid(scale * raw_logits + bias)
    report("refit calibration", refit, outcome)
    print("\nreliability (refit):")
    reliability_table(refit, outcome)

    print("\nPaste into pylatro_agent/risk.py (and bump reward.REWARD_MODEL_VERSION):")
    print(f"ANALYTIC_DEATH_LOGIT_SCALE = {scale:.4f}")
    print(f"ANALYTIC_DEATH_LOGIT_BIAS = {bias:.4f}")

    if args.per_ante:
        print("\nper-ante fits (diagnostic; the runtime mapping stays global):")
        for ante in sorted(set(antes.tolist())):
            sel = antes == ante
            if int(sel.sum()) < max(args.min_count // 5, 100):
                print(f"  ante {ante}: {int(sel.sum())} rows (too few, skipped)")
                continue
            s, b = fit_platt(raw_logits[sel], outcome[sel])
            fit = _sigmoid(s * raw_logits[sel] + b)
            brier = float(np.mean((fit - outcome[sel]) ** 2))
            print(
                f"  ante {ante}: n={int(sel.sum())} scale={s:.3f} bias={b:.3f} "
                f"brier={brier:.4f} base_rate={outcome[sel].mean():.3f}"
            )


if __name__ == "__main__":
    main()
