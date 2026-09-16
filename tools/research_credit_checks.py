#!/usr/bin/env python3
"""CPU-only numerical checks and descriptive forecast audit; no model fitting."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from pylatro_agent.training.cohort import fragment_gae


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def numerical_checks():
    """Verify mathematical examples against the production GAE recursion."""
    rng = np.random.default_rng(1729)
    checks = 0
    maximum_error = 0.0
    for length in (1, 7, 101, 401):
        for lam in (0.0, 0.97, 1.0):
            for fragment in (16, 256, 10000):
                for offset in (0, 9):
                    rewards = rng.normal(size=length).astype(np.float32)
                    values = rng.normal(size=length).astype(np.float32)
                    error = rng.normal(size=length).astype(np.float32)
                    slots = np.arange(length) + offset
                    reference, _ = fragment_gae(
                        rewards, values, slots, gamma=1.0, gae_lambda=lam, fragment_length=fragment
                    )
                    perturbed, _ = fragment_gae(
                        rewards, values + error, slots, gamma=1.0, gae_lambda=lam, fragment_length=fragment
                    )
                    for t in range(length):
                        n = min(length - t, fragment - int(slots[t]) % fragment)
                        tail_error = float(error[t + n]) if t + n < length else 0.0
                        derived = -float(error[t])
                        derived += (1 - lam) * sum(lam ** (j - 1) * float(error[t + j]) for j in range(1, n))
                        derived += lam ** (n - 1) * tail_error
                        difference = abs(float(perturbed[t] - reference[t]) - derived)
                        maximum_error = max(maximum_error, difference)
                        assert difference < 2e-5, (length, lam, fragment, offset, t, difference)
                        checks += 1

    rewards = np.zeros(401, dtype=np.float32)
    rewards[-1] = 10.0
    values = np.full(401, 4.0, dtype=np.float32)
    slots = np.arange(401)
    examples = []
    for lam, fragment in ((0.97, 256), (1.0, 256), (0.97, 10000), (1.0, 10000)):
        advantage, target = fragment_gae(rewards, values, slots, gamma=1.0, gae_lambda=lam, fragment_length=fragment)
        shifted_advantage, shifted_target = fragment_gae(
            rewards, values + 1.0, slots, gamma=1.0, gae_lambda=lam, fragment_length=fragment
        )
        examples.append(
            {
                "lambda": lam,
                "fragment_length": fragment,
                "opening_advantage": float(advantage[0]),
                "opening_return_target": float(target[0]),
                "opening_advantage_change_for_plus_one_value_offset": float(shifted_advantage[0] - advantage[0]),
                "opening_target_change_for_plus_one_value_offset": float(shifted_target[0] - target[0]),
            }
        )
        if lam == 1.0 and fragment > len(values):
            np.testing.assert_allclose(target, 10.0)
            np.testing.assert_allclose(shifted_target, target)
        if fragment == 256:
            np.testing.assert_allclose(shifted_target[:256] - target[:256], 1.0)

    # One-state two-action score-function expectation, evaluated exactly.
    # Both actions can win or lose: a single outcome cannot label action quality.
    action_probability = 0.3
    win_probabilities = (0.2, 0.7)
    expected_gradient = action_probability * (1 - action_probability) * 10 * (0.7 - 0.2)
    baseline_checks = []
    for baseline in (-5.0, 0.0, 3.5, 10.0, 20.0):
        gradient = 0.0
        for action, probability in ((0, 1 - action_probability), (1, action_probability)):
            for win in (0, 1):
                outcome_probability = win_probabilities[action] if win else 1 - win_probabilities[action]
                gradient += probability * outcome_probability * (action - action_probability) * (10 * win - baseline)
        assert abs(gradient - expected_gradient) < 1e-12
        baseline_checks.append({"baseline": baseline, "expected_logit_gradient": gradient})
    return {
        "gae_algebra_checks": checks,
        "maximum_absolute_float32_error": maximum_error,
        "constant_offset_examples": examples,
        "exact_mc_baseline_invariance": baseline_checks,
        "perfect_bernoulli_critic_p04_irreducible_mse": 100 * 0.4 * 0.6,
        "scope": "Synthetic mathematical identities, not evidence of a Balatro gameplay cause or PPO improvement.",
    }


def forecast_audit(root):
    summaries, provenance = {}, {}
    for endpoint in (0, 20, 40, 60):
        games = []
        for replica in (0, 1):
            path = root / f"evaluations/update{endpoint}/seed{replica}/sampled/outcomes.jsonl"
            provenance[str(path)] = sha256(path)
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            assert len(rows) == 50
            assert sorted(row["seed"] for row in rows) == list(range(190000, 190050))
            assert all(row["sampling_seed"] == 0 and not row["stalled"] and not row["censored"] for row in rows)
            games.extend(rows)
        y = np.asarray([int(row["won"]) for row in games], dtype=float)
        reference = float(y.mean())
        state_p, state_y, state_weight = [], [], []
        scores, initial_scores, full_mse, unclipped_brier = [], [], [], []
        saturation_rows = 0
        for row in games:
            forecasts = row["critic_forecasts"]
            assert len(forecasts) == row["steps"]
            p = np.array([f["outcome_probabilities"][-1] for f in forecasts])
            v = np.array([f["expected_return"] for f in forecasts])
            target = float(row["won"])
            scores.append(float(np.mean((p - target) ** 2)))
            initial_scores.append(float((p[0] - target) ** 2))
            full_mse.append(float(np.mean((v - 10 * target) ** 2)))
            unclipped_brier.append(float(np.mean((v / 10 - target) ** 2)))
            target_class = 8 if row["won"] else row["final_ante"] - 1
            saturation_rows += sum(f["outcome_probabilities"][target_class] <= 1e-7 for f in forecasts)
            state_p.extend(p)
            state_y.extend([target] * len(p))
            state_weight.extend([1 / len(p)] * len(p))
        p, outcomes, weights = map(np.asarray, (state_p, state_y, state_weight))
        bins = []
        for lower in np.arange(0, 1, 0.2):
            mask = (p >= lower) & ((p < lower + 0.2) if lower < 0.8 else (p <= 1))
            if mask.any():
                bins.append(
                    {
                        "lower": float(lower),
                        "upper": float(lower + 0.2),
                        "decision_count": int(mask.sum()),
                        "fraction_of_equal_game_weight": float(weights[mask].sum() / weights.sum()),
                        "mean_prediction": float(np.average(p[mask], weights=weights[mask])),
                        "observed_win_fraction": float(np.average(outcomes[mask], weights=weights[mask])),
                    }
                )
        constant_brier = float(np.mean((y - reference) ** 2))
        summaries[str(endpoint)] = {
            "games": len(games),
            "independent_game_seed_clusters": 50,
            "forecast_decisions": len(p),
            "sampled_win_rate": reference,
            "fitted_panel_constant_brier": constant_brier,
            "fitted_panel_constant_return_mse": 100 * constant_brier,
            "outcome_head_game_weighted_brier": float(np.mean(scores)),
            "outcome_head_opening_brier": float(np.mean(initial_scores)),
            "outcome_head_brier_reduction_vs_fitted_constant": 1 - float(np.mean(scores)) / constant_brier,
            "composed_value_game_weighted_mse": float(np.mean(full_mse)),
            "realized_outcome_probability_at_nll_floor": saturation_rows,
            "coarse_game_weighted_reliability_bins": bins,
            "caveats": [
                "Fitted constant uses this evaluation panel's outcomes; descriptive reference, not held-out baseline.",
                "Later states contain more outcome information. Improvement over constant is not action credit.",
                "Repeated states/outcomes are correlated; decision counts are not independent sample sizes.",
                "Reliability bins have no inferential intervals and do not identify approximation error versus noise.",
            ],
        }
        np.testing.assert_allclose(np.mean(unclipped_brier) * 100, np.mean(full_mse), atol=1e-10)
    return summaries, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=Path("eval/complete_cohort_long_20260915"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    forecasts, provenance = forecast_audit(args.study)
    for path in (Path(__file__), Path("src/pylatro_agent/training/cohort.py")):
        provenance[str(path)] = sha256(path)
    result = {
        "accelerator_seconds": 0,
        "model_updates": 0,
        "new_game_rollouts": 0,
        "numerical_checks": numerical_checks(),
        "saved_forecasts": forecasts,
        "input_sha256": provenance,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "numerical_checks": result["numerical_checks"]}, indent=2))


if __name__ == "__main__":
    main()
