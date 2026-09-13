#!/usr/bin/env python3
"""Demonstrate the production Huber loss's target on a calibrated binary win task."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from pylatro_agent.learning_bottleneck import atomic_json
from pylatro_agent.outcome_study import ROOT
from pylatro_agent.transfer import sha256
from pylatro_agent.value_head import return_huber_loss


def audit_probability(probability):
    wins = round(probability * 100)
    returns = torch.tensor([10.] * wins + [0.] * (100 - wins), dtype=torch.float64)
    mean = returns.mean()
    residual = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))

    def outputs():
        return dict(terminal_value=mean.expand_as(returns), return_residual=residual.expand_as(returns))

    loss = return_huber_loss(outputs(), returns)
    initial_gradient = float(torch.autograd.grad(loss, residual)[0])
    mse = ((mean + residual) - returns).square().mean()
    mse_gradient = float(torch.autograd.grad(mse, residual)[0])
    optimizer = torch.optim.SGD([residual], lr=.2)
    for _ in range(1000):
        optimizer.zero_grad()
        return_huber_loss(outputs(), returns).backward()
        optimizer.step()
    optimum = (probability / (1 - probability) if probability < .5 else
               10 - (1 - probability) / probability if probability > .5 else 5.)
    fitted = float((mean + residual).detach())
    if abs(fitted - optimum) > 1e-8 or abs(mse_gradient) > 1e-12:
        raise ValueError("The production loss differs from the analytic benchmark")
    return dict(win_probability=probability, calibrated_mean=float(mean),
                huber_gradient_at_calibrated_mean=initial_gradient, mse_gradient_at_calibrated_mean=mse_gradient,
                fitted_huber_prediction=fitted, analytic_huber_optimum=optimum,
                prediction_bias=fitted - float(mean))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    torch.set_num_threads(1)
    destination = args.root / "return_loss_audit.json"
    if destination.exists():
        raise FileExistsError(destination)
    result = dict(
        reward_support=[0, 10], gamma=1, rows=[audit_probability(p) for p in (.1, .2, .3, .5, .8)],
        source_sha256={str(path): sha256(path) for path in
                       (Path(__file__), Path("src/pylatro_agent/value_head.py"))},
        accelerator_seconds=0,
        interpretation=("A synthetic, exactly calibrated fixed-state Bernoulli win distribution with complete "
                        "Monte Carlo returns. The production Huber correction estimates a robust location rather "
                        "than the conditional mean. Real PPO also has bootstrapped fragments and evolving policies. "
                        "This establishes an objective mismatch on the benchmark, not its contribution to gameplay. "
                        "With gamma one and only terminal utility, a calibrated outcome head already defines the "
                        "expected return; a separate return correction is unnecessary in this specific task."),
    )
    atomic_json(destination, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
