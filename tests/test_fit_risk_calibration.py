"""The risk-calibration fitter must recover known constants and stay bounded."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

_SPEC = importlib.util.spec_from_file_location(
    "fit_risk_calibration",
    Path(__file__).resolve().parents[1] / "tools" / "fit_risk_calibration.py",
)
assert _SPEC is not None and _SPEC.loader is not None
fit_risk_calibration = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(fit_risk_calibration)

fit_platt = fit_risk_calibration.fit_platt


def _synthetic(scale: float, bias: float, n: int = 4000, seed: int = 7):
    rng = np.random.default_rng(seed)
    raw_logit = rng.normal(2.0, 2.0, n)
    probability = 1.0 / (1.0 + np.exp(-(scale * raw_logit + bias)))
    outcome = (rng.random(n) < probability).astype(float)
    return raw_logit, outcome


def test_fit_recovers_known_calibration() -> None:
    raw_logit, outcome = _synthetic(0.35, -0.90)

    scale, bias = fit_platt(raw_logit, outcome)

    assert scale == pytest_approx(0.35, tol=0.06)
    assert bias == pytest_approx(-0.90, tol=0.15)


def test_fit_stays_bounded_on_separable_data() -> None:
    """Undamped Newton exploded here (scale ran to ~1e5); the fit must not."""
    raw_logit = np.concatenate([np.linspace(-6.0, -1.0, 500), np.linspace(1.0, 6.0, 500)])
    outcome = np.concatenate([np.zeros(500), np.ones(500)])

    scale, bias = fit_platt(raw_logit, outcome)

    assert np.isfinite(scale) and np.isfinite(bias)
    assert abs(scale) < 100.0
    assert abs(bias) < 100.0


def test_fit_never_increases_log_loss() -> None:
    raw_logit, outcome = _synthetic(0.5, 0.3, seed=11)
    identity_loss = fit_risk_calibration._mean_log_loss(1.0, 0.0, raw_logit, outcome)

    scale, bias = fit_platt(raw_logit, outcome)
    fitted_loss = fit_risk_calibration._mean_log_loss(scale, bias, raw_logit, outcome)

    assert fitted_loss <= identity_loss


def test_fit_is_a_noop_on_already_calibrated_input() -> None:
    raw_logit, outcome = _synthetic(1.0, 0.0, n=8000, seed=3)

    scale, bias = fit_platt(raw_logit, outcome)

    assert scale == pytest_approx(1.0, tol=0.12)
    assert bias == pytest_approx(0.0, tol=0.12)


def pytest_approx(expected: float, tol: float):
    class _Approx:
        def __eq__(self, other: object) -> bool:
            return abs(float(other) - expected) <= tol  # type: ignore[arg-type]

        def __repr__(self) -> str:
            return f"{expected} +- {tol}"

    return _Approx()
