import copy
import importlib.util
import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def selection_module():
    directory = Path(__file__).resolve().parents[1] / "tools"
    sys.path.insert(0, str(directory))
    try:
        spec = importlib.util.spec_from_file_location(
            "confirm_action_refactor", directory / "confirm_action_refactor.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.path.remove(str(directory))


def report():
    return dict(
        results={
            "sampled": {
                "paired": {"sampled_or_greedy_delta": 0.12, "paired_ci95": [0.05, 0.19]},
                "candidate": {"win_rate": 0.2},
            },
            "greedy": {"paired": {"paired_ci95": [-0.02, 0.1]}, "candidate": {"win_rate": 0.3}},
            "per_replicate": {str(seed): {"sampled": {"sampled_or_greedy_delta": 0.12}} for seed in (0, 1)},
        },
        acceptance={"no_stalled_or_censored_games": True},
    )


def test_development_selection_requires_both_training_replicates(selection_module):
    data = report()
    assert selection_module.selection_rule(data) == (True, (0.2, 0.3))
    data["results"]["per_replicate"]["1"]["sampled"]["sampled_or_greedy_delta"] = -0.01
    assert selection_module.selection_rule(data)[0] is False


def test_development_selection_retains_uncertainty_and_greedy_guard(selection_module):
    original = report()
    for kind in ("sampled", "greedy"):
        data = copy.deepcopy(original)
        data["results"][kind]["paired"]["paired_ci95"][0] = -0.06
        assert selection_module.selection_rule(data)[0] is False
    original["acceptance"]["no_stalled_or_censored_games"] = False
    assert selection_module.selection_rule(original)[0] is False


def test_development_threshold_is_not_misrepresented_as_final_acceptance(selection_module):
    data = report()
    data["results"]["sampled"]["paired"] = {"sampled_or_greedy_delta": 0.04, "paired_ci95": [0.01, 0.07]}
    for seed in ("0", "1"):
        data["results"]["per_replicate"][seed]["sampled"]["sampled_or_greedy_delta"] = 0.04
    assert selection_module.selection_rule(data)[0] is True
    # This only admits an untouched confirmation experiment. Its independent
    # final acceptance still requires at least a five-percentage-point gain.
