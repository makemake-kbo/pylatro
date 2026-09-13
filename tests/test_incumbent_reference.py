import json
import runpy

import pytest


def test_supplementary_reference_does_not_start_without_its_budget(tmp_path, monkeypatch):
    function = runpy.run_path("tools/learning_bottleneck.py")["incumbent_reference"]
    (tmp_path / "incumbent_reference_plan.json").write_text(json.dumps({"budget_gate_seconds": 1656}))
    (tmp_path / "compute.json").write_text(json.dumps({"jobs": [{"elapsed_seconds": 30745}]}))

    def unexpected_evaluation(*args, **kwargs):
        raise AssertionError("An optional evaluation spent the protected remaining budget")

    monkeypatch.setitem(function.__globals__, "eval_one", unexpected_evaluation)
    function(tmp_path, "pure_ar", {}, {})
    result = json.loads((tmp_path / "incumbent_reference_comparison.json").read_text())
    assert result["status"] == "skipped_budget"
    assert result["remaining_seconds"] == 1655


@pytest.mark.parametrize("remaining,expected_games", [(3192, 400), (3191, 200), (1656, 200)])
def test_reference_panel_size_depends_only_on_available_time(tmp_path, monkeypatch, remaining, expected_games):
    function = runpy.run_path("tools/learning_bottleneck.py")["incumbent_reference"]
    seeds = list(range(400))
    plan = dict(seeds=seeds[:200], budget_gate_seconds=1656, larger_panel=dict(seeds=seeds, budget_gate_seconds=3192))
    (tmp_path / "incumbent_reference_plan.json").write_text(json.dumps(plan))
    (tmp_path / "compute.json").write_text(json.dumps({"jobs": [{"elapsed_seconds": 32400 - remaining}]}))

    def panels(panel_seeds):
        return tuple(
            [dict(seed=seed, sampling_seed=sample, won=False, max_ante=1) for seed in panel_seeds for sample in samples]
            for samples in ((0, 1, 2), (None,))
        )

    requested = []

    def evaluate(root, checkpoint, name, panel_seeds, **kwargs):
        requested.append(list(panel_seeds))
        return panels(panel_seeds)

    monkeypatch.setitem(function.__globals__, "eval_one", evaluate)
    runs = {seed: panels(seeds) for seed in (0, 1)}
    function(tmp_path, "hand_weight", runs, runs)
    result = json.loads((tmp_path / "incumbent_reference_comparison.json").read_text())
    assert requested == [seeds[:expected_games]]
    assert result["seeds"] == seeds[:expected_games]
    assert result["comparisons"]["hand_weight"]["sampled"]["game_seeds"] == expected_games
