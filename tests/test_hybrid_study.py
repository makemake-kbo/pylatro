from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def module(name):
    path = Path(__file__).resolve().parents[1] / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


study = module("hybrid_study")
cleanup = module("retire_experiments")


def rows(wins):
    return [
        dict(
            seed=f"seed-{i}",
            won=win,
            complete=True,
            steps=1,
            changes=0,
            searches=0,
            fallbacks=0,
            paired_sample_sets=0,
            seconds=1,
            decision_seconds=[1],
            illegal_actions=0,
        )
        for i, win in enumerate(wins)
    ]


def test_bootstrap_pairs_games_and_reproduces_exact_counts():
    base = rows([False] * 50 + [True] * 50)
    better = rows([False] * 35 + [True] * 65)
    result = study.paired(base, better)
    assert result["points"] == 15
    assert result["gained"] == 15 and result["lost"] == 0
    assert result["interval95"][0] > 0
    assert result == study.paired(base, better)
    assert study.paired(base, base)["interval95"] == [0, 0]
    with pytest.raises(ValueError, match="identical complete"):
        study.paired(base, better[:-1])
    with pytest.raises(ValueError, match="identical complete"):
        study.paired(base, better[::-1])


def test_advancement_needs_five_points_and_prefers_fewer_phases():
    panels = {f"model/{m}": rows([True] * 50 + [False] * 50) for m in study.MODES}
    assert study.select_assisted(panels, "model") is None
    panels["model/both"] = rows([True] * 55 + [False] * 45)
    assert study.select_assisted(panels, "model") == "both"
    panels["model/hand"] = rows([True] * 55 + [False] * 45)
    assert study.select_assisted(panels, "model") == "hand"


def test_resume_rejects_missing_mismatched_or_incomplete_games(tmp_path):
    path = tmp_path / "stage/model/off/seed-0.json"
    with pytest.raises(ValueError, match="Missing"):
        study.read_panel(tmp_path, "stage", "model", "off", ["seed-0"])
    result = dict(rows([False])[0], model="model", mode="off")
    study.atomic(path, result)
    assert study.read_panel(tmp_path, "stage", "model", "off", ["seed-0"]) == [result]
    with pytest.raises(ValueError, match="Duplicate"):
        study.read_panel(tmp_path, "stage", "model", "off", ["seed-0", "seed-0"])
    study.atomic(path, dict(result, complete=False))
    with pytest.raises(ValueError, match="Invalid"):
        study.read_panel(tmp_path, "stage", "model", "off", ["seed-0"])


def test_seed_panels_are_fixed_disjoint_and_distinct_from_historical_integer_ranges():
    if not (study.DEFAULT / "seeds.json").exists():
        pytest.skip("Local study reservation artifact is not installed")
    seeds = json.loads((study.DEFAULT / "seeds.json").read_text())
    assert {key: len(value) for key, value in seeds.items()} == dict(
        pilot=16, baseline=128, development=256, confirmation=400, sampled=128
    )
    all_seeds = [s for panel in seeds.values() for s in panel]
    assert len(all_seeds) == len(set(all_seeds))
    reservation = json.loads((study.DEFAULT / "reservation.json").read_text())
    assert all(s.startswith(reservation["namespace"] + "-") and not s.isdecimal() for s in all_seeds)
    old = reservation.get("invalidated_attempt_seed_exclusions", {})
    assert set(all_seeds).isdisjoint(s for values in old.values() for s in values)


def test_correctness_restart_does_not_extend_original_wall_cap():
    clock = dict(started_unix=0, deadline_unix=48 * 3600)
    assert study.technical_deadline(clock) == 4 * 3600
    clock["technical_started_unix"] = 10 * 3600
    assert study.technical_deadline(clock) == 14 * 3600
    clock["technical_started_unix"] = 47 * 3600
    assert study.technical_deadline(clock) == clock["deadline_unix"]


def test_summary_numpy_scalars_serialize_without_changing_values():
    import numpy as np

    recovery = module("resume_hybrid_study")
    value = dict(success=np.bool_(True), interval=[np.float64(5.25), np.float64(10)], games=np.int64(400))
    converted = recovery.json_native(value)
    assert converted == dict(success=True, interval=[5.25, 10.0], games=400)
    assert type(converted["success"]) is bool
    assert recovery.json_native(dict(success=np.bool_(False))) == dict(success=False)
    with pytest.raises(TypeError, match="Unsupported JSON"):
        recovery.json_native(dict(invalid=object()))


def test_retention_inventory_and_verified_receipt():
    root = cleanup.ROOT
    for name, path in cleanup.MODELS.items():
        assert cleanup.classify(root / path) == ("keep", f"incumbents/{name}/checkpoint.pt")
    for _name, (path, _) in cleanup.DATASETS.items():
        assert cleanup.classify(root / path / "game_1.parquet")[0] == "keep"
    assert (
        cleanup.classify(root / "eval/pretraining_confirmation_20260923/stage3/teacher/game_1.parquet")[0] == "archive"
    )
    assert cleanup.classify(root / "eval/old/checkpoint.pt")[0] == "delete"
    assert cleanup.classify(root / "eval/old/reserved_outcomes.jsonl")[0] == "archive"
    if not (cleanup.ARCHIVE / "receipt.json").exists():
        pytest.skip("Local retirement artifacts are not installed")
    receipt = json.loads((cleanup.ARCHIVE / "receipt.json").read_text())
    assert receipt["original_checkpoints_untouched"]
    assert receipt["eval_after_allocated_bytes"] < receipt["before"]["allocated_bytes"]
    models = json.loads((root / "incumbents/registry.json").read_text())
    assert set(models) == set(cleanup.MODELS)
    assert all(model["strict_load"] for model in models.values())
    datasets = json.loads((root / "datasets/registry.json").read_text())
    assert {name: value["games"] for name, value in datasets.items()} == dict(
        original_5000=5000, current_1000=1000, current_4000=4000
    )
