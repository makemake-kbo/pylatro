"""The long-run launcher must record its recipe and stop on stage failure."""

from __future__ import annotations

import json
import subprocess

import pytest

from pylatro_agent.training import pt_ppo_run as runner


def test_long_run_recipe(tmp_path):
    args = runner._parser().parse_args(["--run-dir", str(tmp_path)])
    agent, supervised, ppo = runner._configs(args)
    assert agent.precision == ppo.precision == "bf16"
    assert (supervised.num_games, supervised.max_epochs, supervised.num_workers) == (1000, 5, 8)
    assert supervised.chunk_size == supervised.num_games
    assert ppo.num_envs * ppo.rollout_length * ppo.total_updates == 4_096_000
    assert ppo.total_timesteps == 4_096_000
    assert ppo.reward_config.objective == "milestone"
    assert ppo.win_ante == supervised.win_ante == 8
    assert ppo.gamma == supervised.gamma
    assert ppo.archive_config.min_ante == 4
    assert ppo.device == supervised.device == "cuda:0"


def test_failure_does_not_start_ppo_or_overwrite_manifest(tmp_path, monkeypatch):
    calls = []

    def fail(command, **kwargs):
        calls.append(command)
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(runner.multiprocessing, "set_start_method", lambda *a, **kw: None)
    monkeypatch.setattr(runner.subprocess, "run", fail)
    monkeypatch.setattr(runner.shutil, "disk_usage", lambda _: type("Disk", (), {"free": 10 * 1024**3})())
    argv = ["--run-dir", str(tmp_path)]
    with pytest.raises(subprocess.CalledProcessError):
        runner.main(argv)
    assert len(calls) == 1 and calls[0][-1] == "supervised"
    assert json.loads((tmp_path / "status.json").read_text())["state"] == "failed"
    manifest = (tmp_path / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        runner.main(argv)
    assert (tmp_path / "manifest.json").read_bytes() == manifest
    assert len(calls) == 1


def test_pipeline_orders_stages_and_records_completion(tmp_path, monkeypatch):
    stages = []
    monkeypatch.setattr(runner.multiprocessing, "set_start_method", lambda *a, **kw: None)
    monkeypatch.setattr(runner.subprocess, "run", lambda command, **kw: stages.append(command[-1]))
    monkeypatch.setattr(runner.shutil, "disk_usage", lambda _: type("Disk", (), {"free": 10 * 1024**3})())
    runner.main(["--run-dir", str(tmp_path)])
    assert stages == ["supervised", "ppo"]
    assert json.loads((tmp_path / "status.json").read_text())["state"] == "complete"
    assert json.loads((tmp_path / "manifest.json").read_text())["optimizer_moments_dtype"] == "float32"
