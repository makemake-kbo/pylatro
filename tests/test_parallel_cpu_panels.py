"""Concurrent CPU evaluation must preserve each policy's seeded game results."""

import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch


def test_parallel_policy_processes_match_serial_sampling_and_greedy(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    from evaluate_action_heads import run_panel
    from parallel_cpu_panels import run_parallel_panels

    from pylatro import load_game_data
    from pylatro_agent.agent import AgentConfig, BalatroAgent
    from pylatro_agent.checkpoint import save_checkpoint
    from pylatro_agent.reward import DEFAULT_REWARD_CONFIG, reward_checkpoint_metadata
    from pylatro_agent.vocab import build_vocab

    assert "tools" in sys.path[0]
    torch.set_num_threads(4)
    torch.manual_seed(1729)
    config = AgentConfig(
        d_model=16, n_layers=1, n_heads=2, d_ff=32, dropout=0.0,
        hand_ar_mixture_eps=0.5, action_head_variant="entity_sale_subset_residual_v1",
    )
    model = BalatroAgent(config, build_vocab(load_game_data()))
    checkpoint = tmp_path / "tiny.pt"
    save_checkpoint(model, checkpoint, extra={
        "agent_config": asdict(config), **reward_checkpoint_metadata(DEFAULT_REWARD_CONFIG),
    })
    seeds = [1729, 1730]
    (tmp_path / "manifest.json").write_text(json.dumps({"test_only": True, "seeds": seeds}))
    (tmp_path / "cache_complete.json").write_text("{}")
    parallel = run_parallel_panels(tmp_path, "parallel", checkpoint, [checkpoint, checkpoint], seeds)
    assert all(
        json.loads(path.read_text())["status"] == "complete"
        for path in (tmp_path / "cpu_workers").glob("*/evaluation_status.json")
    )
    serial = run_panel(tmp_path, "serial", checkpoint, seeds, torch.device("cpu"))

    def strip_timing(results):
        timing = {"elapsed_seconds", "policy_seconds", "env_seconds", "env_wall_seconds"}
        return [[{k: v for k, v in row.items() if k not in timing} for row in rows] for rows in results]

    for results in parallel.values():
        assert strip_timing(results) == strip_timing(serial)
    # A resumed comparison consumes completed artifacts, without rerunning games.
    resumed = run_parallel_panels(tmp_path, "parallel", checkpoint, [checkpoint, checkpoint], seeds)
    assert resumed == parallel
    assert len(list((tmp_path / "evaluations").glob("parallel_*"))) == 3
    artifacts = list(tmp_path.rglob("evaluations/**/manifest.json"))
    assert len(artifacts) == len({path.resolve() for path in artifacts}) == 16
