import importlib.util
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("phase", ["supervised", "ppo"])
def test_training_cli_propagates_versioned_heads_and_reference_mixture(monkeypatch, phase):
    path = Path(__file__).resolve().parents[1] / "train.py"
    spec = importlib.util.spec_from_file_location("refactor_training_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    captured = {}

    def capture(config, *, agent_config, **kwargs):
        captured.update(config=config, agent_config=agent_config)

    target = f"pylatro_agent.training.{phase}.train_{phase}"
    monkeypatch.setattr(target, capture)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            phase,
            "--device",
            "cpu",
            "--deck",
            "b_blue",
            "--action-head-variant",
            "entity_sale_subset_residual_v1",
            "--hand-ar-mixture-eps",
            "0.5",
        ],
    )
    module.main()
    assert captured["agent_config"].action_head_variant == "entity_sale_subset_residual_v1"
    assert captured["agent_config"].hand_ar_mixture_eps == 0.5
    assert captured["config"].hand_ar_mixture_eps == 0.5
    assert captured["config"].deck_key == "b_blue"
    # Both training functions are replaced with the capture stub: this never
    # constructs an optimizer, launches environments, or starts a PPO run.
