from __future__ import annotations

import importlib.util
import sys
from argparse import Namespace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("critic_sil_ablation_test", ROOT / "tools/run_critic_sil_ablation.py")
assert SPEC is not None
assert SPEC.loader is not None
ABLATION = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ABLATION
SPEC.loader.exec_module(ABLATION)

ARM_SPECS = ABLATION.ARM_SPECS
XMULT_BASE_ARGS = ABLATION.XMULT_BASE_ARGS
_paired_effects = ABLATION._paired_effects
_normalized_shared_args = ABLATION._normalized_shared_args
_parse_seeds = ABLATION._parse_seeds
build_summary = ABLATION.build_summary
create_manifest = ABLATION.create_manifest
execute_manifest = ABLATION.execute_manifest


def _run_args(tmp_path: Path, checkpoint: Path, **overrides) -> Namespace:
    values = {
        "checkpoint": str(checkpoint),
        "output_dir": str(tmp_path / "ablation"),
        "updates": 40,
        "seeds": [0, 7],
        "sil_coeff": 0.01,
        "sil_buffer_episodes": 256,
        "sil_batch_size": 64,
        "sil_samples_per_episode": 8,
        "critic_warmup_updates": 20,
        "preset": "xmult",
        "python": "/usr/bin/python3",
        "dry_run": True,
        "keep_going": False,
        "shared_args": ["--", "--rollout-length", "64"],
    }
    values.update(overrides)
    return Namespace(**values)


def test_manifest_builds_three_way_matched_commands(tmp_path: Path) -> None:
    checkpoint = tmp_path / "source checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    repo_root = Path(__file__).resolve().parents[1]

    manifest = create_manifest(_run_args(tmp_path, checkpoint), repo_root)

    assert len(manifest["cells"]) == 12  # 2 critics x 3 treatments x 2 seeds
    assert manifest["seeds"] == [0, 7]
    assert manifest["source_checkpoint_sha256"] == (
        "47320987f9a49d5b00119b960f247a956773f57543982b8bfcb6da5bb3afd9ef"
    )
    assert manifest["advantage_clip_sigma"] == 0.0
    assert manifest["sil_buffer_episodes"] == 256
    assert manifest["sil_batch_size"] == 64
    assert manifest["sil_samples_per_episode"] == 8
    assert manifest["preset_train_args"] == list(XMULT_BASE_ARGS)
    arm_ids = {arm.arm_id for arm in ARM_SPECS}
    assert arm_ids == {
        "scalar_no_sil", "scalar_advantage_sil", "scalar_winning_bc",
        "hl_gauss_no_sil", "hl_gauss_advantage_sil", "hl_gauss_winning_bc",
    }

    for cell in manifest["cells"]:
        command = cell["command"]
        assert command[:3] == ["/usr/bin/python3", str(repo_root / "train.py"), "ppo"]
        assert command[3:5] == ["--seed", str(cell["seed"])]
        assert command[-2:] == ["--rollout-length", "64"]
        assert "--pretrained" in command
        assert "--reinit-value-head" in command
        assert command[command.index("--critic-warmup-updates") + 1] == "20"
        assert command[command.index("--advantage-clip-sigma") + 1] == "0.0"
        # Shared replay constants are identical across arms.
        assert command[command.index("--sil-buffer-episodes") + 1] == "256"
        assert command[command.index("--sil-batch-size") + 1] == "64"
        assert command[command.index("--sil-samples-per-episode") + 1] == "8"

        if cell["critic"] == "hl_gauss":
            assert "--hl-gauss" in command
        else:
            assert "--hl-gauss" not in command

        if cell["sil_treatment"] == "no_sil":
            assert command[command.index("--sil-coeff") + 1] == "0"
            assert "--sil-objective" not in command
        elif cell["sil_treatment"] == "advantage_sil":
            assert command[command.index("--sil-coeff") + 1] == "0.01"
            assert command[command.index("--sil-objective") + 1] == "advantage"
            assert command[command.index("--sil-gate-open-percentile") + 1] == "80"
            assert command[command.index("--sil-gate-saturation-percentile") + 1] == "95"
            assert command[command.index("--sil-advantage-floor") + 1] == "0.25"
            assert command[command.index("--sil-coeff-final") + 1] == "0"
        else:  # winning_bc
            assert command[command.index("--sil-coeff") + 1] == "0.01"
            assert command[command.index("--sil-objective") + 1] == "winning_bc"


def test_seed_parser_accepts_comma_separated_values() -> None:
    assert _parse_seeds("0, 1,42") == [0, 1, 42]
    with pytest.raises(Exception, match="duplicates"):
        _parse_seeds("1,1")


@pytest.mark.parametrize(
    "token",
    [
        "--hl-gauss",
        "--sil-coeff=0.9",
        "--log-dir",
        "--advantage-clip-sigma",
        "--reinit-value-head",
        "--seed",
        "--sil-objective",
        "--sil-buffer-episodes",
        "--sil-samples-per-episode",
        "--sil-logical-minibatches-per-update",
        "--sil-gate-open-percentile",
    ],
)
def test_shared_args_cannot_override_treatments_or_output_paths(token: str) -> None:
    with pytest.raises(ValueError, match="controlled by the ablation runner"):
        _normalized_shared_args(["--", token])


def test_paired_effects_are_three_way_within_a_critic() -> None:
    def arm(value: float) -> dict:
        return {"metrics": {"eval/win_rate": {"tail_mean": value}}}

    seed_arms = {
        "scalar_no_sil": arm(0.40),
        "scalar_advantage_sil": arm(0.46),
        "scalar_winning_bc": arm(0.43),
        "hl_gauss_no_sil": arm(0.50),
        "hl_gauss_advantage_sil": arm(0.55),
        "hl_gauss_winning_bc": arm(0.52),
    }
    effects = _paired_effects(seed_arms, "eval/win_rate", "tail_mean", "scalar")
    assert effects is not None
    assert effects["advantage_sil_minus_no_sil"] == pytest.approx(0.06)
    assert effects["winning_bc_minus_no_sil"] == pytest.approx(0.03)
    assert effects["advantage_sil_minus_winning_bc"] == pytest.approx(0.03)

    effects_hl = _paired_effects(seed_arms, "eval/win_rate", "tail_mean", "hl_gauss")
    assert effects_hl["advantage_sil_minus_no_sil"] == pytest.approx(0.05)


def test_paired_effects_none_when_an_arm_is_missing() -> None:
    seed_arms = {
        "scalar_no_sil": {"metrics": {"eval/win_rate": {"tail_mean": 0.4}}},
        "scalar_advantage_sil": {"metrics": {"eval/win_rate": {"tail_mean": 0.5}}},
    }
    assert _paired_effects(seed_arms, "eval/win_rate", "tail_mean", "scalar") is None


def test_continue_execution_skips_nonplanned_jobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def fake_run(
        command: list[str], *, cwd: Path, log_path: Path, prefix: str
    ) -> int:
        calls.append(prefix)
        return 0

    monkeypatch.setattr(ABLATION, "_run_and_tee", fake_run)
    manifest = {
        "output_dir": str(tmp_path),
        "repo_root": str(ROOT),
        "cells": [
            {"id": "done", "status": "completed"},
            {
                "id": "next",
                "status": "planned",
                "command": ["python", "train.py"],
                "command_shell": "python train.py",
                "train_log": str(tmp_path / "next.log"),
            },
        ],
    }

    assert execute_manifest(manifest) == 0
    assert calls == ["next"]
    assert manifest["cells"][0]["status"] == "completed"
    assert manifest["cells"][1]["status"] == "completed"


def test_summary_groups_arms_and_pairs_effects_by_seed() -> None:
    cells = []
    for seed in (0, 7):
        for arm in ARM_SPECS:
            cell_id = f"{arm.arm_id}_seed_{seed}"
            cells.append(
                {
                    "id": cell_id,
                    "arm_id": arm.arm_id,
                    "seed": seed,
                    "critic": arm.critic,
                    "sil_treatment": arm.sil_treatment,
                    "sil_enabled": arm.sil_treatment != "no_sil",
                    "status": "completed",
                    "log_dir": f"/tmp/{cell_id}",
                }
            )
    manifest = {"output_dir": "/tmp/ablation", "seeds": [0, 7], "cells": cells}
    base_values = {
        "scalar_no_sil": 0.40,
        "scalar_advantage_sil": 0.46,
        "scalar_winning_bc": 0.43,
        "hl_gauss_no_sil": 0.50,
        "hl_gauss_advantage_sil": 0.55,
        "hl_gauss_winning_bc": 0.52,
    }

    def reader(log_dir: Path, _tags) -> dict[str, list[tuple[int, float]]]:
        cell_id = log_dir.name
        seed = int(cell_id.rsplit("_", 1)[1])
        arm_id = cell_id.rsplit("_seed_", 1)[0]
        offset = seed / 100.0
        result = {
            "eval/win_rate": [(1, base_values[arm_id]), (2, base_values[arm_id] + offset)],
            "rollout/win_rate": [(1, 0.2), (2, 0.25)],
            "rollout/ep_reward_mean": [(1, 1.0), (2, 1.1)],
        }
        if arm_id.endswith("advantage_sil"):
            result["sil/gate_mean"] = [(1, 0.3)]
            result["sil/ppo_actor_grad_norm_ratio"] = [(1, 0.01)]
        return result

    summary = build_summary(manifest, tail=1, metric_reader=reader)

    scalar_adv = summary["arms"]["scalar_advantage_sil"]
    assert scalar_adv["metrics"]["sil/gate_mean"]["tail_mean"]["mean"] == pytest.approx(0.3)
    assert scalar_adv["metrics"]["eval/win_rate"]["last"]["by_seed"] == {"0": 0.46, "7": 0.53}
    paired = summary["paired_effects"]["eval/win_rate"]["tail_mean"]
    assert set(paired["per_seed"]) == {"scalar/seed_0", "scalar/seed_7", "hl_gauss/seed_0", "hl_gauss/seed_7"}
    assert paired["aggregate"]["advantage_sil_minus_no_sil"]["mean"] == pytest.approx(
        (0.06 + 0.06 + 0.05 + 0.05) / 4
    )
    # Episode-reward effects are reported too (reward-farming check).
    assert "rollout/ep_reward_mean" in summary["paired_effects"]
