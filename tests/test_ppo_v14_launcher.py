from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_PATH = REPO_ROOT / "scripts" / "validate_ppo_run_checkpoint.py"
LAUNCHER_PATH = REPO_ROOT / "scripts" / "run_ppo_v14_safe_tarot_seal_strategy.sh"


def _load_validator_module():
    spec = importlib.util.spec_from_file_location("pylatro_v14_validator", VALIDATOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


validator = _load_validator_module()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_source(path: Path, **overrides) -> str:
    payload = {
        "tokenizer_version": 6,
        "reward_model_version": 12,
        "update_count": 280,
        "best_eval_update": 280,
    }
    payload.update(overrides)
    torch.save(payload, path)
    return _sha256(path)


def _launcher_env(tmp_path: Path, source: Path, source_sha256: str) -> dict[str, str]:
    return {
        **os.environ,
        "PYLATRO_WORKSPACE": str(tmp_path / "workspace"),
        "PYLATRO_REPO_DIR": str(REPO_ROOT),
        "PYLATRO_PYTHON": sys.executable,
        "PYLATRO_RUN_NAME": "ppo_strategy_v14_launcher_test",
        "PYLATRO_SOURCE_CHECKPOINT": str(source),
        "PYLATRO_SOURCE_SHA256": source_sha256,
        "PYLATRO_VALIDATE_ONLY": "1",
    }


def test_v14_launcher_and_validator_pin_distinct_phase_learning_rates() -> None:
    launcher = LAUNCHER_PATH.read_text()
    assert validator.V14_TRANSITION_CONFIG["lr"] == pytest.approx(3e-6)
    assert validator.V14_TRANSITION_CONFIG["critic_warmup_lr"] == pytest.approx(1e-5)
    assert "--lr 3e-6" in launcher
    assert "--critic-warmup-lr 1e-5" in launcher
    assert "--batch 320" in launcher
    assert "--batch 352" not in launcher
    assert "criticlr1e5_batch320" in launcher
    assert 'recipe_id="pylatro-v14-safe-v3"' in launcher


def test_source_validator_checks_hash_and_exact_metadata(tmp_path: Path) -> None:
    source = tmp_path / "source.pt"
    source_sha256 = _write_source(source)
    validator.validate_source(source, source_sha256)

    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        validator.validate_source(source, "0" * 64)

    renamed_v13 = tmp_path / "renamed_v13.pt"
    renamed_sha256 = _write_source(renamed_v13, update_count=50, best_eval_update=40)
    with pytest.raises(RuntimeError, match="metadata mismatch"):
        validator.validate_source(renamed_v13, renamed_sha256)


def test_resume_validator_requires_v14_transition_recipe(tmp_path: Path) -> None:
    resume = tmp_path / "v14_resume.pt"
    run_uuid = str(uuid.uuid4())
    source_sha256 = "a" * 64
    recipe_id = "pylatro-v14-safe-v3"
    torch.save(
        {
            "checkpoint_format": "ppo_full",
            "tokenizer_version": 7,
            "ppo_transition_state": {"version": 1, "warmup_complete": False},
            "ppo_config_fields": dict(validator.V14_TRANSITION_CONFIG),
            "ppo_active_lr": 1e-5,
            "optimizer_state_dict": {"param_groups": [{"lr": 1e-5}]},
            "ppo_run_provenance": {
                "run_uuid": run_uuid,
                "source_sha256": source_sha256,
                "recipe_id": recipe_id,
            },
        },
        resume,
    )
    validator.validate_resume(
        resume,
        run_uuid=run_uuid,
        source_sha256=source_sha256,
        recipe_id=recipe_id,
    )

    payload = torch.load(resume, map_location="cpu", weights_only=False)
    payload["ppo_config_fields"]["actor_ramp_updates"] = 0
    torch.save(payload, resume)
    with pytest.raises(RuntimeError, match="transition recipe mismatch"):
        validator.validate_resume(
            resume,
            run_uuid=run_uuid,
            source_sha256=source_sha256,
            recipe_id=recipe_id,
        )

    payload["ppo_config_fields"] = dict(validator.V14_TRANSITION_CONFIG)
    payload["ppo_config_fields"]["critic_warmup_lr"] = 3e-6
    torch.save(payload, resume)
    with pytest.raises(RuntimeError, match="critic_warmup_lr"):
        validator.validate_resume(
            resume,
            run_uuid=run_uuid,
            source_sha256=source_sha256,
            recipe_id=recipe_id,
        )

    payload["ppo_config_fields"] = dict(validator.V14_TRANSITION_CONFIG)
    payload["ppo_active_lr"] = 3e-6
    torch.save(payload, resume)
    with pytest.raises(RuntimeError, match="active optimizer LR"):
        validator.validate_resume(
            resume,
            run_uuid=run_uuid,
            source_sha256=source_sha256,
            recipe_id=recipe_id,
        )


def test_launcher_marks_identity_only_after_validation_and_rejects_unowned_resume(tmp_path: Path) -> None:
    source = tmp_path / "source.pt"
    source_sha256 = _write_source(source)
    env = _launcher_env(tmp_path, source, source_sha256)

    first = subprocess.run(
        ["bash", str(LAUNCHER_PATH)],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert first.returncode == 0, first.stdout + first.stderr
    checkpoint_dir = tmp_path / "workspace" / "checkpoints" / env["PYLATRO_RUN_NAME"]
    marker = checkpoint_dir / ".pylatro-v14-safe-run"
    marker_payload = json.loads(marker.read_text())
    assert marker_payload["run_name"] == env["PYLATRO_RUN_NAME"]
    assert marker_payload["source_sha256"] == source_sha256
    assert marker_payload["source_metadata"] == validator.SOURCE_METADATA
    assert str(uuid.UUID(marker_payload["run_uuid"])) == marker_payload["run_uuid"]

    # A renamed/copy-in v13 checkpoint lacks the transition marker and is
    # rejected even inside a directory that otherwise has a valid owner file.
    torch.save(
        {
            "checkpoint_format": "ppo_full",
            "tokenizer_version": 7,
            "ppo_config_fields": dict(validator.V14_TRANSITION_CONFIG),
        },
        checkpoint_dir / "ppo_latest.pt",
    )
    copied_v13 = subprocess.run(
        ["bash", str(LAUNCHER_PATH)],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert copied_v13.returncode == 2
    assert "transition-state marker" in copied_v13.stdout + copied_v13.stderr


def test_launcher_does_not_mark_invalid_source(tmp_path: Path) -> None:
    source = tmp_path / "renamed_regressed_checkpoint.pt"
    source_sha256 = _write_source(source, update_count=50, best_eval_update=40)
    env = _launcher_env(tmp_path, source, source_sha256)

    result = subprocess.run(
        ["bash", str(LAUNCHER_PATH)],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    checkpoint_dir = tmp_path / "workspace" / "checkpoints" / env["PYLATRO_RUN_NAME"]
    assert not (checkpoint_dir / ".pylatro-v14-safe-run").exists()
    assert "metadata mismatch" in result.stdout + result.stderr


def test_launcher_rejects_mismatched_ownership_identity(tmp_path: Path) -> None:
    source = tmp_path / "source.pt"
    source_sha256 = _write_source(source)
    env = _launcher_env(tmp_path, source, source_sha256)
    first = subprocess.run(
        ["bash", str(LAUNCHER_PATH)],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert first.returncode == 0

    marker = tmp_path / "workspace" / "checkpoints" / env["PYLATRO_RUN_NAME"] / ".pylatro-v14-safe-run"
    marker_payload = json.loads(marker.read_text())
    marker_payload["run_name"] = "some-other-run"
    marker.write_text(json.dumps(marker_payload))
    second = subprocess.run(
        ["bash", str(LAUNCHER_PATH)],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert second.returncode == 2
    assert "ownership marker mismatch" in second.stdout + second.stderr


def test_launcher_rejects_foreign_v14_checkpoint_uuid(tmp_path: Path) -> None:
    source = tmp_path / "source.pt"
    source_sha256 = _write_source(source)
    env = _launcher_env(tmp_path, source, source_sha256)
    first = subprocess.run(
        ["bash", str(LAUNCHER_PATH)],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert first.returncode == 0
    checkpoint_dir = tmp_path / "workspace" / "checkpoints" / env["PYLATRO_RUN_NAME"]
    torch.save(
        {
            "checkpoint_format": "ppo_full",
            "tokenizer_version": 7,
            "ppo_transition_state": {"version": 1, "warmup_complete": False},
            "ppo_config_fields": dict(validator.V14_TRANSITION_CONFIG),
            "ppo_active_lr": 1e-5,
            "optimizer_state_dict": {"param_groups": [{"lr": 1e-5}]},
            "ppo_run_provenance": {
                "run_uuid": str(uuid.uuid4()),
                "source_sha256": source_sha256,
                "recipe_id": "pylatro-v14-safe-v3",
            },
        },
        checkpoint_dir / "ppo_latest.pt",
    )

    foreign = subprocess.run(
        ["bash", str(LAUNCHER_PATH)],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert foreign.returncode == 2
    assert "run provenance mismatch" in foreign.stdout + foreign.stderr


def test_launcher_rejects_unowned_latest_checkpoint(tmp_path: Path) -> None:
    source = tmp_path / "source.pt"
    source_sha256 = _write_source(source)
    env = _launcher_env(tmp_path, source, source_sha256)
    checkpoint_dir = tmp_path / "workspace" / "checkpoints" / env["PYLATRO_RUN_NAME"]
    checkpoint_dir.mkdir(parents=True)
    torch.save({"checkpoint_format": "ppo_full"}, checkpoint_dir / "ppo_latest.pt")

    result = subprocess.run(
        ["bash", str(LAUNCHER_PATH)],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "unmarked non-empty checkpoint directory" in result.stdout + result.stderr
