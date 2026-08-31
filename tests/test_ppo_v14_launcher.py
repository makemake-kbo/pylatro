"""Deployment-launcher checks; filename remains v14 for compatibility."""

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
    spec = importlib.util.spec_from_file_location("pylatro_v8_validator", VALIDATOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


validator = _load_validator_module()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_source(path: Path, *, version: int = validator.TOKENIZER_VERSION) -> str:
    torch.save(
        {
            "tokenizer_version": version,
            "tokenizer_semantics": validator.TOKENIZER_SEMANTICS,
            "state_dict": {},
        },
        path,
    )
    return _sha256(path)


def _launcher_env(tmp_path: Path, source: Path, source_sha256: str) -> dict[str, str]:
    return {
        **os.environ,
        "PYLATRO_WORKSPACE": str(tmp_path / "workspace"),
        "PYLATRO_REPO_DIR": str(REPO_ROOT),
        "PYLATRO_PYTHON": sys.executable,
        "PYLATRO_RUN_NAME": "ppo_v8_launcher_test",
        "PYLATRO_SOURCE_CHECKPOINT": str(source),
        "PYLATRO_SOURCE_SHA256": source_sha256,
        "PYLATRO_VALIDATE_ONLY": "1",
    }


def _valid_resume(
    path: Path, *, run_uuid: str, source_sha256: str, recipe_id: str
) -> None:
    torch.save(
        {
            "checkpoint_format": "ppo_full",
            "tokenizer_version": validator.TOKENIZER_VERSION,
            "tokenizer_semantics": validator.TOKENIZER_SEMANTICS,
            "ppo_config_fields": dict(validator.V8_PPO_CONFIG),
            "ppo_run_provenance": {
                "run_uuid": run_uuid,
                "source_sha256": source_sha256,
                "recipe_id": recipe_id,
            },
        },
        path,
    )


def test_launcher_recipe_uses_only_v8_critic_controls() -> None:
    launcher = LAUNCHER_PATH.read_text()
    assert validator.V8_PPO_CONFIG["lr"] == pytest.approx(3e-6)
    assert validator.V8_PPO_CONFIG["mini_batch_size"] == 320
    assert validator.V8_PPO_CONFIG["micro_batch_size"] == 160
    assert "--outcome-loss-coeff 0.10" in launcher
    assert "--critic-warmup" not in launcher
    assert "--actor-ramp" not in launcher
    assert "--hl-gauss" not in launcher
    assert "--reinit-value-head" not in launcher
    assert 'recipe_id="pylatro-v8-conditional-survival-v1"' in launcher


def test_source_validator_accepts_v8_and_rejects_v7(tmp_path: Path) -> None:
    source = tmp_path / "v8.pt"
    source_hash = _write_source(source)
    validator.validate_source(source, source_hash)
    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        validator.validate_source(source, "0" * 64)

    old = tmp_path / "v7.pt"
    old_hash = _write_source(old, version=7)
    with pytest.raises(RuntimeError, match="fresh supervised training"):
        validator.validate_source(old, old_hash)


def test_resume_validator_requires_v8_recipe_and_no_transition_state(tmp_path: Path) -> None:
    resume = tmp_path / "resume.pt"
    run_uuid = str(uuid.uuid4())
    source_hash = "a" * 64
    recipe_id = "pylatro-v8-conditional-survival-v1"
    _valid_resume(
        resume,
        run_uuid=run_uuid,
        source_sha256=source_hash,
        recipe_id=recipe_id,
    )
    validator.validate_resume(
        resume,
        run_uuid=run_uuid,
        source_sha256=source_hash,
        recipe_id=recipe_id,
    )

    payload = torch.load(resume, map_location="cpu", weights_only=False)
    payload["ppo_transition_state"] = {"version": 1}
    torch.save(payload, resume)
    with pytest.raises(RuntimeError, match="legacy transition state"):
        validator.validate_resume(
            resume,
            run_uuid=run_uuid,
            source_sha256=source_hash,
            recipe_id=recipe_id,
        )


def test_launcher_creates_v8_owner_marker_after_source_validation(tmp_path: Path) -> None:
    source = tmp_path / "source.pt"
    source_hash = _write_source(source)
    env = _launcher_env(tmp_path, source, source_hash)
    result = subprocess.run(
        ["bash", str(LAUNCHER_PATH)],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    marker = (
        tmp_path
        / "workspace"
        / "checkpoints"
        / env["PYLATRO_RUN_NAME"]
        / ".pylatro-v8-run"
    )
    payload = json.loads(marker.read_text())
    assert payload["tokenizer_version"] == validator.TOKENIZER_VERSION
    assert payload["source_sha256"] == source_hash
    assert str(uuid.UUID(payload["run_uuid"])) == payload["run_uuid"]


def test_launcher_rejects_invalid_win_ante(tmp_path: Path) -> None:
    source = tmp_path / "source.pt"
    source_hash = _write_source(source)
    env = _launcher_env(tmp_path, source, source_hash)
    env["PYLATRO_WIN_ANTE"] = "five"
    result = subprocess.run(
        ["bash", str(LAUNCHER_PATH)],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "integer from 1 through 8" in result.stdout + result.stderr


def test_launcher_rejects_unowned_latest_checkpoint(tmp_path: Path) -> None:
    source = tmp_path / "source.pt"
    source_hash = _write_source(source)
    env = _launcher_env(tmp_path, source, source_hash)
    checkpoint_dir = (
        tmp_path / "workspace" / "checkpoints" / env["PYLATRO_RUN_NAME"]
    )
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
