import pytest
import torch

from pylatro_agent.experiment_budget import accelerator_job
from pylatro_agent.learning_bottleneck import atomic_json, read_json


def test_accelerator_admission_margin_preserves_remaining_limit(tmp_path):
    atomic_json(tmp_path / "compute.json", {"limit_seconds": 100, "jobs": []})
    with pytest.raises(TimeoutError, match="remaining"), accelerator_job(tmp_path, "too_large", 80):
        pytest.fail("Job with insufficient admission margin started")
    assert not read_json(tmp_path / "compute.json")["jobs"]


def test_accelerator_failure_is_charged_and_does_not_leave_running_marker(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.mps, "synchronize", lambda: None)
    atomic_json(tmp_path / "compute.json", {"limit_seconds": 100, "jobs": []})
    with pytest.raises(RuntimeError, match="deliberate"), accelerator_job(tmp_path, "failure", 1):
        raise RuntimeError("deliberate test failure")
    ledger = read_json(tmp_path / "compute.json")
    assert ledger["jobs"][0]["status"] == "failed"
    assert ledger["elapsed_seconds"] > 0
    with accelerator_job(tmp_path, "success", 1):
        pass
    assert read_json(tmp_path / "compute.json")["jobs"][-1]["status"] == "complete"


def test_running_job_requires_explicit_crash_accounting(tmp_path):
    atomic_json(tmp_path / "compute.json", {"limit_seconds": 100, "jobs": [{"status": "running"}]})
    with pytest.raises(RuntimeError, match="existing accelerator"), accelerator_job(tmp_path, "second", 1):
        pass
