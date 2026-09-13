import json

import pytest

from pylatro_agent.transfer import sha256
from tools.outcome_resource_plan import cpu_confirmation_plan, training_reserve


def test_resource_override_preserves_the_cap_and_requires_cpu_confirmation_evidence(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(dict(confirmation_reserve_seconds=10800)))
    (tmp_path / "compute.json").write_text(json.dumps(dict(limit_seconds=43200)))
    evidence = tmp_path / "cpu_verified.json"
    evidence.write_text(json.dumps(dict(passed=True)))
    plan = dict(
        version="cpu_confirmation_allocation_v1", study_manifest_sha256=sha256(manifest),
        accelerator_limit_seconds=43200, confirmation_device="cpu", confirmation_games=400,
        sampling_seeds=[0, 1, 2], training_replicates=[0, 1], parallel_modes=4,
        remaining_accelerator_reserve_seconds=300,
        cpu_scheduling_verification=dict(path=str(evidence), sha256=sha256(evidence)),
    )
    path = tmp_path / "resource_allocation_v2.json"
    path.write_text(json.dumps(plan))
    assert training_reserve(tmp_path) == (10800, None)
    reserve, provenance = training_reserve(tmp_path, path)
    assert reserve == 300
    assert provenance == dict(path=str(path), sha256=sha256(path))
    assert cpu_confirmation_plan(tmp_path)[0] == plan
    for key, value in (("accelerator_limit_seconds", 86400), ("confirmation_device", "mps"),
                       ("confirmation_games", 200), ("sampling_seeds", [0]),
                       ("training_replicates", [0]), ("remaining_accelerator_reserve_seconds", 0)):
        path.write_text(json.dumps({**plan, key: value}))
        with pytest.raises(ValueError):
            training_reserve(tmp_path, path)
    path.write_text(json.dumps(plan))
    evidence.write_text(json.dumps(dict(passed=False)))
    with pytest.raises(ValueError, match="verification"):
        training_reserve(tmp_path, path)
    evidence.write_text(json.dumps(dict(passed=True)))
    (tmp_path / "candidate_selection.json").write_text("{}")
    with pytest.raises(ValueError, match="after confirmation"):
        training_reserve(tmp_path, path)
    (tmp_path / "candidate_selection.json").unlink()
    partial = tmp_path / "evaluations/baseline_confirmation400/sample0"
    partial.mkdir(parents=True)
    (partial / "manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="after confirmation"):
        training_reserve(tmp_path, path)
