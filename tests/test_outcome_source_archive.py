import json

import pytest

from pylatro_agent.transfer import sha256
from tools.verify_outcome_ppo import reference_source_gate


def test_archived_source_cannot_bypass_a_live_or_changed_study(tmp_path):
    archive = tmp_path / "reference_source"
    source = archive / "src/example.py"
    source.parent.mkdir(parents=True)
    source.write_text("fixed reference")
    (archive / "manifest.json").write_text(json.dumps({"source_hashes": {"src/example.py": sha256(source)}}))
    ledger = dict(jobs=[dict(status="complete", elapsed_seconds=100)], elapsed_seconds=100)
    prior = dict(passed=True, final=True, source_verification=dict(mode="current_workspace"), compute=ledger)
    (tmp_path / "verification_final.json").write_text(json.dumps(prior))
    (tmp_path / "compute.json").write_text(json.dumps(ledger))
    with pytest.raises(ValueError, match="completed final"):
        reference_source_gate(tmp_path, archive, final=False)
    result, saved = reference_source_gate(tmp_path, archive, final=True)
    assert result["mode"] == "archived_reference_after_maintenance" and saved == prior
    source.write_text("changed reference")
    with pytest.raises(ValueError, match="reference source changed"):
        reference_source_gate(tmp_path, archive, final=True)
    source.write_text("fixed reference")
    ledger["jobs"].append(dict(status="running"))
    (tmp_path / "compute.json").write_text(json.dumps(ledger))
    with pytest.raises(ValueError, match="Accelerator activity changed"):
        reference_source_gate(tmp_path, archive, final=True)
