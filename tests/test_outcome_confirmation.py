import json

import pytest

from pylatro_agent.transfer import sha256
from tools.report_outcome_ppo import frozen_checkpoints
from tools.run_outcome_confirmation import confirmation_plan


def test_frozen_confirmation_includes_both_trained_controls_and_rejects_changed_inputs(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    development = tmp_path / "development.json"
    development.write_text("{}")
    paths = {}
    for key in ("control", "seed0", "seed1", "matched_control_seed0", "matched_control_seed1"):
        checkpoint = tmp_path / f"{key}.pt"
        checkpoint.write_text(key)
        paths[key] = str(checkpoint)
    hashes = {key: sha256(path) for key, path in paths.items()}
    primary = {"control", "seed0", "seed1"}
    selection = dict(
        recipe="gae100_complete", updates=30, confirmation_device="mps",
        study_manifest_sha256=sha256(manifest),
        development_comparison=str(development), development_comparison_sha256=sha256(development),
        checkpoints={key: path for key, path in paths.items() if key in primary},
        checkpoint_hashes={key: digest for key, digest in hashes.items() if key in primary},
        matched_training_control=dict(
            recipe="gae100", updates=30,
            development_comparison=str(development), development_comparison_sha256=sha256(development),
            checkpoints={key: path for key, path in paths.items() if key not in primary},
            checkpoint_hashes={key: digest for key, digest in hashes.items() if key not in primary},
        ),
    )
    (tmp_path / "candidate_selection.json").write_text(json.dumps(selection))
    assert frozen_checkpoints(selection) == (paths, hashes)
    _, jobs = confirmation_plan(tmp_path)
    assert len(jobs) == 5
    assert {job["name"] for job in jobs} == {
        "baseline_confirmation400", "gae100_complete_seed0_u30_confirmation400",
        "gae100_complete_seed1_u30_confirmation400", "gae100_seed0_u30_confirmation400",
        "gae100_seed1_u30_confirmation400",
    }
    for job in jobs:
        command = job["command"]
        assert command[command.index("--games") + 1] == "400"
        assert command[command.index("--device") + 1] == "mps"
    selection["matched_training_control"]["updates"] = 20
    with pytest.raises(ValueError, match="equal training"):
        frozen_checkpoints(selection)
    selection["matched_training_control"]["updates"] = 30
    (tmp_path / "matched_control_seed1.pt").write_text("changed")
    with pytest.raises(ValueError, match="checkpoint changed"):
        confirmation_plan(tmp_path)
    (tmp_path / "matched_control_seed1.pt").write_text("matched_control_seed1")
    development.write_text('{"changed": true}')
    with pytest.raises(ValueError, match="development comparison changed"):
        confirmation_plan(tmp_path)
