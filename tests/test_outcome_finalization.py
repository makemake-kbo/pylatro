import json
import subprocess
import sys

import pytest

from tools import finalize_outcome_study as workflow


def test_finalization_keeps_an_older_best_candidate_and_stops_before_confirmation_on_backend_failure(
    tmp_path, monkeypatch,
):
    def write(relative, data):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))

    write("manifest.json", dict(study="fixture"))
    write("compute.json", dict(jobs=[]))
    write("longer_continuation_selection.json", dict(
        additional_updates=12, target_update=42, recipes=["gae097", "gae097_lr3e5"]))
    write("longer_continuation/manifest.json", dict(status="complete"))
    for recipe, updates, rate in (("gae097_lr3e5", 30, .41), ("gae097", 42, .40),
                                  ("gae097_lr3e5", 42, .39)):
        write(f"development_{recipe}_u{updates}_comparison.json", dict(
            recipe=recipe, updates=updates,
            results=dict(sampled=dict(candidate=dict(win_rate=rate)),
                         greedy=dict(candidate=dict(win_rate=.45))),
            acceptance=dict(greedy_rules_out_five_point_regression=True)))
    monkeypatch.setattr(workflow, "compare_panel", lambda root, recipe, updates, stage:
                        json.loads((root / f"development_{recipe}_u{updates}_comparison.json").read_text()))
    commands = []

    def execute(command, **kwargs):
        commands.append(command)
        if command[2] == "tools/report_outcome_ppo.py":
            assert command[command.index("--recipe") + 1] == "gae097_lr3e5"
            assert command[command.index("--updates") + 1] == "30"
            assert command[command.index("--matched-control-recipe") + 1] == "gae097"
            write("candidate_selection.json", dict(checkpoints=dict(seed0="fixture.pt")))
        if command[2] == "tools/check_subset_backend.py":
            raise subprocess.CalledProcessError(1, command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(workflow.subprocess, "run", execute)
    monkeypatch.setattr(sys, "argv", ["finalize_outcome_study.py", "--root", str(tmp_path)])
    with pytest.raises(subprocess.CalledProcessError):
        workflow.main()
    assert [command[2] for command in commands] == [
        "tools/verify_outcome_ppo.py", "tools/report_outcome_ppo.py", "tools/check_subset_backend.py"]
    saved = json.loads((tmp_path / "finalization/manifest.json").read_text())
    assert saved["status"] == "failed"
    assert saved["chosen"]["updates"] == 30
    assert saved["jobs"][-1]["status"] == "failed"
    assert not (tmp_path / "confirmation_stage").exists()
