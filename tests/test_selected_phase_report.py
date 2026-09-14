import json

import pytest

from pylatro_agent.phase_oracle import ORACLE_ARMS
from pylatro_agent.transfer import paired_statistics, sha256
from tools.report_learning_bottleneck import phase_oracle_results


def test_explicit_phase_panel_preserves_previous_analysis_and_rejects_missing_games(tmp_path):
    folder = tmp_path / "phase_oracle_cpu"
    (folder / "games").mkdir(parents=True)
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_text("fixture")
    training = tmp_path / "training.json"
    training.write_text("[1000]")
    seeds = [100, 101]
    manifest = dict(status="complete", device="cpu", accelerator_seconds=0, seeds=seeds,
                    never_train_on_seeds=True, checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
                    seed_separation=[dict(path=str(training))])
    (folder / "manifest.json").write_text(json.dumps(manifest))
    groups = {arm: [] for arm in ORACLE_ARMS}
    for arm in ORACLE_ARMS:
        for seed in seeds:
            for repeat in ((None,) if arm == "teacher_all" else (0, 1, 2)):
                won = arm != "student" or seed == 100
                row = dict(arm=arm, seed=seed, sampling_seed=repeat, won=won, steps=3,
                           max_ante=9 if won else 1, final_ante=9 if won else 1,
                           cleared_antes=list(range(1, 9)) if won else [], stalled=False, censored=False)
                groups[arm].append(row)
                (folder / "games" / f"{arm}_{seed}_sample{repeat}.json").write_text(json.dumps(row))
    comparisons = {arm: paired_statistics(groups[arm], groups["student"]) for arm in ORACLE_ARMS[1:-1]}
    teachers = [dict(row, sampling_seed=repeat) for row in groups["teacher_all"] for repeat in (0, 1, 2)]
    comparisons["teacher_all"] = paired_statistics(teachers, groups["student"])
    (folder / "results.json").write_text(json.dumps(dict(comparisons=comparisons)))
    analysis = folder / "analysis_canonical.json"
    analysis.write_text("preserved historical analysis\n")
    with pytest.raises(ValueError, match="reservation changed"):
        phase_oracle_results(tmp_path, write_analysis=False)
    result = phase_oracle_results(tmp_path, expected_seeds=seeds, write_analysis=False)
    assert result["summaries"]["student"]["games"] == 6
    assert result["summaries"]["teacher_all"]["games"] == 2
    assert analysis.read_text() == "preserved historical analysis\n"
    (folder / "games/hand_100_sample0.json").unlink()
    with pytest.raises(ValueError, match="Incomplete or duplicate"):
        phase_oracle_results(tmp_path, expected_seeds=seeds, write_analysis=False)
