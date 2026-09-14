import pytest

from tools.benchmark_outcome_teacher import compare_teacher


def test_teacher_pairing_retains_physical_counts_and_requires_every_sampling_repeat():
    teacher = [dict(seed=seed, sampling_seed=None, won=False, max_ante=1) for seed in (10, 20)]
    candidates = {seed: [dict(seed=game, sampling_seed=repeat, won=repeat < 2, max_ante=8)
                         for game in (10, 20) for repeat in (0, 1, 2)] for seed in (0, 1)}
    result = compare_teacher(candidates, teacher)
    assert result["candidate"]["games"] == 12
    assert result["teacher"]["games"] == 2
    assert result["paired"]["physical_candidate_games"] == 12
    assert result["paired"]["physical_teacher_games"] == 2
    assert result["paired"]["game_seeds"] == 2
    assert result["paired"]["sampled_or_greedy_delta"] == pytest.approx(2 / 3)
    assert result["paired"]["paired_ci95"] == pytest.approx([2 / 3, 2 / 3])
    assert "games_per_arm" not in result["paired"]
    with pytest.raises(ValueError, match="duplicate"):
        compare_teacher(candidates, [*teacher, teacher[0]])
    candidates[1].pop()
    with pytest.raises(ValueError, match="missing sampling repeats"):
        compare_teacher(candidates, teacher)
