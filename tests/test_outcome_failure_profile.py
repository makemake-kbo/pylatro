import pytest

from tools.audit_outcome_failures import failure_profile


def rows():
    data = []
    for seed in (10, 11):
        for arm in ("control", "seed0", "seed1", "teacher"):
            for repeat in ((None,) if arm == "teacher" else (None, 0, 1, 2)):
                won = ((arm == "teacher" and seed == 10) or
                       (arm == "seed0" and (seed == 11 or repeat is None)) or
                       (arm == "seed1" and seed == 11 and repeat == 0))
                data.append(dict(arm=arm, seed=seed, sampling_seed=repeat, won=won,
                                 max_ante=9 if won else 5, stalled=False, censored=False))
    return data


def test_failure_profile_preserves_seed_and_physical_game_denominators():
    result = failure_profile(rows(), [10, 11])
    assert result["selected_sampled"] == dict(games=12, wins=4, losses=8,
                                               highest_ante_reached_on_losses={5: 8}, stalls=0, censored=0)
    assert result["selected_greedy"]["games"] == 4
    assert result["sampled_wins_per_seed_histogram"] == {0: 1, 1: 0, 2: 0, 3: 0, 4: 1, 5: 0, 6: 0}
    assert result["teacher_outcome_by_sampled_wins_per_seed"][True][0] == 1
    assert result["teacher_outcome_by_sampled_wins_per_seed"][False][4] == 1
    for key in ("no_sampled_wins", "teacher_wins_with_no_sampled_wins",
                "any_greedy_win_with_no_sampled_wins"):
        assert result[key] == dict(count=1, seeds=[10])


def test_failure_profile_rejects_duplicate_missing_and_unexpected_games():
    data = rows()
    with pytest.raises(ValueError, match="Duplicate game"):
        failure_profile([*data, data[0]], [10, 11])
    with pytest.raises(ValueError, match="Missing or unexpected"):
        failure_profile(data[1:], [10, 11])
    with pytest.raises(ValueError, match="Missing or unexpected"):
        failure_profile(data, [10, 11, 12])
    with pytest.raises(ValueError, match="distinct"):
        failure_profile(data, [10, 10])
