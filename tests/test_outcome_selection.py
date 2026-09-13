import pytest

from tools.advance_outcome_study import select_development
from tools.report_outcome_ppo import compare_matched_training
from tools.verify_outcome_ppo import validate_training_rows


def comparison(recipe, sampled, greedy, safe):
    return dict(recipe=recipe,
                results=dict(sampled=dict(candidate=dict(win_rate=sampled)),
                             greedy=dict(candidate=dict(win_rate=greedy))),
                acceptance=dict(greedy_rules_out_five_point_regression=safe))


def test_selection_respects_greedy_guardrail_and_reports_when_none_qualifies():
    unsafe = comparison("higher_sampled", .4, .1, False)
    safe = comparison("guardrail_pass", .35, .4, True)
    assert select_development([unsafe, safe]) == (safe, True)
    also_unsafe = comparison("weaker", .3, .2, False)
    assert select_development([unsafe, also_unsafe]) == (unsafe, False)
    with pytest.raises(ValueError, match="No complete"):
        select_development([])


def test_training_episode_audit_rejects_contamination_and_wrong_outcome_targets():
    row = dict(seed=123, reward=10., won=True, stalled=False, censored=False,
               archive_start=False, deck="b_blue")
    validate_training_rows([row], {190000, 200000})
    with pytest.raises(ValueError, match="Reserved"):
        validate_training_rows([dict(row, seed=200000)], {190000, 200000})
    with pytest.raises(ValueError, match="pure outcome"):
        validate_training_rows([dict(row, won=False)], set())
    with pytest.raises(ValueError, match="Incomplete"):
        validate_training_rows([dict(row, censored=True)], set())


def test_interventions_pair_with_their_own_training_replica():
    def arm(winning_seeds):
        def rows(repeats):
                return [dict(seed=seed, sampling_seed=repeat, won=seed in winning_seeds,
                         final_ante=8 if seed in winning_seeds else 1,
                         max_ante=8 if seed in winning_seeds else 1, stalled=False, censored=False,
                         cleared_antes=list(range(1, 9)) if seed in winning_seeds else [])
                    for seed in range(4) for repeat in repeats]

        return dict(sampled=rows((0, 1, 2)), greedy=rows((None,)))

    controls = {0: arm({0, 1}), 1: arm({2, 3})}
    candidates = {0: arm({0, 1, 2}), 1: arm({1, 2, 3})}
    result = compare_matched_training(candidates, controls)
    assert result["sampled"]["control"]["games"] == 24
    assert result["sampled"]["paired"]["sampled_or_greedy_delta"] == .25
    assert result["sampled"]["paired"]["paired_ci95"] == [0., .5]
    with pytest.raises(ValueError, match="Both matched"):
        compare_matched_training(candidates, {0: controls[0]})
