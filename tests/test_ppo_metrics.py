from types import SimpleNamespace

import numpy as np
import pytest

from pylatro_agent.training.ppo_metrics import (
    EpisodeHistory,
    _record_action_diagnostics,
    _RolloutMetrics,
    write_rollout_metrics,
)


@pytest.mark.parametrize("done", [False, True])
def test_reported_diagnostics_respect_terminal_info_masks_and_zero_values(done):
    metrics = _RolloutMetrics(
        step_rewards=[0.0] * 1000,
        chosen_action_probs=[0.8],
        max_action_type_probs=[0.9],
        done_flags=[1.0],
        progress_flags=[1.0],
        completed_episode_rewards=[1.0],
        completed_episode_lengths=[1000],
        completed_episode_wins=[0.0],
        completed_episode_stalls=[0.0],
        completed_episode_antes=[1],
    )
    infos = {
        key: np.array([value])
        for key, value in {
            "shop_leave_observed": True,
            "shop_unsafe_leave": False,
            "shop_best_confident_upgrade_delta": 0.4,
            "hand_play_observed": True,
            "hand_play_in_candidates": True,
            "hand_play_chosen_hand": "Pair",
            "joker_roster_changed": True,
            "joker_churn_count": 2,
            "joker_replacement_event": True,
            "strategic_planet_uses": 1,
            "strategic_planet_pack_auto_uses": 1,
            "planet_claim_plan_supported": True,
            "planet_claim_active_plan_match": True,
            "consumable_planet_active_plan_legal": True,
            "build_diagnostics_observed": True,
            "hand_plan_post_type": "Pair",
            "hand_plan_post_reliability": 0.75,
            "potential_post_seal_value": 0.5,
        }.items()
    }
    infos["final_info"] = {
        "shop_unsafe_leave": np.array([True]),
        "shop_best_confident_upgrade_delta": np.array([0.0]),
        "hand_play_chosen_hand": np.array(["Flush"]),
        "_hand_play_chosen_hand": np.array([False]),
    }
    _record_action_diagnostics(metrics, infos, 0, done=done)
    scalars = {}
    writer = SimpleNamespace(add_scalar=lambda tag, value, step: scalars.update({tag: (value, step)}))
    buffer = SimpleNamespace(
        _flat_returns=np.array([1.0, 2.0]),
        _flat_values=np.array([1.0, 2.0]),
        outcome_label_diagnostics=lambda: {},
    )
    write_rollout_metrics(writer, metrics, buffer, EpisodeHistory(), win_ante=8, step=17)

    expected = {
        "shop/unsafe_leave_fraction": float(done),
        "shop/best_confident_upgrade_delta_mean": 0.0 if done else 0.4,
        "rollout/hands_played/pair": 1.0,
        "joker/replacements_per_episode": 1.0,
        "joker/churn_per_episode": 2.0,
        "strategy/planets/matched_use_rate": 1.0,
        "strategy/planets/active_plan_use_rate_given_legal": 0.0,
        "strategy/consumables/planet/used_per_1k_steps": 1.0,
        "strategy/consumables/planet/pack_auto_used_per_1k_steps": 1.0,
        "strategy/hand_plan/pair_share": 1.0,
        "strategy/hand_plan/reliability_mean": 0.75,
        "strategy/potential/seal_value_mean": 0.5,
    }
    for tag, value in expected.items():
        assert scalars[tag] == (value, 17)
    assert "rollout/hands_played/flush" not in scalars
    assert "strategy/hand_plan/readiness_mean" not in scalars
    assert "strategy/risk/shop_death_brier" not in scalars
