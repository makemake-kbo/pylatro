import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Target tags
SCALAR_TAGS = [
    'rollout/ep_reward_mean', 'rollout/ep_length_mean',
    'rollout/episode_reward_mean', 'rollout/episode_length_mean',
    'rollout/win_rate', 'rollout/final_ante_mean', 'eval/win_rate',
    'recent_100/win_rate', 'recent_100/final_ante_mean',
    'reward/terminal_mean', 'reward/score_progress_mean', 'reward/pressure_progress_mean',
    'reward/blind_clear_mean', 'reward/ante_bonus_mean', 'reward/idle_penalty_mean',
    'reward/consumable_targeted_use_mean',
    'ppo/entropy_normalized', 'ppo/entropy_coeff', 'ppo/approx_kl', 'ppo/clip_fraction',
    'ppo/value_loss', 'ppo/policy_loss', 'ppo/survival_loss',
    'ppo/win_probability_loss', 'ppo/explained_variance', 'ppo/actual_lr',
    'ppo/minibatch_fraction', 'ppo/critic_warmup_active',
    'debug/chosen_action_prob_mean', 'debug/returns_mean', 'debug/returns_std',
    'actions/use_consumable_hand_subset_fraction', 'actions/use_consumable_joker_fraction',
    'actions/use_consumable_no_target_fraction',
    'actions/shop_reroll_fraction', 'actions/shop_sell_joker_fraction',
    'actions/shop_sell_consumable_fraction', 'actions/play_subset_fraction',
    'actions/discard_subset_fraction',
    'build/estimated_score_mean', 'build/required_score_mean', 'build/readiness_mean',
    'build/score_gain_ratio_mean', 'build/modeled_fraction_mean',
    'potential/post_total_mean', 'potential/delta_total_mean',
    'joker/churn_per_episode', 'joker/hologram_scaling_count_per_episode',
    'joker/hologram_x_mult_delta_mean', 'joker/hologram_build_score_delta_mean',
    'counterfactual/calls', 'counterfactual/failure_fraction',
    'counterfactual/representative_vs_realized_abs_log_ratio_gap_mean',
    'strategy/hand_plan/reliability_mean', 'strategy/hand_plan/readiness_mean',
    'strategy/seals/purple_tarots_generated_per_1k_steps',
    'strategy/seals/blue_planets_generated_per_1k_steps',
    'strategy/potential/economy_mean', 'strategy/potential/tarot_option_value_mean',
    'strategy/potential/planet_option_value_mean',
    'strategy/potential/seal_value_mean', 'strategy/potential/joker_search_option_mean',
    'strategy/potential/standard_pack_search_option_mean',
    'strategy/risk/clear_probability_mean',
    'strategy/risk/immediate_death_probability_mean',
    'shop/joker_offers_per_1k_steps', 'shop/joker_buys_per_1k_steps',
    'shop/joker_sells_per_1k_steps', 'shop/unsafe_leave_fraction',
    'shop/unsafe_can_reroll_fraction', 'shop/missed_confident_upgrade_fraction',
    'shop/full_weak_leave_fraction', 'shop/best_confident_upgrade_delta_mean',
    'joker/replacements_per_episode', 'critic/shop_survival_brier',
    'critic/shop_survival_prediction_mean', 'critic/shop_survival_outcome_mean',
    'rollout/reward_components/reward_danger_reroll_bonus',
    'rollout/reward_components/reward_joker_upgrade_bonus',
]

SCALAR_PREFIXES = [
    'shop/offered_joker/', 'shop/bought_joker/', 'shop/sold_joker/',
    'joker/marginal_ratio/', 'joker/modeled_fraction/',
    'terminal/',
    'strategy/hand_plan/', 'strategy/seals/claims/',
]

run_path = sys.argv[1]
ea = EventAccumulator(run_path, size_guidance={'scalars': 0})
ea.Reload()

all_tags = sorted(ea.Tags()['scalars'])
print(f"=== ALL TAGS IN {run_path.split('/')[-1]} ({len(all_tags)} total) ===")
for i, tag in enumerate(all_tags[:80]):
    print(f"{i+1:3d}. {tag}")

print("\n=== DATA FOR TARGET TAGS ===")
dynamic_tags = [
    tag
    for tag in all_tags
    if any(tag.startswith(prefix) for prefix in SCALAR_PREFIXES) and tag not in SCALAR_TAGS
]
for tag in [*SCALAR_TAGS, *dynamic_tags]:
    if tag in all_tags:
        events = ea.Scalars(tag)
        print(f"\n{tag}:")
        for event in events:
            print(f"  step={event.step} val={event.value:.6g}")
    else:
        print(f"\n{tag}: [NOT FOUND]")
