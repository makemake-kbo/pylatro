import sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Target tags
SCALAR_TAGS = [
    'rollout/ep_reward_mean', 'rollout/ep_length_mean',
    'reward/terminal_mean', 'reward/score_progress_mean', 'reward/pressure_progress_mean',
    'reward/blind_clear_mean', 'reward/ante_bonus_mean', 'reward/idle_penalty_mean', 'reward/consumable_commit_mean',
    'ppo/entropy_normalized', 'ppo/entropy_coeff', 'ppo/approx_kl', 'ppo/clip_fraction',
    'ppo/value_loss', 'ppo/policy_loss',
    'debug/chosen_action_prob_mean', 'debug/returns_mean', 'debug/returns_std',
    'actions/consumable_hand_target_fraction', 'actions/consumable_joker_target_fraction',
    'actions/consumable_slot_fraction', 'actions/use_consumable_fraction',
    'actions/shop_reroll_fraction', 'actions/shop_sell_joker_fraction',
    'actions/shop_sell_consumable_fraction', 'actions/play_subset_fraction',
    'actions/discard_subset_fraction'
]

run_path = sys.argv[1]
ea = EventAccumulator(run_path, size_guidance={'scalars': 0})
ea.Reload()

all_tags = sorted(ea.Tags()['scalars'])
print(f"=== ALL TAGS IN {run_path.split('/')[-1]} ({len(all_tags)} total) ===")
for i, tag in enumerate(all_tags[:80]):
    print(f"{i+1:3d}. {tag}")

print(f"\n=== DATA FOR TARGET TAGS ===")
for tag in SCALAR_TAGS:
    if tag in all_tags:
        events = ea.Scalars(tag)
        print(f"\n{tag}:")
        for step, val, _ in events:
            print(f"  step={step} val={val:.6g}")
    else:
        print(f"\n{tag}: [NOT FOUND]")
