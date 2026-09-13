import json
from dataclasses import asdict

import numpy as np
import pytest
import torch

from pylatro import load_game_data
from pylatro_agent.constants import MAX_SEQ_LEN, NUM_ACTIONS, SCALAR_DIM, TOKEN_DIM
from pylatro_agent.env import BalatroEnv
from pylatro_agent.outcome_study import outcome_reward
from pylatro_agent.reward import RewardConfig
from pylatro_agent.training.rollout_buffer import RolloutBuffer
from pylatro_agent.transfer import sha256, summarize
from tools.audit_outcome_gradients import gradient_summary
from tools.report_outcome_ppo import compare_replicates, verified_panel


def outcome(seed, sampling, won):
    return dict(seed=seed, sampling_seed=sampling, won=won, max_ante=9 if won else 1,
                final_ante=9 if won else 1, cleared_antes=list(range(1, 9)) if won else [],
                stalled=False, censored=False)


def test_shared_control_and_training_replicates_preserve_game_seed_bootstrap():
    def rows(winning_seed):
        return dict(
            sampled=[outcome(seed, sampling, seed == winning_seed) for seed in (11, 12) for sampling in (0, 1, 2)],
            greedy=[outcome(seed, None, seed == winning_seed) for seed in (11, 12)],
        )

    result = compare_replicates({0: rows(11), 1: rows(12)}, rows(None))
    for mode in ("sampled", "greedy"):
        assert result[mode]["paired"]["game_seeds"] == 2
        assert result[mode]["paired"]["paired_ci95"] == [0.5, 0.5]
        assert result[mode]["paired"]["sampled_or_greedy_delta"] == 0.5
    assert result["sampled"]["control"]["games"] == 6
    assert result["sampled"]["candidate"]["games"] == 12
    with pytest.raises(ValueError, match="Both training"):
        compare_replicates({0: rows(11)}, rows(None))


def test_gradient_attribution_detects_opposed_shared_gradients_and_private_critic_norm():
    names = ("backbone.weight", "action_grammar_head.weight", "value_head.weight")
    actor = [torch.tensor([3., 4.]), torch.zeros(1), torch.zeros(1)]
    returns = [torch.tensor([-3., -4.]), torch.zeros(1), torch.ones(1)]
    outcomes = [torch.zeros_like(value) for value in actor]
    result = gradient_summary(names, actor, returns, outcomes)
    assert result["backbone"]["actor_critic_cosine"] == -1
    assert result["backbone"]["combined_norm"] == 0
    assert result["value_head"]["actor_norm"] == 0
    assert result["value_head"]["critic_to_actor_norm"] is None
    assert result["all"]["combined_norm"] == 1
    assert result["global_clip_multiplier"] == pytest.approx(0.5 / 1.000001)


def test_raw_panel_verification_rejects_seed_or_repeat_changes(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"a fixed test checkpoint")
    seeds = [11, 12]
    panel = tmp_path / "panel"
    for sampling in (0, 1, 2, None):
        mode = "greedy" if sampling is None else f"sample{sampling}"
        directory = panel / mode
        directory.mkdir(parents=True)
        rows = [outcome(seed, sampling, seed == 11) for seed in seeds]
        metadata = dict(status="complete", seeds=seeds, sampling_seed=sampling, greedy=sampling is None,
                        deck="b_blue", stake=1, win_ante=8, temperature=1.0, batch_size=32, workers=4,
                        device="cpu", reward_config=asdict(outcome_reward()), agent_config={"hand_ar_mixture_eps": .5},
                        checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint), elapsed_seconds=1.)
        (directory / "manifest.json").write_text(json.dumps(metadata))
        (directory / "summary.json").write_text(json.dumps(summarize(rows)))
        (directory / "outcomes.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    groups, evidence = verified_panel(panel, seeds)
    assert len(groups["sampled"]) == 6 and len(evidence["files"]) == 4
    path = panel / "sample0/outcomes.jsonl"
    path.write_text(json.dumps(outcome(11, 0, True)) + "\n" + json.dumps(outcome(11, 0, False)) + "\n")
    with pytest.raises(ValueError, match="Missing, duplicate"):
        verified_panel(panel, seeds)
    path.write_text(json.dumps(outcome(11, 1, True)) + "\n" + json.dumps(outcome(12, 0, False)) + "\n")
    with pytest.raises(ValueError, match="Wrong sampling"):
        verified_panel(panel, seeds)


def test_outcome_reward_change_preserves_observations_and_gameplay():
    data = load_game_data()
    rng = np.random.default_rng(1729)
    for seed in (6412345, 6412346):
        environments = [BalatroEnv(seed=seed, data=data, deck_key="b_blue", enable_teacher=False,
                                  reward_config=reward) for reward in (RewardConfig(), outcome_reward())]
        try:
            observations = [env.reset(seed=seed)[0] for env in environments]
            for _ in range(32):
                assert observations[0].keys() == observations[1].keys()
                for name in observations[0]:
                    np.testing.assert_array_equal(observations[0][name], observations[1][name], err_msg=name)
                legal = np.flatnonzero(observations[0]["action_mask"])
                action = int(rng.choice(legal))
                results = [env.step(action) for env in environments]
                assert results[0][2:4] == results[1][2:4]
                assert results[0][4]["won"] == results[1][4]["won"]
                observations = [result[0] for result in results]
                if results[0][2] or results[0][3]:
                    break
        finally:
            for env in environments:
                env.close()


def test_undiscounted_full_trace_uses_outcomes_and_separate_fragment_bootstraps():
    buffer = RolloutBuffer(num_envs=2, rollout_length=5, gamma=1, gae_lambda=1)
    obs = dict(tokens=np.zeros((2, MAX_SEQ_LEN, TOKEN_DIM), dtype=np.int16),
               token_types=np.zeros((2, MAX_SEQ_LEN), dtype=np.int8),
               scalars=np.zeros((2, SCALAR_DIM), dtype=np.float32),
               attention_mask=np.ones((2, MAX_SEQ_LEN), dtype=np.int8),
               action_mask=np.ones((2, NUM_ACTIONS), dtype=np.float32))
    # Env 0 wins at step 2, then starts an unfinished episode. Env 1 loses
    # at step 1, then starts another unfinished episode with a different value.
    for step in range(5):
        buffer.add_batch(step, obs, actions=np.zeros(2, dtype=np.int64),
                         rewards=np.array([10. if step == 2 else 0., 0.], dtype=np.float32),
                         values=np.array([step + .25, 8. - step], dtype=np.float32),
                         log_probs=np.zeros(2, dtype=np.float32),
                         terminated=np.array([step == 2, step == 1]), truncated=np.zeros(2, dtype=bool))
    buffer.compute_returns_and_advantages(np.array([3., 7.], dtype=np.float32))
    np.testing.assert_allclose(buffer.returns.reshape(2, 5), [[10., 10., 10., 3., 3.], [0., 0., 7., 7., 7.]])
