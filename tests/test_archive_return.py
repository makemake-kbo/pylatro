"""Archive continuations preserve the simulator and never reissue past rewards."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pylatro import add_joker, load_game_data
from pylatro_agent.agent import AgentConfig, BalatroAgent
from pylatro_agent.archive import ArchiveConfig, StateArchive, pack_snapshot, unpack_snapshot
from pylatro_agent.constants import JOKER_START, ActionRange, SubPhase
from pylatro_agent.env import BalatroEnv
from pylatro_agent.joker_features import JOKER_FEATURE_NAMES, JOKER_FEATURE_START
from pylatro_agent.reward import RewardConfig, default_reward_components
from pylatro_agent.training.ppo import (
    PPOConfig,
    _make_vectorized_envs,
    _save_checkpoint,
    load_actor_transfer,
    resolve_milestone_scale,
)
from pylatro_agent.value_head import ValueHead
from pylatro_agent.vocab import build_vocab


@pytest.fixture(scope="module")
def data():
    return load_game_data()


def make_env(data, **kwargs):
    return BalatroEnv(
        data=data,
        seed=42,
        enable_teacher=False,
        win_ante=8,
        reward_config=RewardConfig(objective="milestone"),
        **kwargs,
    )


def clear_blind(env):
    env.step(int(ActionRange.BLIND_PLAY))
    env._controller.round_score = env._controller.blind_target() - 1
    actions = np.flatnonzero(env.action_masks())
    action = next(int(a) for a in actions if ActionRange.PLAY_SUBSET_START <= a <= ActionRange.PLAY_SUBSET_END)
    return env.step(action)


def test_snapshot_replays_rng_history_and_observations_without_aliasing(data):
    env = make_env(data)
    env.reset()
    clear_blind(env)
    assert env._sub_phase == SubPhase.SHOP
    env.state.dollars = 50
    env._rewarded_gold_card_ids.add(env.state.deck_cards[0].reward_uid)
    snapshot = env.snapshot()
    payload = pack_snapshot(snapshot, data)
    assert len(payload) < 100_000  # immutable GameData is not copied into each cell
    restored = make_env(data)
    restored.restore_snapshot(unpack_snapshot(payload, data))
    assert restored.state is not env.state
    assert restored.state.data is data
    assert restored._rewarded_gold_card_ids == {restored.state.deck_cards[0].reward_uid}
    for original, copied in zip(env.state.deck_cards, restored.state.deck_cards, strict=True):
        assert original is not copied
    for action in (ActionRange.SHOP_REROLL, ActionRange.SHOP_LEAVE, ActionRange.BLIND_PLAY):
        left = env.step(int(action))
        right = restored.step(int(action))
        assert left[1:4] == right[1:4]
        for key in left[0]:
            np.testing.assert_array_equal(left[0][key], right[0][key], err_msg=key)
    assert snapshot["controller"].state.dollars == 50
    assert restored.state.hand_cards[0] in restored.state.deck_cards
    assert any(restored.state.hand_cards[0] is card for card in restored.state.deck_cards)


def test_archive_reset_and_explicit_seed_fresh_start(data):
    config = ArchiveConfig(min_ante=1, return_probability=1.0)
    env = make_env(data, archive_config=config)
    env.reset()
    clear_blind(env)
    assert env.archive_metrics()["states"] == 1
    obs, info = env.reset()
    assert info["archive_start"] and info["start_ante"] == 1
    assert env._sub_phase == SubPhase.SHOP
    assert obs["history_event_mask"].sum() == 1
    _, info = env.reset(seed=42)
    assert not info["archive_start"]
    assert env._sub_phase == SubPhase.BLIND_SELECT
    assert env.archive_metrics()["states"] == 1


def test_milestones_survive_returns_and_pay_once(data):
    env = make_env(data)
    env.reset()
    for _ in range(2):
        clear_blind(env)
        env.step(int(ActionRange.SHOP_LEAVE))
    _, reward, done, _, info = clear_blind(env)
    assert not done
    assert info["boss_cleared_ante"] == 1
    assert reward == pytest.approx(1 / 7)
    snapshot = env.snapshot()
    restored = make_env(data)
    restored.set_milestone_scale(0.25)
    restored.restore_snapshot(snapshot)
    assert restored._cleared_boss_antes == {1}
    assert restored._milestone_scale == 0.25
    # Simulate returning to an already cleared Ante through Hieroglyph-like
    # ante changes. It must not create a second milestone payment.
    restored.state.round_resets.ante = 1
    restored.state.blind_on_deck = "Boss"
    restored._controller.leave_shop()
    restored._sub_phase = SubPhase.BLIND_SELECT
    _, reward, _, _, info = clear_blind(restored)
    assert info["boss_cleared_ante"] == 0
    assert reward == 0


def test_milestone_budget_terminal_and_disabled_event_bonuses():
    state = SimpleNamespace(win_ante=8, round_resets=SimpleNamespace(ante=8))
    config = RewardConfig(objective="milestone", milestone_reward_budget=2)
    total = 0
    for ante in range(1, 8):
        parts = default_reward_components(
            state,
            {},
            {
                "boss_cleared_ante": ante,
                "strategic_attributable_cash_payout": 100,
                "strategic_blue_planets_generated": 10,
            },
            False,
            False,
            config,
        )
        total += parts["total"]
        assert parts["strategic_cash_payout"] == parts["potential_shaping"] == 0
    assert total == pytest.approx(2)
    assert default_reward_components(state, {}, {"boss_cleared_ante": 8}, True, True, config)["total"] == 10
    assert default_reward_components(state, {}, {}, True, False, config)["total"] == 0
    assert default_reward_components(state, {}, {"stalled": True}, False, False, config)["total"] == 0


def test_archive_reservoir_is_bounded_diverse_and_resumable():
    config = ArchiveConfig(return_probability=1.0, capacity_per_bucket=2)
    archive = StateArchive(config, seed=1)
    for ante in (4, 5, 6):
        for index in range(30):
            archive.add(ante=ante, phase="shop", identity=str(index), payload=f"{ante}:{index}".encode())
    assert archive.metrics()["states"] == 6
    for bucket in archive.buckets.values():
        assert len({identity for identity, _ in bucket}) == 2
    resumed = StateArchive(config, seed=99)
    resumed.load_state_dict(deepcopy(archive.state_dict()))
    first = [archive.choose() for _ in range(100)]
    assert first == [resumed.choose() for _ in range(100)]
    assert {int(payload[:1]) for payload in first} == {4, 5, 6}
    with pytest.raises(ValueError, match="mismatch"):
        StateArchive(ArchiveConfig(), seed=1).load_state_dict(archive.state_dict())


@pytest.mark.parametrize("use_async", [False, True])
def test_vector_workers_restore_their_own_archives(data, use_async):
    config = ArchiveConfig(min_ante=1, return_probability=1)
    source = make_env(data, archive_config=config)
    source.reset()
    clear_blind(source)
    state = source.archive_state_dict()
    vec = _make_vectorized_envs(
        2,
        data,
        build_vocab(data),
        use_async=use_async,
        win_ante=8,
        archive_config=config,
        reward_config=RewardConfig(objective="milestone"),
    )
    try:
        vec.call("load_archive_states", [deepcopy(state), deepcopy(state)])
        obs, info = vec.reset()
        assert info["archive_start"].tolist() == [True, True]
        assert obs["history_event_mask"].sum() == 2
        vec.call("set_milestone_scale", 0.5)
        result = vec.step(np.array([int(ActionRange.SHOP_LEAVE)] * 2))
        assert not result[2].any()
    finally:
        vec.close()


def test_new_joker_features_distinguish_growth_and_decay(data):
    env = make_env(data)
    env.reset()
    green = add_joker(env.state, "j_green_joker")
    runner = add_joker(env.state, "j_runner")
    ice = add_joker(env.state, "j_ice_cream")
    before = env._build_obs().tokens.copy()
    green.mult = 5
    runner.extra["chips"] = 300
    ice.extra["chips"] = 10
    after = env._build_obs().tokens
    assert before[JOKER_START, 4] == after[JOKER_START, 4]  # legacy alias remains compatible
    mult_index = JOKER_FEATURE_START + JOKER_FEATURE_NAMES.index("mult")
    chips_index = JOKER_FEATURE_START + JOKER_FEATURE_NAMES.index("extra_chips")
    assert before[JOKER_START, mult_index] < after[JOKER_START, mult_index]
    assert before[JOKER_START + 1, chips_index] < after[JOKER_START + 1, chips_index]
    assert before[JOKER_START + 2, chips_index] > after[JOKER_START + 2, chips_index]


def test_actor_transfer_keeps_actor_resets_critic_and_initializes_target(data, tmp_path):
    config = AgentConfig(d_model=32, n_layers=1, n_heads=4, d_ff=64)
    source = BalatroAgent(config, build_vocab(data))
    old_weights = {k: v.clone() for k, v in source.state_dict().items() if k != "embedding.joker_emb.state_proj.weight"}
    for key in old_weights:
        if key.startswith("value_head."):
            old_weights[key].fill_(42)
    path = tmp_path / "actor_v11.pt"
    torch.save(
        {
            "tokenizer_version": 11,
            "tokenizer_semantics": "v8_conditional_survival_critic",
            "state_dict": old_weights,
            "ppo_config_fields": {"win_ante": 5},
        },
        path,
    )
    target = BalatroAgent(config, build_vocab(data))
    critic_before = {k: v.clone() for k, v in target.state_dict().items() if k.startswith("value_head.")}
    load_actor_transfer(target, str(path), torch.device("cpu"))
    for key, value in critic_before.items():
        torch.testing.assert_close(target.state_dict()[key], value)
    torch.testing.assert_close(target.backbone.layers[0].norm1.weight, source.backbone.layers[0].norm1.weight)
    torch.testing.assert_close(
        target.embedding.meta_emb.win_ante_emb.weight[8], source.embedding.meta_emb.win_ante_emb.weight[5]
    )
    assert target.embedding.joker_emb.state_proj.weight.count_nonzero() == 0


def test_win_only_critic_and_schedule():
    head = ValueHead(16, win_only=True)
    output = head(torch.randn(2, 3, 16), torch.ones(2, 3), torch.tensor([1, 6]), torch.tensor([8, 8]))
    torch.testing.assert_close(output["terminal_value"], 10 * output["win_prob"])
    config = PPOConfig(milestone_decay_fraction=0.5, milestone_final_scale=0.1)
    assert resolve_milestone_scale(config, 0, 1000) == 1
    assert resolve_milestone_scale(config, 250, 1000) == pytest.approx(0.55)
    assert resolve_milestone_scale(config, 500, 1000) == pytest.approx(0.1)
    assert resolve_milestone_scale(config, 2000, 1000) == pytest.approx(0.1)


def test_checkpoint_contains_archive_and_can_restart_from_it(data, tmp_path):
    archive_config = ArchiveConfig(min_ante=1, return_probability=1)
    env = make_env(data, archive_config=archive_config)
    env.reset()
    clear_blind(env)
    vec = _make_vectorized_envs(1, data, build_vocab(data), use_async=False, archive_config=archive_config)
    try:
        vec.call("load_archive_states", [env.archive_state_dict()])
        model = torch.nn.Linear(2, 2)
        path = _save_checkpoint(
            model=model,
            optimizer=torch.optim.Adam(model.parameters()),
            save_path=tmp_path,
            update_count=1,
            total_steps=4,
            planned_updates=5,
            entropy_coeff=0.01,
            entropy_signal_ema=None,
            lr=1e-4,
            config=PPOConfig(archive_config=archive_config),
            vector_env=vec,
        )
        saved = torch.load(path, weights_only=False)
        assert saved["archive_states"][0]["buckets"]
        restored = make_env(data, archive_config=archive_config)
        restored.load_archive_states(saved["archive_states"])
        _, info = restored.reset()
        assert info["archive_start"]
    finally:
        vec.close()


def test_ppo_collects_new_archive_suffixes_and_resumes(data, tmp_path, monkeypatch):
    from dataclasses import replace

    from pylatro_agent.training import ppo

    archive_config = ArchiveConfig(min_ante=1, return_probability=1)
    source = make_env(data, archive_config=archive_config)
    source.reset()
    clear_blind(source)
    archived = source.archive_state_dict()
    original_factory = ppo._make_vectorized_envs

    def factory(*args, **kwargs):
        vec = original_factory(*args, **kwargs)
        vec.call("load_archive_states", [deepcopy(archived)])
        return vec

    eval_targets = []

    def evaluate(*args, **kwargs):
        eval_targets.append(kwargs["win_ante"])
        return 0.0

    monkeypatch.setattr(ppo, "_make_vectorized_envs", factory)
    monkeypatch.setattr(ppo, "evaluate_model", evaluate)
    config = PPOConfig(
        num_envs=1,
        rollout_length=2,
        total_updates=2,
        ppo_epochs=1,
        mini_batch_size=2,
        micro_batch_size=2,
        async_envs=False,
        archive_config=archive_config,
        win_ante=8,
        reward_config=RewardConfig(objective="milestone"),
        hand_ar_mixture_eps=0,
        terminal_replay_updates_per_ppo_update=0,
        eval_games=1,
        eval_interval=1,
        checkpoint_interval=1,
        save_dir=str(tmp_path / "checkpoints"),
        log_dir=str(tmp_path / "logs"),
        risk_forecast_log=False,
    )
    model_config = AgentConfig(d_model=32, n_layers=1, n_heads=4, d_ff=64)
    model = ppo.train_ppo(config, model_config, data=data)
    checkpoint = tmp_path / "checkpoints" / "ppo_latest.pt"
    saved = torch.load(checkpoint, weights_only=False)
    assert saved["archive_states"][0]["returns"] >= 1
    assert saved["agent_config"]["win_only_value"] is True
    assert model.config.win_only_value
    assert saved["total_steps"] == 4  # only newly collected suffix steps count
    assert saved["archive_states"][0]["buckets"]
    ppo.train_ppo(replace(config, total_updates=3), model_config, resume_path=str(checkpoint), data=data)
    resumed = torch.load(checkpoint, weights_only=False)
    assert resumed["update_count"] == 3 and resumed["total_steps"] == 6
    assert resumed["schedule_total_steps"] == saved["schedule_total_steps"]
    assert eval_targets == [8, 8, 8]
