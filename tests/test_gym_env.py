"""Smoke tests for the Gymnasium environment."""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium.vector.vector_env import AutoresetMode

from pylatro import add_consumable, add_joker, load_game_data, populate_shop
from pylatro.models import PackState, ShopCard
from pylatro_agent.constants import (
    CURRENT_ANTE_SCALAR_INDEX,
    NUM_ACTIONS,
    TOKEN_DIM,
    WIN_ANTE_SCALAR_INDEX,
    ActionRange,
    SubPhase,
)
from pylatro_agent.env import BalatroEnv
from pylatro_agent.training.ppo import _make_vectorized_envs
from pylatro_agent.vocab import build_vocab
from pylatro_cli.controller import GamePhase


@pytest.fixture(scope="module")
def game_data():
    return load_game_data()


@pytest.fixture(scope="module")
def vocab(game_data):
    return build_vocab(game_data)


def test_env_reset(game_data, vocab):
    env = BalatroEnv(seed=42, data=game_data, vocab=vocab)
    obs, info = env.reset()

    assert "tokens" in obs
    assert "token_types" in obs
    assert "scalars" in obs
    assert "attention_mask" in obs
    assert "action_mask" in obs
    assert obs["tokens"].shape == (160, TOKEN_DIM)
    assert obs["action_mask"].shape == (NUM_ACTIONS,)
    assert info["sub_phase"] == "blind_select"


def test_env_action_mask_has_valid_actions(game_data, vocab):
    env = BalatroEnv(seed=42, data=game_data, vocab=vocab)
    obs, _ = env.reset()
    mask = obs["action_mask"]
    assert mask.sum() > 0, "Must have at least one valid action"


def test_env_emits_ante1_realized_play_and_clear_efficiency(game_data, vocab):
    from pylatro_agent.action import ActionType, decode_action

    env = BalatroEnv(seed=42, data=game_data, vocab=vocab, enable_teacher=False)
    env.reset()
    env.step(int(ActionRange.BLIND_PLAY))
    target = env._controller.blind_target()
    env._controller.round_score = target - 1
    hands_before = env.state.current_round.hands_left
    play_action = next(
        int(action)
        for action in np.flatnonzero(env.action_masks())
        if decode_action(int(action)).action_type == ActionType.PLAY_SUBSET
    )

    _, _, terminated, truncated, info = env.step(play_action)

    assert not terminated and not truncated
    assert info["ante1_play_observed"] is True
    assert info["ante1_play_realized_score"] > 0.0
    assert info["ante1_play_realized_to_remaining_target"] == info["ante1_play_realized_score"]
    assert info["ante1_play_hand"]
    assert info["ante1_blind_cleared"] is True
    assert info["ante1_blind_clear_type"] == "small"
    assert info["ante1_blind_clear_hands_used"] == 1
    assert info["ante1_blind_clear_hands_unused"] == hands_before - 1
    assert info["ante1_blind_clear_discards_used"] == 0


def test_env_next_ante_suit_boss_is_visible_after_disabled_boss_cashout(game_data, vocab):
    env = BalatroEnv(seed=91, data=game_data, vocab=vocab, enable_teacher=False)
    env.reset()
    state = env.state
    state.blind_on_deck = "Boss"
    state.round_resets.blind_choices["Boss"] = "bl_goad"
    env._controller.select_blind("Boss")
    for card in state.deck_cards:
        card.seal = None
    next(card for card in state.deck_cards if card.suit == "Spades").seal = "Blue"
    state.blind_disabled = True
    env._controller.cash_out()
    state.round_resets.blind_choices["Boss"] = "bl_goad"
    env._sub_phase = SubPhase.BLIND_SELECT

    env.step(int(ActionRange.BLIND_SKIP))
    obs, _, terminated, truncated, _ = env.step(int(ActionRange.BLIND_SKIP))

    assert not terminated and not truncated
    assert state.blind_on_deck == "Boss"
    assert not state.blind_disabled
    assert obs["scalars"][13] == 0.0  # p_blue
    assert obs["scalars"][18] == 0.0  # Spade target utility


def test_env_random_rollout(game_data, vocab):
    """Run random valid actions and ensure no crashes."""
    env = BalatroEnv(seed=42, data=game_data, vocab=vocab, max_steps=500)
    obs, _ = env.reset()
    steps = 0

    for _ in range(500):
        mask = obs["action_mask"]
        valid = np.where(mask == 1)[0]
        if len(valid) == 0:
            break
        action = np.random.choice(valid)
        obs, _reward, terminated, truncated, _info = env.step(action)
        steps += 1
        if terminated or truncated:
            break

    assert steps > 0, "Should have taken at least one step"


def test_env_step_returns_teacher_action_in_info(game_data, vocab):
    """The env should expose a heuristic-teacher action via info["teacher_action"]."""
    env = BalatroEnv(seed=42, data=game_data, vocab=vocab, max_steps=100)
    obs, _ = env.reset()
    pre_mask = obs["action_mask"]
    valid = np.where(pre_mask == 1)[0]
    assert len(valid) > 0
    _obs, _r, _t, _tr, info = env.step(int(valid[0]))
    assert "teacher_action" in info
    teacher = info["teacher_action"]
    assert isinstance(teacher, int)
    # -1 sentinel allowed; otherwise must be in range AND have been valid under
    # the pre-step mask.
    if teacher >= 0:
        assert teacher < NUM_ACTIONS
        assert pre_mask[teacher] == 1


def test_env_step_marks_exact_teacher_action_match(game_data, vocab):
    env = BalatroEnv(seed=42, data=game_data, vocab=vocab, max_steps=100)
    _obs, info = env.reset()
    teacher = info["teacher_action"]
    if teacher < 0:
        pytest.skip("heuristic did not expose a valid action for this seed")

    _next_obs, _reward, _terminated, _truncated, step_info = env.step(teacher)

    assert step_info["teacher_action"] == teacher
    assert step_info["teacher_action_match"] is True


def test_env_enable_teacher_false_emits_sentinels(game_data, vocab):
    """With the teacher disabled, every teacher field is the -1 sentinel and
    no HeuristicAgent is ever constructed (it is per-step overhead when no
    training objective consumes the labels)."""
    env = BalatroEnv(seed=42, data=game_data, vocab=vocab, max_steps=100, enable_teacher=False)
    obs, info = env.reset()

    assert env._teacher is None
    assert info["teacher_action"] == -1

    action = int(np.flatnonzero(obs["action_mask"])[0])
    _next_obs, _reward, _terminated, _truncated, step_info = env.step(action)

    assert step_info["teacher_action"] == -1
    assert step_info["next_teacher_action"] == -1
    assert not step_info["teacher_action_match"]


def test_env_reset_returns_current_teacher_action_in_info(game_data, vocab):
    env = BalatroEnv(seed=42, data=game_data, vocab=vocab, max_steps=100)
    obs, info = env.reset()

    assert "teacher_action" in info
    teacher = info["teacher_action"]
    assert isinstance(teacher, int)
    if teacher >= 0:
        assert teacher < NUM_ACTIONS
        assert obs["action_mask"][teacher] == 1


def test_env_step_returns_next_teacher_action_for_nonterminal_step(game_data, vocab):
    env = BalatroEnv(seed=42, data=game_data, vocab=vocab, max_steps=100)
    obs, _ = env.reset()
    action = int(np.where(obs["action_mask"] == 1)[0][0])

    next_obs, _r, terminated, truncated, info = env.step(action)

    assert not terminated
    assert not truncated
    assert "next_teacher_action" in info
    teacher = info["next_teacher_action"]
    assert isinstance(teacher, int)
    if teacher >= 0:
        assert teacher < NUM_ACTIONS
        assert next_obs["action_mask"][teacher] == 1


def test_env_multiple_resets(game_data, vocab):
    """Ensure environment can be reset multiple times."""
    env = BalatroEnv(seed=1, data=game_data, vocab=vocab, max_steps=100)
    for seed in range(3):
        obs, _info = env.reset(seed=seed)
        mask = obs["action_mask"]
        assert mask.sum() > 0

        for _ in range(10):
            valid = np.where(mask == 1)[0]
            if len(valid) == 0:
                break
            action = np.random.choice(valid)
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
            mask = obs["action_mask"]


def test_env_constructor_seed_is_only_used_for_first_reset(game_data, vocab):
    env = BalatroEnv(seed=7, data=game_data, vocab=vocab, max_steps=100)

    _obs1, _ = env.reset()
    first_seed = env.state.seed

    _obs2, _ = env.reset()
    second_seed = env.state.seed

    assert first_seed == "7"
    assert second_seed != first_seed


def test_env_win_ante_override_is_applied_on_every_reset(game_data, vocab):
    env = BalatroEnv(seed=7, data=game_data, vocab=vocab, max_steps=100, win_ante=3)

    env.reset()
    assert env.state.win_ante == 3

    env.reset()
    assert env.state.win_ante == 3


def test_winning_terminal_observation_stays_within_critic_horizon(game_data, vocab):
    env = BalatroEnv(
        seed=7,
        data=game_data,
        vocab=vocab,
        max_steps=100,
        win_ante=1,
        enable_teacher=False,
    )
    env.reset()
    assert env.state is not None

    # Jump to the target Ante's boss and make the next legal play clear it.
    env.state.blind_on_deck = "Boss"
    env.state.round_resets.blind_states = {
        "Small": "Defeated",
        "Big": "Defeated",
        "Boss": "Select",
    }
    hand_obs, _, _, _, _ = env.step(int(ActionRange.BLIND_PLAY))
    env._controller.round_score = env._controller.blind_target()
    play_action = _first_valid(
        hand_obs["action_mask"],
        ActionRange.PLAY_SUBSET_START,
        ActionRange.PLAY_SUBSET_END,
    )

    terminal_obs, _, terminated, truncated, info = env.step(play_action)

    assert terminated and not truncated
    assert info["won"] is True
    assert info["ante"] == 2  # The engine still advances after cashing out.
    assert terminal_obs["scalars"][CURRENT_ANTE_SCALAR_INDEX] == 1.0
    assert terminal_obs["scalars"][WIN_ANTE_SCALAR_INDEX] == 1.0


def test_env_shop_buy_opens_booster_pack_without_index_error(game_data, vocab):
    env = BalatroEnv(seed=42, data=game_data, vocab=vocab)
    env.reset()

    assert env.state is not None
    env.state.dollars = 100
    populate_shop(env.state)
    env._controller.phase = GamePhase.SHOP
    env._sub_phase = SubPhase.SHOP

    obs = env._obs_to_dict(env._build_obs())
    booster_offset = len(env.state.shop.cards) + len(env.state.shop.vouchers)
    action = ActionRange.SHOP_BUY_START + booster_offset

    assert obs["action_mask"][action] == 1

    _, _, terminated, truncated, info = env.step(action)

    assert not terminated
    assert not truncated
    assert "error" not in info
    assert env.state.pack is not None
    assert env._sub_phase == SubPhase.BOOSTER_PACK


def test_env_play_subset_executes_directly(game_data, vocab):
    env = BalatroEnv(seed=42, data=game_data, vocab=vocab, max_steps=3)
    env.reset()

    _, _, terminated, truncated, info = env.step(ActionRange.BLIND_PLAY)
    assert not terminated
    assert not truncated
    assert info["progress_made"]

    play_action = _first_valid(
        env.action_masks(),
        ActionRange.PLAY_SUBSET_START,
        ActionRange.PLAY_SUBSET_END,
    )
    _, _, terminated, truncated, info = env.step(play_action)
    assert not terminated
    assert not truncated
    assert info["progress_made"]
    assert env._sub_phase == SubPhase.CHOOSE_ACTION


def test_env_play_subset_reports_progress(game_data, vocab):
    env = BalatroEnv(seed=42, data=game_data, vocab=vocab, max_steps=3)
    env.reset()

    _, _, _, _, _ = env.step(ActionRange.BLIND_PLAY)
    play_action = _first_valid(
        env.action_masks(),
        ActionRange.PLAY_SUBSET_START,
        ActionRange.PLAY_SUBSET_END,
    )
    _, _, terminated, truncated, info = env.step(play_action)
    assert not terminated
    assert not truncated
    assert info["progress_made"]
    assert info["steps_since_progress"] == 0
    assert not info["stalled"]
    assert info["hand_play_observed"]
    assert info.get("hand_play_in_candidates") or info.get("hand_play_not_in_candidates")
    assert "counterfactual_call" not in info


def _count_build_diagnostics_on_a_play(game_data, vocab, monkeypatch, *, potential):
    """Play one hand and report how often the build evaluator ran."""
    import pylatro_agent.env as env_module
    from pylatro_agent.reward import RewardConfig

    calls = 0
    original = env_module.build_step_diagnostics

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(env_module, "build_step_diagnostics", counted)
    env = BalatroEnv(
        seed=42,
        data=game_data,
        vocab=vocab,
        enable_teacher=False,
        reward_config=RewardConfig(enable_score_build_potential=potential),
    )
    env.reset()
    env.step(ActionRange.BLIND_PLAY)
    play_action = _first_valid(
        env.action_masks(),
        ActionRange.PLAY_SUBSET_START,
        ActionRange.PLAY_SUBSET_END,
    )
    _, _, _, _, info = env.step(play_action)
    return calls, info


def test_play_skips_detailed_build_diagnostics_without_the_build_potential(
    game_data, vocab, monkeypatch
):
    calls, info = _count_build_diagnostics_on_a_play(
        game_data, vocab, monkeypatch, potential=False
    )

    assert calls == 0
    assert "build_diagnostics_observed" not in info


def test_build_potential_evaluates_the_build_on_every_play(game_data, vocab, monkeypatch):
    """Potential shaping is a per-transition difference, so it cannot skip plays.

    This is the throughput cost of the build family: roughly 3x slower env
    stepping (about 90 -> 30 steps/s), paid on every step rather than only on
    shop and pack events.
    """
    calls, _info = _count_build_diagnostics_on_a_play(
        game_data, vocab, monkeypatch, potential=True
    )

    assert calls > 0


def test_env_atomic_consumable_use_commits_in_one_step(game_data, vocab):
    """A no-target consumable (Pluto = planet) is a single flat action now."""
    from pylatro_agent.action import ActionType, encode_action

    env = BalatroEnv(seed=42, data=game_data, vocab=vocab, max_steps=5)
    env.reset()

    _, _, terminated, truncated, _info = env.step(ActionRange.BLIND_PLAY)
    assert not terminated
    assert not truncated

    assert env.state is not None
    add_consumable(env.state, "c_pluto")
    obs = env._obs_to_dict(env._build_obs())
    action = encode_action(ActionType.USE_CONSUMABLE_NO_TARGET, 0)
    assert obs["action_mask"][action] == 1

    _, _, terminated, truncated, info = env.step(action)
    assert not terminated
    assert not truncated
    # After committing Pluto we stay in CHOOSE_ACTION with the consumable consumed
    assert info["sub_phase"] == SubPhase.CHOOSE_ACTION
    assert info["progress_made"], "using a planet bumps the hand-level tracker"
    assert info["planet_use_observed"]
    assert info["planet_use_key"] == "c_pluto"
    assert info["planet_use_hand_type"] == "High Card"


def test_env_pack_skip_counts_as_progress_and_triggers_red_card(game_data, vocab):
    env = BalatroEnv(seed=42, data=game_data, vocab=vocab)
    env.reset()

    assert env.state is not None
    env.state.dollars = 100
    add_joker(env.state, "j_red_card")
    populate_shop(env.state)
    env._controller.phase = GamePhase.SHOP
    env._sub_phase = SubPhase.SHOP

    booster_offset = len(env.state.shop.cards) + len(env.state.shop.vouchers)
    buy_action = ActionRange.SHOP_BUY_START + booster_offset
    _, _, terminated, truncated, info = env.step(buy_action)
    assert not terminated
    assert not truncated
    assert info["progress_made"]
    assert env._sub_phase == SubPhase.BOOSTER_PACK

    _, _, terminated, truncated, info = env.step(ActionRange.PACK_SKIP)
    assert not terminated
    assert not truncated
    assert info["progress_made"]
    assert info["steps_since_progress"] == 0
    assert env._sub_phase == SubPhase.SHOP
    assert env.state.jokers[0].mult == 3


def test_env_sell_joker_exposes_build_event_and_potential_diagnostics(game_data, vocab):
    from pylatro_agent.action import ActionType, encode_action

    env = BalatroEnv(seed=42, data=game_data, vocab=vocab, enable_teacher=False)
    env.reset()
    assert env.state is not None
    add_joker(env.state, "j_joker")
    env._controller.phase = GamePhase.SHOP
    env._sub_phase = SubPhase.SHOP

    action = encode_action(ActionType.SHOP_SELL_JOKER, 0)
    _, _, terminated, truncated, info = env.step(action)

    assert not terminated
    assert not truncated
    assert info["shop_sold_joker_id"] == "j_joker"
    assert info["joker_removed_count"] == 1
    assert info["joker_removed_0_id"] == "j_joker"
    assert info["joker_churn_count"] == 1
    assert info["build_diagnostics_observed"] is True
    assert info["build_pre_estimated_score"] >= info["build_post_estimated_score"]
    assert info["build_pre_required_score"] >= 0.0
    assert info["build_post_readiness"] >= 0.0
    assert "potential_pre_total" in info
    assert "potential_post_total" in info
    assert "potential_delta_total" in info


def test_env_counterfactual_replays_one_copied_play_without_mutating_live_state_twice(game_data, vocab, monkeypatch):
    from pylatro_agent import diagnostics as diagnostics_module
    from pylatro_agent.action import ActionType, decode_action

    copy_calls = 0
    replay_calls = 0
    original_deepcopy = diagnostics_module.deepcopy
    original_play_cards = diagnostics_module.play_cards

    def counted_deepcopy(*args, **kwargs):
        nonlocal copy_calls
        copy_calls += 1
        return original_deepcopy(*args, **kwargs)

    def counted_play_cards(*args, **kwargs):
        nonlocal replay_calls
        replay_calls += 1
        return original_play_cards(*args, **kwargs)

    monkeypatch.setattr(diagnostics_module, "deepcopy", counted_deepcopy)
    monkeypatch.setattr(diagnostics_module, "play_cards", counted_play_cards)

    env = BalatroEnv(
        seed=42,
        data=game_data,
        vocab=vocab,
        enable_teacher=False,
        counterfactual_diagnostic_interval=1,
    )
    env.reset()
    env.step(ActionRange.BLIND_PLAY)
    assert env.state is not None
    add_joker(env.state, "j_joker")
    hands_before = env.state.current_round.hands_left
    play_action = next(
        int(action)
        for action in np.flatnonzero(env.action_masks())
        if decode_action(int(action)).action_type == ActionType.PLAY_SUBSET
    )

    _, _, _, _, info = env.step(play_action)

    assert info["counterfactual_call"] is True
    assert info["counterfactual_focal_joker_id"] == "j_joker"
    assert info["counterfactual_failure"] is False
    assert info["counterfactual_realized_score_with"] >= 0.0
    assert info["counterfactual_realized_score_without"] >= 0.0
    assert info["counterfactual_representative_vs_realized_abs_log_ratio_gap"] >= 0.0
    assert copy_calls == 1
    assert replay_calls == 1
    assert env.state.current_round.hands_left == hands_before - 1


def test_env_hologram_scaling_reports_xmult_and_build_delta(game_data, vocab):
    from pylatro_agent.action import ActionType, encode_action

    env = BalatroEnv(seed=42, data=game_data, vocab=vocab, enable_teacher=False)
    env.reset()
    assert env.state is not None
    hologram = add_joker(env.state, "j_hologram")
    before_x_mult = hologram.x_mult
    env.state.pack = PackState(
        booster_key="p_standard_normal_1",
        state_name="STANDARD_PACK",
        choices_remaining=1,
        cards=[
            ShopCard(
                center_key="c_base",
                card_type="Default",
                cost=0,
                base_cost=0,
                front_key="S_A",
                seal="Blue",
            )
        ],
    )
    env._controller.phase = GamePhase.BOOSTER_PACK
    env._sub_phase = SubPhase.BOOSTER_PACK

    action = encode_action(ActionType.PACK_CLAIM, 0)
    _, _, terminated, truncated, info = env.step(action)

    assert not terminated
    assert not truncated
    assert info["hologram_scaling_count"] == 1
    assert info["hologram_x_mult_prev"] == pytest.approx(before_x_mult)
    assert info["hologram_x_mult_current"] == pytest.approx(hologram.x_mult)
    assert info["hologram_x_mult_delta"] > 0.0
    assert info["hologram_build_score_delta"] > 0.0
    assert info["pack_claim_seal"] == "Blue"
    assert info["seal_pre_blue_count"] == 0
    assert info["seal_post_blue_count"] == 1


def _first_valid(mask: np.ndarray, start: int, end: int) -> int:
    valid = np.where(mask[start : end + 1] == 1)[0]
    assert len(valid) > 0
    return start + int(valid[0])


def test_vector_env_uses_same_step_autoreset(game_data, vocab):
    vec_env = _make_vectorized_envs(1, game_data, vocab, use_async=False)

    try:
        assert vec_env.metadata["autoreset_mode"] == AutoresetMode.SAME_STEP
    finally:
        vec_env.close()


def test_vector_env_carries_numeric_build_diagnostics_when_build_potential_enabled(game_data, vocab):
    from pylatro_agent.reward import RewardConfig

    vec_env = _make_vectorized_envs(
        1,
        game_data,
        vocab,
        use_async=False,
        reward_config=RewardConfig(enable_score_build_potential=True),
    )

    try:
        vec_env.reset()
        obs, _, _, _, _ = vec_env.step(np.array([ActionRange.BLIND_PLAY]))
        play_action = _first_valid(
            obs["action_mask"][0],
            ActionRange.PLAY_SUBSET_START,
            ActionRange.PLAY_SUBSET_END,
        )
        _, _, _, _, infos = vec_env.step(np.array([play_action]))

        assert bool(infos["build_diagnostics_observed"][0])
        assert np.issubdtype(infos["build_post_estimated_score"].dtype, np.number)
        assert np.issubdtype(infos["potential_post_total"].dtype, np.number)
    finally:
        vec_env.close()


def test_vector_env_passes_win_ante_override(game_data, vocab):
    vec_env = _make_vectorized_envs(1, game_data, vocab, use_async=False, win_ante=2)

    try:
        vec_env.reset()
        assert vec_env.envs[0].state.win_ante == 2
    finally:
        vec_env.close()


def test_async_vector_env_reset_matches_declared_token_shape(game_data, vocab):
    vec_env = _make_vectorized_envs(2, game_data, vocab, use_async=True)

    try:
        obs, _info = vec_env.reset()
        assert obs["tokens"].shape == (2, 160, TOKEN_DIM)
    finally:
        vec_env.close()


def test_env_step_info_carries_boss_key(game_data, vocab):
    """Episode-end death diagnostics group losses by info["boss_key"]."""
    env = BalatroEnv(seed=42, data=game_data, vocab=vocab, max_steps=100)
    obs, _ = env.reset()
    valid = np.where(obs["action_mask"] == 1)[0]
    _obs, _r, _t, _tr, info = env.step(int(valid[0]))

    assert info["boss_key"].startswith("bl_")
    assert info["boss_key"] in game_data.blinds
    assert info["blind_target"] > 0
