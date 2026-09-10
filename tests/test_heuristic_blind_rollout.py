import pickle
from copy import deepcopy

import pytest

from pylatro import create_run_state, start_blind
from pylatro.rng import PseudorandomState
from pylatro_agent.constants import ActionRange, SubPhase
from pylatro_agent.heuristic import HeuristicAgent
from pylatro_agent.heuristic_blind_rollout import BlindRollout, choose_action
from pylatro_agent.masks import compute_action_mask
from pylatro_agent.subset_actions import subset_index


def last_hand(boss="bl_hook", blind="Boss"):
    state = create_run_state('blind_rollout_last_hand', deck_key='b_blue')
    state.round_resets.blind_choices['Boss'] = boss
    start_blind(state, blind)
    cards = {card.front_key: card for card in state.deck_cards}
    state.hand_cards = [cards[key] for key in ('C_2', 'C_3', 'C_4', 'C_5', 'C_6', 'D_9', 'H_K', 'S_A')]
    held = {card.reward_uid for card in state.hand_cards}
    state.draw_pile = [card for card in state.deck_cards if card.reward_uid not in held]
    state.discard_pile = []
    state.current_round.hands_left = 1
    state.current_round.discards_left = 0
    return state


def test_rollout_finds_actual_last_hand_clear_without_mutating_state():
    state = last_hand()
    mask = compute_action_mask(state, SubPhase.CHOOSE_ACTION)
    weak = ActionRange.PLAY_SUBSET_START + subset_index((0,))
    before = pickle.dumps(state)
    action, results = choose_action(HeuristicAgent(shop_policy='search'), state, mask, weak, 0, samples=2)
    assert mask[action] and action != weak
    assert all(not won for won, _ in results[0][2])
    assert all(won for won, _ in next(row[2] for row in results if row[1] == action))
    assert pickle.dumps(state) == before


def test_rollout_is_independent_of_future_draw_order_rng_and_absolute_ids():
    state = last_hand()
    state.current_round.hands_left = 2
    mask = compute_action_mask(state, SubPhase.CHOOSE_ACTION)
    weak = ActionRange.PLAY_SUBSET_START + subset_index((0,))
    expected = choose_action(HeuristicAgent(shop_policy='search'), state, mask, weak, 0, samples=2)
    clone = deepcopy(state, {id(state.data): state.data})
    clone.draw_pile.reverse()
    clone.pseudorandom = PseudorandomState('unrelated_future')
    for card in clone.deck_cards:
        card.reward_uid += 10001
    assert choose_action(HeuristicAgent(shop_policy='search'), clone, mask, weak, 0, samples=2) == expected


def test_perfect_baseline_is_preserved_and_search_runs_once_per_boss():
    state = last_hand()
    mask = compute_action_mask(state, SubPhase.CHOOSE_ACTION)
    agent = HeuristicAgent(shop_policy='search')
    baseline = agent.select_action(state, SubPhase.CHOOSE_ACTION, mask, round_score=0)
    search = BlindRollout()
    assert search.select(agent, state, mask, baseline, 0) == baseline
    assert search.last_decision['outcomes'] == []
    assert search.select(agent, state, mask, baseline, 0) == baseline
    assert search.last_decision is None


def test_adaptive_mode_recalculates_after_a_real_play_and_draw():
    from pylatro_agent.training.fast_runner import FastRunner
    from pylatro_cli.controller import GamePhase

    state = last_hand(boss="bl_head")
    state.current_round.hands_left = 2
    runner = FastRunner(0, state.data, deck_key="b_blue", raise_errors=True)
    runner._state = runner._ctrl.state = state
    runner._ctrl.phase = GamePhase.HAND_PLAY
    runner._sub_phase = SubPhase.CHOOSE_ACTION
    agent = HeuristicAgent(shop_policy="search")
    search = BlindRollout(repeat=True)
    # Spend an irrelevant card first: the retained Straight Flush can still
    # finish next hand, so the sampled baseline already clears every trial.
    spare = ActionRange.PLAY_SUBSET_START + subset_index((5,))
    assert search.select(agent, state, runner.compute_mask(), spare, 0) == spare
    runner.step(spare)
    assert state.current_round.hands_left == 1 and not runner.done
    weak = ActionRange.PLAY_SUBSET_START + subset_index((0,))
    action = search.select(agent, state, runner.compute_mask(), weak, runner.round_score)
    assert action != weak
    runner.step(action)
    assert not runner.done and runner.sub_phase == SubPhase.SHOP


def test_all_blinds_mode_rescues_a_small_blind_final_hand():
    from pylatro_agent.training.fast_runner import FastRunner
    from pylatro_cli.controller import GamePhase

    state = last_hand(blind="Small")
    mask = compute_action_mask(state, SubPhase.CHOOSE_ACTION)
    weak = ActionRange.PLAY_SUBSET_START + subset_index((0,))
    agent = HeuristicAgent(shop_policy="search")
    assert BlindRollout(repeat=True).select(agent, state, mask, weak, 0) == weak
    search = BlindRollout(repeat=True, all_blinds=True)
    action = search.select(agent, state, mask, weak, 0)
    assert action != weak
    runner = FastRunner(0, state.data, deck_key="b_blue", raise_errors=True)
    runner._state = runner._ctrl.state = state
    runner._ctrl.phase = GamePhase.HAND_PLAY
    runner._sub_phase = SubPhase.CHOOSE_ACTION
    runner.step(action)
    assert not runner.done and runner.sub_phase == SubPhase.SHOP


@pytest.mark.parametrize("ante", [1, 5])
def test_confirmation_is_independent_and_only_used_for_early_switches(ante):
    state = last_hand()
    state.round_resets.ante = ante
    state.hands["Straight Flush"].update(level=20, chips=1500, mult=20)
    agent = HeuristicAgent(shop_policy="search")
    weak = ActionRange.PLAY_SUBSET_START + subset_index((0,))
    mask = compute_action_mask(state, SubPhase.CHOOSE_ACTION)
    before = pickle.dumps(state)
    search = BlindRollout(repeat=True, all_blinds=True, confirm_early=True)
    action = search.select(agent, state, mask, weak, 0)
    assert action != weak and mask[action]
    assert pickle.dumps(state) == before
    if ante == 1:
        confirmed = search.last_decision["confirmation"]
        assert confirmed and all(len(outcomes) == 6 for _, _, outcomes in confirmed)
        assert all(won for won, _ in next(outcomes for _, candidate, outcomes in confirmed if candidate == action))
    else:
        assert search.last_decision["confirmation"] is None


def test_minimum_ante_preserves_early_policy_then_allows_a_real_clear():
    from pylatro_agent.training.fast_runner import FastRunner
    from pylatro_cli.controller import GamePhase

    state = last_hand()
    state.round_resets.ante = 4
    state.hands["Straight Flush"].update(level=20, chips=1500, mult=20)
    agent = HeuristicAgent(shop_policy="search")
    weak = ActionRange.PLAY_SUBSET_START + subset_index((0,))
    mask = compute_action_mask(state, SubPhase.CHOOSE_ACTION)
    search = BlindRollout(repeat=True, all_blinds=True, min_ante=5)
    before = pickle.dumps(state)
    assert search.select(agent, state, mask, weak, 0) == weak
    assert search.last_decision is None
    assert pickle.dumps(state) == before

    state.round_resets.ante = 5
    action = search.select(agent, state, mask, weak, 0)
    assert action != weak and mask[action]
    runner = FastRunner(0, state.data, deck_key="b_blue", raise_errors=True)
    runner._state = runner._ctrl.state = state
    runner._ctrl.phase = GamePhase.HAND_PLAY
    runner._sub_phase = SubPhase.CHOOSE_ACTION
    runner.step(action)
    assert not runner.done and runner.sub_phase == SubPhase.SHOP


@pytest.mark.parametrize("ante", [0, 9])
def test_minimum_ante_rejects_values_outside_the_evaluated_game(ante):
    with pytest.raises(ValueError, match="min_ante"):
        BlindRollout(min_ante=ante)
