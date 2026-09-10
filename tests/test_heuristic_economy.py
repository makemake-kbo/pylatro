import pickle

from pylatro import load_game_data
from pylatro_agent.constants import ActionRange as AR
from pylatro_agent.heuristic_economy import EconomyAgent
from pylatro_agent.training.fast_runner import FastRunner


def prepared():
    runner = FastRunner(0, load_game_data(), deck_key='b_blue', raise_errors=True)
    state = runner.state
    state.round_resets.ante = 3
    state.round_resets.blind_tags['Small'] = 'tag_economy'
    state.round_resets.blind_choices['Boss'] = 'bl_hook'
    state.dollars = 25
    state.hands['High Card']['chips'] = 1000
    state.hands['High Card']['mult'] = 100
    return runner, EconomyAgent(shop_policy='search', grow_scalers=True)


def test_safe_economy_skip_uses_actual_double_tag_cash_without_mutating_live_state():
    runner, agent = prepared()
    runner.state.tags = ['tag_double']
    before = pickle.dumps(runner.state)
    action = agent.select_action(runner.state, runner.sub_phase, runner.compute_mask(), round_score=0)
    assert action == AR.BLIND_SKIP
    assert agent._economy_decision['sampled_clears'] == 6
    assert agent._economy_decision['cash_after'] == 90
    assert pickle.dumps(runner.state) == before
    runner.step(action)
    assert runner.state.dollars == 90 and runner.state.tags == []
    assert runner.state.blind_on_deck == 'Big'


def test_weak_build_keeps_the_round_and_shop():
    runner, agent = prepared()
    runner.state.hands['High Card']['chips'] = 5
    runner.state.hands['High Card']['mult'] = 1
    assert agent.select_action(runner.state, runner.sub_phase, runner.compute_mask(), round_score=0) == AR.BLIND_PLAY
    assert not agent._economy_decision['skip']


def test_economy_skip_requires_cash_tag_and_time_to_use_proceeds():
    runner, agent = prepared()
    runner.state.dollars = 24
    assert agent.select_action(runner.state, runner.sub_phase, runner.compute_mask(), round_score=0) == AR.BLIND_PLAY
    runner.state.dollars = 25
    runner.state.round_resets.ante = 8
    assert agent.select_action(runner.state, runner.sub_phase, runner.compute_mask(), round_score=0) == AR.BLIND_PLAY
    runner.state.round_resets.ante = 3
    runner.state.round_resets.blind_tags['Small'] = 'tag_investment'
    assert agent.select_action(runner.state, runner.sub_phase, runner.compute_mask(), round_score=0) == AR.BLIND_PLAY
