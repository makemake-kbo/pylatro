import pickle

from pylatro import add_joker, load_game_data
from pylatro_agent.constants import ActionRange as AR
from pylatro_agent.heuristic_investment import InvestmentAgent
from pylatro_agent.training.fast_runner import FastRunner


def prepared():
    runner = FastRunner(0, load_game_data(), deck_key='b_blue', raise_errors=True)
    state = runner.state
    state.round_resets.blind_tags['Small'] = 'tag_investment'
    state.round_resets.blind_choices['Boss'] = 'bl_hook'
    add_joker(state, 'j_banner')
    add_joker(state, 'j_half')
    return runner, InvestmentAgent(shop_policy='search', grow_scalers=True)


def test_safe_poor_build_skips_and_records_six_public_boss_clears():
    runner, agent = prepared()
    before = pickle.dumps(runner.state)
    action = agent.select_action(runner.state, runner.sub_phase, runner.compute_mask(), round_score=0)
    assert action == AR.BLIND_SKIP
    assert agent._investment_decision['sampled_clears'] == 6
    assert pickle.dumps(runner.state) == before
    runner.step(action)
    assert runner.state.tags == ['tag_investment'] and runner.state.blind_on_deck == 'Big'


def test_unsafe_build_keeps_the_blind_and_shop_opportunity():
    runner, agent = prepared()
    runner.state.jokers.clear()
    assert agent.select_action(runner.state, runner.sub_phase, runner.compute_mask(), round_score=0) == AR.BLIND_PLAY
    assert not agent._investment_decision['skip']


def test_no_investment_skip_when_reward_is_too_late_or_cash_is_already_available():
    runner, agent = prepared()
    runner.state.round_resets.ante = 8
    assert agent.select_action(runner.state, runner.sub_phase, runner.compute_mask(), round_score=0) == AR.BLIND_PLAY
    runner.state.round_resets.ante = 1
    runner.state.dollars = 25
    assert agent.select_action(runner.state, runner.sub_phase, runner.compute_mask(), round_score=0) == AR.BLIND_PLAY
    runner.state.dollars = 4
    runner.state.round_resets.blind_tags['Small'] = 'tag_double'
    assert agent.select_action(runner.state, runner.sub_phase, runner.compute_mask(), round_score=0) == AR.BLIND_PLAY
