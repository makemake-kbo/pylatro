from copy import deepcopy

from pylatro import create_run_state, load_game_data
from pylatro.blind import skip_blind
from pylatro_agent.constants import ActionRange as AR
from pylatro_agent.training.fast_runner import FastRunner


def investment_run():
    runner = FastRunner(0, load_game_data(), deck_key='b_blue', raise_errors=True)
    state = runner.state
    state.round_resets.blind_tags['Small'] = 'tag_investment'
    state.round_resets.blind_choices['Boss'] = 'bl_hook'
    state.hands['High Card']['chips'] = 1000
    state.hands['High Card']['mult'] = 1000
    runner.step(AR.BLIND_SKIP)
    return runner


def test_investment_waits_through_big_blind_then_pays_at_boss_settlement():
    runner = investment_run()
    control = deepcopy(runner, {id(runner.state.data): runner.state.data})
    control.state.tags.clear()
    for trial in (runner, control):
        trial.step(AR.BLIND_PLAY)
        trial.step(AR.PLAY_SUBSET_START)
    assert runner.state.dollars == control.state.dollars
    assert runner.state.tags == ['tag_investment']
    for trial in (runner, control):
        trial.state.dollars = 4
        trial.step(AR.SHOP_LEAVE)
        trial.step(AR.BLIND_PLAY)
        trial.step(AR.PLAY_SUBSET_START)
    assert runner.state.round_resets.ante == 2
    assert runner.state.dollars == control.state.dollars + 25 == 38
    assert runner.state.current_round.round_dollars == control.state.current_round.round_dollars + 25
    assert runner.state.tags == []


def test_losing_to_the_boss_does_not_pay_investment():
    runner = FastRunner(0, load_game_data(), deck_key='b_blue', raise_errors=True)
    state = runner.state
    state.blind_on_deck = 'Big'
    state.round_resets.blind_tags['Big'] = 'tag_investment'
    state.round_resets.blind_choices['Boss'] = 'bl_hook'
    runner.step(AR.BLIND_SKIP)
    before = state.dollars
    runner.step(AR.BLIND_PLAY)
    state.current_round.hands_left = 1
    runner.step(AR.PLAY_SUBSET_START)
    assert runner.done and not runner.won
    assert state.dollars == before and state.tags == ['tag_investment']


def test_double_tags_copy_investment_and_are_consumed():
    state = create_run_state('double_investment', deck_key='b_blue')
    state.tags = ['tag_double', 'tag_juggle', 'tag_double']
    state.round_resets.blind_tags['Small'] = 'tag_investment'
    skip_blind(state)
    assert state.tags == ['tag_juggle', 'tag_investment', 'tag_investment', 'tag_investment']
    assert state.skips == 1 and state.blind_on_deck == 'Big'


def test_taking_another_double_tag_preserves_held_double_tags():
    state = create_run_state('double_double', deck_key='b_blue')
    state.tags = ['tag_double']
    state.round_resets.blind_tags['Small'] = 'tag_double'
    skip_blind(state)
    assert state.tags == ['tag_double', 'tag_double']


def test_copied_economy_tags_apply_the_cap_to_each_doubling():
    state = create_run_state('double_economy', deck_key='b_blue')
    state.tags = ['tag_double']
    state.dollars = 25
    state.round_resets.blind_tags['Small'] = 'tag_economy'
    skip_blind(state)
    assert state.dollars == 90 and state.tags == []


def test_multiple_investments_each_pay_once_and_preserve_other_tags():
    runner = investment_run()
    runner.state.tags = ['tag_investment', 'tag_juggle', 'tag_investment']
    for action in (AR.BLIND_PLAY, AR.PLAY_SUBSET_START, AR.SHOP_LEAVE, AR.BLIND_PLAY):
        runner.step(action)
    control = deepcopy(runner, {id(runner.state.data): runner.state.data})
    control.state.tags = ['tag_juggle']
    for trial in (runner, control):
        trial.step(AR.PLAY_SUBSET_START)
    assert runner.state.dollars == control.state.dollars + 50
    assert runner.state.tags == ['tag_juggle']
