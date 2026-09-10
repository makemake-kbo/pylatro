import pickle

from pylatro import add_consumable, add_joker, load_game_data
from pylatro_agent.constants import ActionRange, SubPhase
from pylatro_agent.heuristic import HeuristicAgent
from pylatro_agent.heuristic_consumable_rescue import winning_consumable
from pylatro_agent.training.fast_runner import FastRunner


def prepared():
    runner = FastRunner(0, load_game_data(), deck_key='b_blue', raise_errors=True)
    runner.state.round_resets.ante = 2
    runner.step(ActionRange.BLIND_PLAY)
    state = runner.state
    cards = {c.front_key: c for c in state.deck_cards}
    state.hand_cards = [cards[k] for k in ('C_2', 'C_4', 'C_6', 'D_8', 'H_9', 'S_T', 'D_Q', 'H_K')]
    ids = {c.reward_uid for c in state.hand_cards}
    state.draw_pile = [c for c in state.deck_cards if c.reward_uid not in ids]
    state.discard_pile = []
    state.current_round.hands_left = 1
    state.current_round.discards_left = 0
    add_joker(state, 'j_droll')
    add_joker(state, 'j_half')
    add_consumable(state, 'c_judgement')
    add_consumable(state, 'c_moon')
    return runner, HeuristicAgent(shop_policy='search', grow_scalers=True)


def test_suit_tarot_enables_a_real_final_hand_clear_without_mutating_probe_input():
    runner, agent = prepared()
    before = pickle.dumps(runner.state)
    result = winning_consumable(agent, runner.state, runner.compute_mask(), 800)
    assert result is not None and pickle.dumps(runner.state) == before
    action, play, estimate = result
    runner.step(action)
    assert runner.compute_mask()[play]
    runner.step(play)
    assert runner.round_score == estimate >= 800
    assert not runner.done and runner.sub_phase == SubPhase.SHOP


def test_search_does_not_return_a_nonwinning_or_illegal_consumable():
    runner, agent = prepared()
    assert winning_consumable(agent, runner.state, runner.compute_mask(), 1000000) is None
    mask = runner.compute_mask().copy()
    mask[:] = 0
    assert winning_consumable(agent, runner.state, mask, 800) is None
