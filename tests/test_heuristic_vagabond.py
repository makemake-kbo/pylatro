import pickle

from pylatro import add_consumable,add_joker,load_game_data
from pylatro_agent.action import ActionType,decode_action,encode_action
from pylatro_agent.constants import ActionRange,SubPhase
from pylatro_agent.heuristic_vagabond import (
    VagabondAgent, active_limit, clear_blocked_tarot_slot, defer_cash_mask, farm_tarot,
)
from pylatro_agent.training.fast_runner import FastRunner


def prepared():
    runner=FastRunner(0,load_game_data(),deck_key='b_blue',raise_errors=True)
    runner.step(ActionRange.BLIND_PLAY)
    state=runner.state
    cards={c.front_key:c for c in state.deck_cards}
    state.hand_cards=[cards[key] for key in ('C_2','C_3','C_4','C_5','C_6','D_7','H_8','S_9')]
    ids={c.reward_uid for c in state.hand_cards}
    state.draw_pile=[c for c in state.deck_cards if c.reward_uid not in ids]
    state.discard_pile=[]
    state.dollars=3
    add_joker(state,'j_vagabond')
    add_joker(state,'j_banner')
    return runner,VagabondAgent(shop_policy='search',grow_scalers=True)


def test_money_tarot_is_deferred_without_mutating_mask_or_state():
    runner,_=prepared()
    state=runner.state
    add_consumable(state,'c_hermit')
    mask=runner.compute_mask()
    before=pickle.dumps(state)
    action=encode_action(ActionType.USE_CONSUMABLE_NO_TARGET,0)
    revised=defer_cash_mask(state,mask)
    assert mask[action] and not revised[action]
    assert pickle.dumps(state)==before
    state.current_round.hands_left=2
    assert defer_cash_mask(state,mask) is mask


def test_safe_spare_hand_generates_a_tarot_before_finishing_the_blind():
    runner,agent=prepared()
    state=runner.state
    add_consumable(state,'c_hermit')
    before=pickle.dumps(state)
    assert farm_tarot(agent,state,runner.compute_mask(),300) is not None
    assert pickle.dumps(state)==before
    action=agent.select_action(state,runner.sub_phase,runner.compute_mask(),round_score=0)
    assert decode_action(action).action_type==ActionType.PLAY_SUBSET
    runner.step(action)
    assert state.dollars==3 and len(state.consumables)==2
    assert runner.sub_phase==SubPhase.CHOOSE_ACTION
    for _ in range(40):
        if runner.sub_phase!=SubPhase.CHOOSE_ACTION:break
        action=agent.select_action(state,runner.sub_phase,runner.compute_mask(),round_score=runner.round_score)
        assert runner.compute_mask()[action]
        runner.step(action)
    assert not runner.done and runner.sub_phase==SubPhase.SHOP


def test_generation_stops_above_cash_threshold_or_without_a_safe_finisher():
    runner,agent=prepared()
    state=runner.state
    state.dollars=4
    assert active_limit(state)==4
    state.dollars=5
    assert active_limit(state) is None
    assert farm_tarot(agent,state,runner.compute_mask(),300) is None
    state.dollars=3
    assert farm_tarot(agent,state,runner.compute_mask(),100000) is None
    state.current_round.hands_left=1
    assert farm_tarot(agent,state,runner.compute_mask(),300) is None


def test_full_money_inventory_can_be_cashed_out_instead_of_stalling():
    runner,_=prepared()
    add_consumable(runner.state,'c_hermit')
    add_consumable(runner.state,'c_temperance')
    mask=runner.compute_mask()
    assert defer_cash_mask(runner.state,mask) is mask


def test_blocked_suit_tarot_frees_slot_without_changing_deck_then_generates():
    runner, agent = prepared()
    state = runner.state
    add_consumable(state, 'c_sun')
    add_consumable(state, 'c_world')
    before = pickle.dumps(state)
    cards = [(c.reward_uid, c.front_key, c.suit, c.center_key) for c in state.deck_cards]
    action = clear_blocked_tarot_slot(state, runner.compute_mask())
    assert action is not None and pickle.dumps(state) == before
    runner.step(action)
    assert len(state.consumables) == 1 and state.dollars == 3
    assert cards == [(c.reward_uid, c.front_key, c.suit, c.center_key) for c in state.deck_cards]
    assert state.consumeable_usage_total['tarot'] == 1
    action = farm_tarot(agent, state, runner.compute_mask(), 300)
    assert action is not None
    runner.step(action)
    assert len(state.consumables) == 2


def test_slot_clearing_requires_full_inventory_and_active_generation():
    runner, _ = prepared()
    add_consumable(runner.state, 'c_sun')
    assert clear_blocked_tarot_slot(runner.state, runner.compute_mask()) is None
    add_consumable(runner.state, 'c_world')
    runner.state.dollars = 5
    assert clear_blocked_tarot_slot(runner.state, runner.compute_mask()) is None
    runner.state.dollars = 3
    for card in runner.state.hand_cards:
        card.face_down = True
    assert clear_blocked_tarot_slot(runner.state, runner.compute_mask()) is None


def test_slot_clearing_does_not_apply_a_different_suit():
    runner, _ = prepared()
    add_consumable(runner.state, 'c_sun')
    add_consumable(runner.state, 'c_world')
    runner.state.hand_cards = [c for c in runner.state.hand_cards if c.suit in {'Clubs', 'Diamonds'}]
    assert clear_blocked_tarot_slot(runner.state, runner.compute_mask()) is None
