import pickle
from copy import deepcopy

import pytest
from pylatro import add_joker,load_game_data
from pylatro.rng import PseudorandomState
from pylatro_agent.constants import ActionRange,SubPhase
from pylatro_agent.heuristic import HeuristicAgent
from pylatro_agent.heuristic_trading import trading_discard,trading_income
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
    add_joker(state,'j_trading')
    add_joker(state,'j_banner')
    return runner,HeuristicAgent(shop_policy='search',grow_scalers=True)


def decision(runner,agent):
    state=runner.state
    probe=deepcopy(state,{id(state.data):state.data})
    play=tuple(sorted(agent._cached_best_hand(probe,probe.hand_cards)))
    score=agent._estimate_hand_score(probe,play)
    return trading_discard(agent,state,runner.compute_mask(),play,agent._get_blind_target(state)-runner.round_score,score)


def test_single_discard_earns_money_thins_deck_and_retains_a_real_clear():
    runner,agent=prepared()
    state=runner.state
    before=pickle.dumps(state)
    action=decision(runner,agent)
    assert action is not None and runner.compute_mask()[action]
    assert pickle.dumps(state)==before
    dollars,size=state.dollars,len(state.deck_cards)
    runner.step(action)
    assert state.dollars==dollars+3
    assert len(state.deck_cards)==size-1
    assert state.current_round.discards_used==1
    assert decision(runner,agent) is None
    action=agent.select_action(state,runner.sub_phase,runner.compute_mask(),round_score=runner.round_score)
    runner.step(action)
    assert not runner.done and runner.sub_phase==SubPhase.SHOP


def test_trading_choice_does_not_use_future_draw_order_or_rng():
    runner,agent=prepared()
    expected=decision(runner,agent)
    clone=deepcopy(runner,{id(runner.state.data):runner.state.data})
    clone.state.draw_pile.reverse()
    clone.state.pseudorandom=PseudorandomState('unknown_future')
    assert decision(clone,HeuristicAgent(shop_policy='search',grow_scalers=True))==expected


@pytest.mark.parametrize('blocker',['j_green_joker','j_ramen','j_blackboard'])
def test_trading_avoids_permanent_discard_costs_and_unknown_held_card_effects(blocker):
    runner,agent=prepared()
    add_joker(runner.state,blocker)
    assert trading_income(runner.state)==0
    assert decision(runner,agent) is None


def test_trading_never_removes_protected_or_upgraded_cards():
    runner,agent=prepared()
    for card in runner.state.hand_cards[5:]:card.seal='Blue'
    assert decision(runner,agent) is None
