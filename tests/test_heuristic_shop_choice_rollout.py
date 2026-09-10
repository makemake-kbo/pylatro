import pickle
from copy import deepcopy

from pylatro import add_joker, create_run_state
from pylatro.pool import create_card_spec
from pylatro.shop import buy_shop_card
from pylatro_agent.constants import ActionRange
from pylatro_agent.heuristic import HeuristicAgent
from pylatro_agent.heuristic_shop_choice_rollout import choose_purchase, trial_blind
from pylatro_agent.heuristic_shop_capacity_choice import choose_capacity, trial_capacity
from pylatro_agent.heuristic_shop_search import ShopSearch


def portfolios():
    state = create_run_state('purchase_trials', deck_key='b_blue')
    state.round_resets.ante = 2
    state.blind_on_deck = 'Boss'
    state.round_resets.blind_choices['Boss'] = 'bl_needle'
    state.dollars = 20
    add_joker(state, 'j_half')
    state.shop.cards = [create_card_spec(state, 'Joker', forced_key=key) for key in ('j_egg', 'j_banner')]
    result = []
    for index, gain in enumerate((1.0, 0.5)):
        trial = deepcopy(state, {id(state.data): state.data})
        buy_shop_card(trial, index)
        result.append((gain, ActionRange.SHOP_BUY_START + index, trial))
    return result, HeuristicAgent(shop_policy='search', grow_scalers=True)


def test_actual_blind_trials_prefer_a_purchase_that_clears_needle():
    proposals, agent = portfolios()
    before = [pickle.dumps(p[2]) for p in proposals]
    action, results = choose_purchase(agent, proposals[0][:2], proposals)
    assert action == proposals[1][1]
    assert results[0]['clears'] < results[1]['clears'] == 3
    assert [pickle.dumps(p[2]) for p in proposals] == before


def test_perfect_original_purchase_is_kept_without_testing_alternatives():
    proposals, agent = portfolios()
    action, results = choose_purchase(agent, proposals[1][:2], proposals)
    assert action == proposals[1][1] and len(results) == 1
    assert results[0]['clears'] == 3


def test_trial_ignores_original_future_draw_order_and_rng():
    proposals, agent = portfolios()
    state = proposals[1][2]
    expected = trial_blind(agent, state, 0)
    state.draw_pile.reverse()
    state.deck_cards.reverse()
    state.pseudorandom.pseudorandom('unrelated_future')
    assert trial_blind(agent, state, 0) == expected


def test_capacity_shortlist_keeps_actual_survival_ahead_of_projected_value():
    proposals, agent = portfolios()
    before = [pickle.dumps(p[2]) for p in proposals]
    action, results = choose_capacity(ShopSearch(), agent, proposals[0][:2], proposals)
    assert action == proposals[1][1]
    assert results[0]['clears'] < results[1]['clears'] == 3
    assert [pickle.dumps(p[2]) for p in proposals] == before


def test_capacity_continues_past_the_clear_without_changing_live_state():
    proposals, agent = portfolios()
    state = proposals[1][2]
    state.round_resets.ante = 1
    state.round_resets.blind_choices['Boss'] = 'bl_hook'
    before = pickle.dumps(state)
    clear, terminal_score = trial_blind(agent, state, 0)
    capacity_clear, capacity = trial_capacity(state, 0)
    assert clear == capacity_clear == 1 and capacity > terminal_score
    assert pickle.dumps(state) == before
