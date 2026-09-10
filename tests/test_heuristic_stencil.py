import pickle

from pylatro import add_joker, create_run_state
from pylatro.pool import create_card_spec
from pylatro.runtime import sell_joker
from pylatro.shop import buy_shop_card
from pylatro_agent.constants import ActionRange as AR, SubPhase
from pylatro_agent.heuristic import HeuristicAgent
from pylatro_agent.heuristic_shop_search import ShopSearch
from pylatro_agent.heuristic_stencil import acquisition_plan
from pylatro_agent.masks import compute_action_mask


def prepared():
    state = create_run_state('compound_stencil', deck_key='b_blue')
    state.dollars = 12
    state.round_resets.ante = 5
    for key in ('j_banner', 'j_half', 'j_joker', 'j_sly'):
        add_joker(state, key)
    state.shop.cards = [create_card_spec(state, 'Joker', forced_key='j_stencil')]
    state.shop.boosters = []
    state.shop.vouchers = []
    return state, HeuristicAgent(shop_policy='search', grow_scalers=True), ShopSearch()


def plan(search, state, agent):
    score, value = search.purchase_output(state, agent)
    return acquisition_plan(search, state, compute_action_mask(state, SubPhase.SHOP),
                            agent, score, value, False, 0, 0)


def test_compound_purchase_preserves_state_then_reaches_a_stronger_portfolio():
    state, agent, search = prepared()
    before = pickle.dumps(state)
    original, _ = search.purchase_output(state, agent)
    gain, action = plan(search, state, agent)
    assert gain > 0.03 and AR.SHOP_SELL_JOKER_START <= action <= AR.SHOP_SELL_JOKER_END
    assert pickle.dumps(state) == before
    for _ in range(8):
        mask = compute_action_mask(state, SubPhase.SHOP)
        # This experimental planner is deliberately not installed in V68.
        _, action = plan(search, state, agent)
        if action < 0:
            # No additional sale improves the acquisition; execute the offer.
            action = AR.SHOP_BUY_START
        assert mask[action]
        if AR.SHOP_SELL_JOKER_START <= action <= AR.SHOP_SELL_JOKER_END:
            sell_joker(state, action - AR.SHOP_SELL_JOKER_START)
        elif action == AR.SHOP_BUY_START:
            buy_shop_card(state, 0)
            break
        else:
            raise AssertionError(f'Acquisition stalled at {action}')
    assert any(j.center_key == 'j_stencil' for j in state.jokers)
    improved, _ = search.purchase_output(state, agent)
    assert improved > original * 1.2


def test_plan_requires_sellable_cards_and_an_affordable_offer():
    state, agent, search = prepared()
    for joker in state.jokers:
        joker.eternal = True
    assert plan(search, state, agent)[1] == -1
    for joker in state.jokers:
        joker.eternal = False
    state.shop.cards[0].cost = 1000
    assert plan(search, state, agent)[1] == -1


def test_plan_does_not_sell_a_strong_portfolio_for_stencil():
    state, agent, search = prepared()
    state.jokers.clear()
    for key in ('j_banner', 'j_half', 'j_cavendish', 'j_card_sharp'):
        add_joker(state, key)
    assert plan(search, state, agent)[1] == -1
