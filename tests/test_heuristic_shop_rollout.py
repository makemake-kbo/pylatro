import pickle
from copy import deepcopy

from pylatro import add_joker, create_run_state
from pylatro_agent.heuristic import HeuristicAgent
from pylatro_agent.heuristic_shop_rollout import ShopRollout


def test_round_forecast_preserves_state_and_does_not_peek_at_draw_order():
    state = create_run_state("round_forecast", deck_key="b_blue")
    add_joker(state, "j_half")
    add_joker(state, "j_banner")
    original = pickle.dumps(state)
    score = ShopRollout().output(state, HeuristicAgent())
    assert pickle.dumps(state) == original
    other = deepcopy(state, {id(state.data): state.data})
    other.draw_pile.reverse()
    other.deck_cards.reverse()
    assert ShopRollout().output(other, HeuristicAgent()) == score


def test_round_forecast_keeps_useful_discards_for_banner():
    state = create_run_state("banner_forecast", deck_key="b_blue")
    add_joker(state, "j_half")
    base, _ = ShopRollout().output(state, HeuristicAgent())
    add_joker(state, "j_banner")
    improved, _ = ShopRollout().output(state, HeuristicAgent())
    assert improved > base * 1.5
