import pytest

from pylatro import add_joker, create_run_state, score_hand
from pylatro.models import PlayingCard


@pytest.mark.parametrize("joker", ["j_even_steven", "j_odd_todd"])
@pytest.mark.parametrize("uid", [1, 2, 10001, 10002])
def test_stone_cards_never_trigger_rank_parity_jokers(joker, uid):
    state = create_run_state("stone_rankless")
    stone = PlayingCard(front_key="H_4", suit="Hearts", rank="4", center_key="m_stone", reward_uid=uid)
    before = score_hand(state, [stone])
    add_joker(state, joker)
    after = score_hand(state, [stone])
    assert (after.chips, after.mult, after.total) == (before.chips, before.mult, before.total)


def test_shop_forecast_with_stone_and_odd_todd_is_independent_of_card_id_parity():
    from copy import deepcopy

    from pylatro_agent.heuristic import HeuristicAgent
    from pylatro_agent.heuristic_shop_search import ShopSearch

    state = create_run_state("stone_shop_forecast", deck_key="b_blue")
    state.deck_cards[0].center_key = "m_stone"
    add_joker(state, "j_odd_todd")
    add_joker(state, "j_even_steven")
    clone = deepcopy(state, {id(state.data): state.data})
    for card in clone.deck_cards:
        card.reward_uid += 101
    assert ShopSearch().output(state, HeuristicAgent()) == ShopSearch().output(clone, HeuristicAgent())


def test_fallback_action_does_not_depend_on_or_advance_numpy_rng():
    import pickle

    import numpy as np

    from pylatro_agent.heuristic import HeuristicAgent

    mask = np.array([0, 1, 0, 1, 1], dtype=np.int8)
    before = pickle.dumps(np.random.get_state())
    assert {HeuristicAgent()._random_valid(mask) for _ in range(20)} == {1}
    assert pickle.dumps(np.random.get_state()) == before


def test_search_score_matches_the_joker_order_that_the_runner_will_use():
    import pickle
    from copy import deepcopy

    from pylatro import play_cards, start_blind
    from pylatro_agent.heuristic import HeuristicAgent
    from pylatro_agent.joker_layout import apply_best_joker_order

    state = create_run_state("order_aware_prediction", deck_key="b_blue")
    start_blind(state, "Small")
    add_joker(state, "j_card_sharp")
    bus = add_joker(state, "j_ride_the_bus")
    bus.mult = 20
    state.hands["Pair"]["played_this_round"] = 1
    state.hand_cards = [PlayingCard(front_key="C_8", suit="Clubs", rank="8"),
                        PlayingCard(front_key="H_8", suit="Hearts", rank="8")]
    before = pickle.dumps(state)
    agent = HeuristicAgent(shop_policy="search")
    predicted = agent._estimate_hand_score(state, (0, 1))
    assert pickle.dumps(state) == before
    trial = deepcopy(state, {id(state.data): state.data})
    apply_best_joker_order(trial, (0, 1))
    assert predicted == play_cards(trial, [0, 1]).score.total
    assert predicted > HeuristicAgent()._estimate_hand_score(state, (0, 1)) * 2


def test_card_order_ties_survive_scoring_state_copies():
    from copy import deepcopy

    from pylatro.flow import _card_nominal as flow_nominal
    from pylatro.models import PlayingCard
    from pylatro.scoring import _card_nominal as scoring_nominal

    state = create_run_state("stable_card_order")
    older = PlayingCard(front_key="H_K", suit="Hearts", rank="K")
    newer = PlayingCard(front_key="H_K", suit="Hearts", rank="K", center_key="m_mult")
    for nominal in (flow_nominal, scoring_nominal):
        expected = [nominal(state, card) for card in (older, newer)]
        assert expected[0] > expected[1]
        assert [nominal(state, card) for card in deepcopy([older, newer])] == expected


def test_raised_fist_selects_last_tied_lowest_card_even_when_debuffed():
    state = create_run_state("raised_fist_tie")
    add_joker(state, "j_raised_fist")
    played = PlayingCard(front_key="S_A", suit="Spades", rank="A")
    first = PlayingCard(front_key="H_2", suit="Hearts", rank="2")
    last = PlayingCard(front_key="D_2", suit="Diamonds", rank="2", debuff=True)
    assert score_hand(state, [played], held_hand=[first, last]).mult == 1
    assert score_hand(state, [played], held_hand=[last, first]).mult == 5


def test_fast_card_copy_preserves_deck_aliases_and_isolates_scoring_mutations():
    from copy import deepcopy

    state = create_run_state("copy_aliases")
    state.hand_cards = list(state.deck_cards[:8])
    clone = deepcopy(state, {id(state.data): state.data})
    assert clone.hand_cards[0] is clone.deck_cards[0]
    assert clone.hand_cards[0] is not state.hand_cards[0]
    assert clone.hand_cards[0].reward_uid == state.hand_cards[0].reward_uid
    clone.hand_cards[0].perma_bonus += 100
    clone.hand_cards[0].debuff = True
    assert state.hand_cards[0].perma_bonus == 0
    assert not state.hand_cards[0].debuff


def test_hand_score_cache_refreshes_after_joker_edition_changes():
    from pylatro import add_joker, create_run_state, select_blind, start_blind
    from pylatro_agent.heuristic import HeuristicAgent

    state = create_run_state("edition_cache", deck_key="b_blue")
    select_blind(state, "Small")
    start_blind(state, "Small")
    joker = add_joker(state, "j_joker")
    agent = HeuristicAgent()
    indices = tuple(sorted(agent._cached_best_hand(state, state.hand_cards)))
    before = agent._estimate_hand_score(state, indices)
    joker.edition = {"foil": True}
    after = agent._estimate_hand_score(state, indices)
    assert after > before
    assert after == HeuristicAgent()._estimate_hand_score(state, indices)


def test_hand_score_cache_refreshes_after_draw_pile_size_changes():
    from pylatro import add_joker, create_run_state, select_blind, start_blind
    from pylatro_agent.heuristic import HeuristicAgent

    state = create_run_state("blue_cache", deck_key="b_blue")
    select_blind(state, "Small")
    start_blind(state, "Small")
    add_joker(state, "j_blue_joker")
    agent = HeuristicAgent()
    indices = tuple(sorted(agent._cached_best_hand(state, state.hand_cards)))
    before = agent._estimate_hand_score(state, indices)
    state.draw_pile.clear()
    after = agent._estimate_hand_score(state, indices)
    assert after < before
    assert after == HeuristicAgent()._estimate_hand_score(state, indices)
