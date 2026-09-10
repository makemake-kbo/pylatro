import pickle
from copy import deepcopy

from pylatro import add_joker, create_run_state, start_blind
from pylatro.models import PlayingCard
from pylatro_agent.heuristic import HeuristicAgent
from pylatro_agent.heuristic_continuation import next_hand_score


def continuation_state():
    state = create_run_state("continuation", deck_key="b_blue")
    start_blind(state, "Small")
    state.deck_cards = [PlayingCard(front_key="C_8", suit="Clubs", rank="8") for _ in range(20)]
    state.hand_cards = list(state.deck_cards[:8])
    state.draw_pile = list(state.deck_cards[8:])
    state.discard_pile = []
    add_joker(state, "j_half")
    add_joker(state, "j_blue_joker")
    return state


def test_continuation_accounts_for_card_sharp_activation_and_preserves_live_state():
    state = continuation_state()
    add_joker(state, "j_card_sharp")
    agent = HeuristicAgent(shop_policy="search")
    before = pickle.dumps(state)
    opening = agent._estimate_hand_score(state, (0, 1))
    following = next_hand_score(agent, state, (0, 1))
    assert following > opening * 2
    assert pickle.dumps(state) == before


def test_continuation_is_independent_of_future_draw_order_and_absolute_card_ids():
    state = continuation_state()
    agent = HeuristicAgent(shop_policy="search")
    expected = next_hand_score(agent, state, (0, 1))
    clone = deepcopy(state, {id(state.data): state.data})
    clone.draw_pile.reverse()
    for card in clone.deck_cards:
        card.reward_uid += 10001
    assert next_hand_score(agent, clone, (0, 1)) == expected


def test_final_hand_has_no_continuation_value():
    state = continuation_state()
    state.current_round.hands_left = 1
    assert next_hand_score(HeuristicAgent(shop_policy="search"), state, (0, 1)) == 0


def test_rare_leveled_hand_is_not_extrapolated_across_all_remaining_hands():
    state = create_run_state("rare_continuation", deck_key="b_blue")
    start_blind(state, "Small")
    by_key = {c.front_key: c for c in state.deck_cards}
    state.hand_cards = [by_key[k] for k in ("C_K", "D_K", "H_K", "C_Q", "D_Q", "S_2", "H_4", "C_7")]
    held = {c.reward_uid for c in state.hand_cards}
    state.draw_pile = [c for c in state.deck_cards if c.reward_uid not in held]
    state.discard_pile = []
    state.hands["Full House"].update(level=20, chips=500, mult=50)
    agent = HeuristicAgent(shop_policy="search")
    opening = agent._estimate_hand_score(state, (0, 1, 2, 3, 4))
    following = next_hand_score(agent, state, (0, 1, 2, 3, 4))
    assert following < opening * 0.4


def test_scoreless_mouth_play_preserves_a_straight_draw():
    from pylatro_agent.action import ActionType, decode_action
    from pylatro_agent.constants import SubPhase
    from pylatro_agent.heuristic_continuation import locked_straight_redraw
    from pylatro_agent.heuristic_growth import pad_scoring_hand
    from pylatro_agent.masks import compute_action_mask
    from pylatro_agent.subset_actions import subset_indices

    state = create_run_state("mouth_redraw", deck_key="b_blue")
    state.round_resets.ante = 4
    state.round_resets.blind_choices["Boss"] = "bl_mouth"
    state.starting_params.hand_size = 9
    start_blind(state, "Boss")
    state.mouth_only_hand = "Straight"
    state.current_round.hands_left = 3
    state.current_round.discards_left = 0
    state.hands["Straight"].update(level=3, chips=90, mult=12)
    by_key = {c.front_key: c for c in state.deck_cards}
    state.hand_cards = [by_key[k] for k in ("S_A", "D_A", "S_Q", "H_Q", "H_J", "H_T", "H_9", "D_7", "H_3")]
    held = {c.reward_uid for c in state.hand_cards}
    state.draw_pile = [c for c in state.deck_cards if c.reward_uid not in held]
    state.discard_pile = []
    agent = HeuristicAgent(shop_policy="search")
    mask = compute_action_mask(state, SubPhase.CHOOSE_ACTION)
    best = tuple(sorted(agent._cached_best_hand(state, state.hand_cards)))
    greedy = pad_scoring_hand(agent, state, mask, best)
    assert agent._estimate_hand_score(state, greedy) == 0
    before = pickle.dumps(state)
    redraw = locked_straight_redraw(agent, state, mask)
    assert next_hand_score(agent, state, redraw) > next_hand_score(agent, state, greedy)
    action = decode_action(agent.select_action(state, SubPhase.CHOOSE_ACTION, mask, round_score=0))
    assert action.action_type == ActionType.PLAY_SUBSET
    assert subset_indices(action.index) == redraw
    assert pickle.dumps(state) == before
