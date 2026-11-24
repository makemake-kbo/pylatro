"""v20 teacher fixes: Mail-In Rebate / Castle discards pay, and an owned Runner builds Straights."""
from pylatro import add_joker, create_run_state, start_blind
from pylatro.scoring import RANK_TO_ID
from pylatro_agent.constants import ActionRange, SubPhase
from pylatro_agent.heuristic import HeuristicAgent
from pylatro_agent.masks import compute_action_mask
from pylatro_agent.subset_actions import subset_index, subset_indices


def _state(fronts, joker=None):
    state = create_run_state("round_targets", deck_key="b_red")
    start_blind(state, "Small")
    by_key = {c.front_key: c for c in state.deck_cards}
    state.hand_cards = [by_key[k] for k in fronts]
    if joker:
        add_joker(state, joker)
    return state


HAND = ("C_K", "D_K", "H_9", "S_7", "D_5", "H_4", "C_3", "S_2")


def test_rebate_rank_is_discarded_first():
    state = _state(HAND, "j_mail")
    state.current_round.mail_card = {"rank": "9", "id": RANK_TO_ID["9"]}
    worst = HeuristicAgent(fixes="rebate")._find_worst_cards(state, state.hand_cards, max_discard=1)
    assert worst == {2}
    worst_off = HeuristicAgent(fixes="none")._find_worst_cards(state, state.hand_cards, max_discard=1)
    assert worst_off != {2}


def test_topup_adds_free_rebate_card_but_not_the_pair():
    state = _state(HAND, "j_mail")
    state.current_round.mail_card = {"rank": "9", "id": RANK_TO_ID["9"]}
    mask = compute_action_mask(state, SubPhase.CHOOSE_ACTION)
    action = ActionRange.DISCARD_SUBSET_START + subset_index((7,))
    out = HeuristicAgent(fixes="rebate")._income_topup(state, mask, action)
    chosen = set(subset_indices(out - ActionRange.DISCARD_SUBSET_START))
    assert {2, 7} <= chosen and not chosen & {0, 1}
    assert HeuristicAgent(fixes="none")._income_topup(state, mask, action) == action


def test_rebate_rank_in_a_pair_is_kept():
    state = _state(("C_9", "D_9", "H_K", "S_7", "D_5", "H_4", "C_3", "S_2"), "j_mail")
    state.current_round.mail_card = {"rank": "9", "id": RANK_TO_ID["9"]}
    worst = HeuristicAgent(fixes="rebate")._find_worst_cards(state, state.hand_cards, max_discard=1)
    assert not worst & {0, 1}


def test_runner_straight_window_is_kept():
    state = _state(("C_9", "D_8", "H_7", "S_6", "D_K", "H_K", "C_2", "S_Q"), "j_runner")
    agent = HeuristicAgent(fixes="runner")
    assert agent._get_main_hand_type(state) == "Straight"
    worst = agent._find_worst_cards(state, state.hand_cards, max_discard=3)
    assert not worst & {0, 1, 2, 3}


def test_topup_ignores_cards_beyond_the_action_space():
    # 18-card hand: the only rebate-rank card sits in slot 16, which no discard action can address.
    fronts = ("C_K", "D_K", "H_K", "S_K", "C_Q", "D_Q", "H_Q", "S_Q", "C_J", "D_J", "H_J", "S_J",
              "C_8", "D_8", "H_8", "S_8", "H_9", "S_2")
    state = _state(fronts, "j_mail")
    state.current_round.mail_card = {"rank": "9", "id": RANK_TO_ID["9"]}
    mask = compute_action_mask(state, SubPhase.CHOOSE_ACTION)
    action = ActionRange.DISCARD_SUBSET_START + subset_index((0, 1, 2))
    out = HeuristicAgent(fixes="rebate")._income_topup(state, mask, action)
    assert max(subset_indices(out - ActionRange.DISCARD_SUBSET_START)) < 16
