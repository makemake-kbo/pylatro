import pickle

import pytest

from pylatro import add_joker, create_run_state, start_blind
from pylatro_agent.constants import SubPhase
from pylatro_agent.heuristic import HeuristicAgent
from pylatro_agent.heuristic_draw import sampled_early_discard
from pylatro_agent.masks import compute_action_mask


@pytest.mark.parametrize("type_name,fronts", [
    ("Pair", ("C_K", "D_K", "C_Q", "D_Q", "H_9", "S_2")),
    ("Two Pair", ("C_K", "D_K", "H_K", "C_Q", "D_Q", "H_9")),
    ("Three of a Kind", ("C_K", "D_K", "H_K", "C_Q", "D_Q", "H_9")),
    ("Four of a Kind", ("C_K", "D_K", "H_K", "S_K", "C_Q", "H_9")),
])
def test_requested_hand_type_is_not_changed_by_added_kickers(type_name, fronts):
    state = create_run_state("exact_hand_plan", deck_key="b_blue")
    start_blind(state, "Small")
    by_key = {c.front_key: c for c in state.deck_cards}
    state.hand_cards = [by_key[k] for k in fronts]
    agent = HeuristicAgent(shop_policy="search")
    before = pickle.dumps(state)
    indices = agent._find_type_hand(state, state.hand_cards, type_name)
    assert indices is not None
    assert agent._quick_hand_quality(state, [state.hand_cards[i] for i in indices]) == type_name
    assert pickle.dumps(state) == before


def test_summit_spends_one_card_per_discard_when_ramen_is_live():
    from pylatro_agent.action import ActionType, decode_action
    from pylatro_agent.subset_actions import subset_indices

    state = create_run_state("summit_ramen", deck_key="b_blue")
    state.round_resets.ante = 6
    start_blind(state, "Small")
    for key in ("j_half", "j_mystic_summit", "j_ramen", "j_blue_joker"):
        add_joker(state, key)
    by_key = {c.front_key: c for c in state.deck_cards}
    state.hand_cards = [by_key[k] for k in ("S_K", "H_K", "D_J", "C_9", "S_7", "D_5", "H_3", "C_2")]
    agent = HeuristicAgent(shop_policy="search")
    action = decode_action(agent.select_action(
        state, SubPhase.CHOOSE_ACTION, compute_action_mask(state, SubPhase.CHOOSE_ACTION), round_score=0,
    ))
    assert action.action_type == ActionType.DISCARD_SUBSET
    assert len(subset_indices(action.index)) == 1


@pytest.mark.parametrize("last_hand", [False, True])
def test_costly_discard_keeps_banner_score_but_still_takes_a_final_chance(last_hand):
    state = create_run_state("banner_discard_cost", deck_key="b_blue")
    state.round_resets.ante = 4
    start_blind(state, "Small")
    banner = add_joker(state, "j_banner")
    banner.extra = 3000
    half = add_joker(state, "j_half")
    half.extra["mult"] = 1000
    if last_hand:
        state.current_round.hands_left = 1
    agent = HeuristicAgent(shop_policy="search")
    best = tuple(sorted(agent._cached_best_hand(state, state.hand_cards)))
    score = agent._estimate_hand_score(state, best)
    before = pickle.dumps(state)
    discarded = sampled_early_discard(
        agent, state, compute_action_mask(state, SubPhase.CHOOSE_ACTION), score * 10, score, costly=True,
    )
    assert pickle.dumps(state) == before
    if last_hand:
        assert discarded
    else:
        assert discarded is None


def test_final_chance_search_can_spend_blue_seals_and_runs_in_late_antes():
    state = create_run_state("final_chance_blue", deck_key="b_blue")
    state.round_resets.ante = 7
    start_blind(state, "Small")
    state.current_round.hands_left = 1
    for card in state.hand_cards:
        card.seal = "Blue"
    agent = HeuristicAgent(shop_policy="search")
    best = tuple(sorted(agent._cached_best_hand(state, state.hand_cards)))
    score = agent._estimate_hand_score(state, best)
    before = pickle.dumps(state)
    discarded = sampled_early_discard(
        agent, state, compute_action_mask(state, SubPhase.CHOOSE_ACTION), score * 100, score,
    )
    assert discarded
    assert pickle.dumps(state) == before


@pytest.mark.parametrize("discards", [0, 3])
def test_mouth_opening_preserves_a_repeatable_type_when_activating_summit(discards):
    from pylatro_agent.action import ActionType, decode_action
    from pylatro_agent.subset_actions import subset_indices

    state = create_run_state("mouth_repeatable", deck_key="b_blue")
    state.round_resets.ante = 5
    state.round_resets.blind_choices["Boss"] = "bl_mouth"
    start_blind(state, "Boss")
    for key in ("j_trousers", "j_mystic_summit"):
        add_joker(state, key)
    state.current_round.discards_left = discards
    by_key = {c.front_key: c for c in state.deck_cards}
    fronts = ("D_K", "C_T", "S_6", "C_6", "S_5", "S_4", "D_4", "H_3", "H_2") if discards else (
        "C_J", "H_9", "S_8", "D_8", "D_7", "D_6", "H_5", "D_4", "D_2",
    )
    state.hand_cards = [by_key[k] for k in fronts]
    agent = HeuristicAgent(shop_policy="search")
    action = decode_action(agent.select_action(
        state, SubPhase.CHOOSE_ACTION, compute_action_mask(state, SubPhase.CHOOSE_ACTION), round_score=0,
    ))
    indices = subset_indices(action.index)
    if discards:
        assert action.action_type == ActionType.DISCARD_SUBSET
        assert set(indices).isdisjoint({2, 3, 5, 6})
    else:
        assert action.action_type == ActionType.PLAY_SUBSET
        assert agent._quick_hand_quality(state, [state.hand_cards[i] for i in indices]) in {"Pair", "High Card"}


@pytest.mark.parametrize(
    "joker,fronts,expected", [
        ("j_four_fingers", ("C_2", "C_3", "C_4", "C_5"), "Straight Flush"),
        ("j_shortcut", ("C_2", "H_4", "S_6", "D_8", "H_T"), "Straight"),
        ("j_smeared", ("H_2", "D_3", "H_4", "D_5", "H_6"), "Straight Flush"),
    ],
)
def test_heuristic_hand_names_follow_special_hand_rules(joker, fronts, expected):
    state = create_run_state("special_hand_rules")
    by_key = {c.front_key: c for c in state.deck_cards}
    cards = [by_key[key] for key in fronts]
    agent = HeuristicAgent()
    agent._quick_hand_quality(state, cards)
    add_joker(state, joker)
    assert agent._quick_hand_quality(state, cards) == expected


def test_wild_enhancement_changes_hand_name_without_reusing_normal_hand_cache():
    state = create_run_state("wild_hand_rule")
    by_key = {c.front_key: c for c in state.deck_cards}
    cards = [by_key[key] for key in ("H_2", "H_3", "H_4", "H_5", "S_6")]
    agent = HeuristicAgent()
    assert agent._quick_hand_quality(state, cards) == "Straight"
    cards[-1].center_key = "m_wild"
    assert agent._quick_hand_quality(state, cards) == "Straight Flush"


@pytest.mark.parametrize("ranks,expected", [("AAAAA", "Flush Five"), ("AAAKK", "Flush House")])
def test_heuristic_recognizes_duplicate_rank_flush_hands(ranks, expected):
    from pylatro.models import PlayingCard

    state = create_run_state("duplicate_flush")
    cards = [PlayingCard(front_key=f"H_{rank}", rank=rank, suit="Hearts") for rank in ranks]
    assert HeuristicAgent()._quick_hand_quality(state, cards) == expected


def test_discard_search_preserves_live_state_and_ignores_draw_order():
    state = create_run_state("flush_draw_probe", deck_key="b_blue")
    state.round_resets.blind_choices["Boss"] = "bl_manacle"
    start_blind(state, "Boss")
    by_key = {card.front_key: card for card in state.deck_cards}
    state.hand_cards = [by_key[key] for key in ("C_J", "H_8", "S_7", "C_7", "C_6", "D_4", "C_2")]
    held = {card.reward_uid for card in state.hand_cards}
    state.draw_pile = [card for card in state.deck_cards if card.reward_uid not in held]
    for key in ("j_gluttenous_joker", "j_crafty", "j_clever"):
        add_joker(state, key)
    agent = HeuristicAgent(shop_policy="search")
    mask = compute_action_mask(state, SubPhase.CHOOSE_ACTION)
    play = tuple(sorted(agent._cached_best_hand(state, state.hand_cards)))
    score = agent._estimate_hand_score(state, play)
    original = pickle.dumps(state)
    discarded = sampled_early_discard(agent, state, mask, 600, score)
    assert discarded == (1, 2, 5)
    assert pickle.dumps(state) == original
    state.draw_pile.reverse()
    assert sampled_early_discard(HeuristicAgent(), state, mask, 600, score) == discarded


def test_hand_plan_keeps_an_equally_scoring_existing_pair():
    from pylatro_agent.action import ActionType, decode_action
    from pylatro_agent.heuristic_growth import pad_scoring_hand
    from pylatro_agent.subset_actions import subset_indices

    state = create_run_state("keep_equal_pair", deck_key="b_blue")
    state.round_resets.ante = 2
    start_blind(state, "Big")
    add_joker(state, "j_selzer").extra = 5
    add_joker(state, "j_supernova")
    state.hands["Pair"].update(level=3, chips=40, mult=4, played=1, played_this_round=1)
    by_key = {c.front_key: c for c in state.deck_cards}
    state.hand_cards = [by_key[k] for k in ("C_K", "D_K", "C_J", "D_J", "S_9", "H_5", "C_5", "S_3")]
    held = {c.reward_uid for c in state.hand_cards}
    state.draw_pile = [c for c in state.deck_cards if c.reward_uid not in held]
    agent = HeuristicAgent(shop_policy="search")
    mask = compute_action_mask(state, SubPhase.CHOOSE_ACTION)
    best = tuple(sorted(agent._cached_best_hand(state, state.hand_cards)))
    best = pad_scoring_hand(agent, state, mask, best)
    assert agent._quick_hand_quality(state, [state.hand_cards[i] for i in best]) == "Pair"
    action = decode_action(agent.select_action(state, SubPhase.CHOOSE_ACTION, mask, round_score=0))
    assert action.action_type == ActionType.PLAY_SUBSET
    assert subset_indices(action.index) == best


@pytest.mark.parametrize("seal", [None, "Red"])
def test_pair_draw_keeps_a_live_steel_multiplier(seal):
    from pylatro_agent.action import ActionType, decode_action
    from pylatro_agent.subset_actions import subset_indices

    state = create_run_state("held_steel_pair_draw", deck_key="b_blue")
    state.round_resets.ante = 5
    state.round_resets.blind_choices["Boss"] = "bl_window"
    start_blind(state, "Boss")
    for key in ("j_walkie_talkie", "j_even_steven", "j_splash", "j_odd_todd"):
        add_joker(state, key)
    state.hands["Pair"].update(level=3, chips=40, mult=4, played=10)
    by_key = {c.front_key: c for c in state.deck_cards}
    state.hand_cards = [by_key[k] for k in ("H_9", "D_9", "H_8", "D_8", "C_6", "D_6", "H_5", "C_3")]
    for card in state.hand_cards:
        card.debuff = card.suit == "Diamonds"
    state.hand_cards[4].center_key = "m_steel"
    state.hand_cards[4].seal = seal
    held = {c.reward_uid for c in state.hand_cards}
    state.draw_pile = [c for c in state.deck_cards if c.reward_uid not in held]
    agent = HeuristicAgent(shop_policy="search")
    mask = compute_action_mask(state, SubPhase.CHOOSE_ACTION)
    action = decode_action(agent.select_action(state, SubPhase.CHOOSE_ACTION, mask, round_score=0))
    assert action.action_type in {ActionType.PLAY_SUBSET, ActionType.DISCARD_SUBSET}
    assert 4 not in subset_indices(action.index)
