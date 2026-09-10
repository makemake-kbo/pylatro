from pylatro import add_joker, load_game_data
from pylatro.models import PlayingCard
from pylatro_agent.constants import ActionRange, SubPhase
from pylatro_agent.heuristic import HeuristicAgent
from pylatro_agent.training.fast_runner import FastRunner


def test_half_joker_can_redraw_a_third_card_without_losing_its_multiplier():
    from pylatro_agent.heuristic_growth import pad_scoring_hand

    runner = prepared_runner()
    state = runner.state
    add_joker(state, "j_half")
    agent = HeuristicAgent(shop_policy="search")
    pair = (0, 1)
    padded = pad_scoring_hand(agent, state, runner.compute_mask(), pair)
    assert len(padded) == 3
    assert {0, 1}.issubset(padded)
    assert agent._estimate_hand_score(state, padded) >= agent._estimate_hand_score(state, pair)


def test_padding_keeps_a_steel_king_held_when_playing_it_would_reduce_score():
    from pylatro_agent.heuristic_growth import pad_scoring_hand

    runner = prepared_runner()
    state = runner.state
    state.hand_cards[0].center_key = "m_steel"
    add_joker(state, "j_baron")
    half = add_joker(state, "j_half")
    half.extra["mult"] = 100
    agent = HeuristicAgent(shop_policy="search")
    pair = (2, 3)
    padded = pad_scoring_hand(agent, state, runner.compute_mask(), pair)
    assert len(padded) == 3
    assert {0, 1}.isdisjoint(padded)
    assert agent._estimate_hand_score(state, padded) >= agent._estimate_hand_score(state, pair)


def test_main_pair_preference_does_not_reset_bus_when_a_stronger_faceless_play_exists():
    from pylatro_agent.action import decode_action
    from pylatro_agent.subset_actions import subset_indices

    runner = prepared_runner()
    state = runner.state
    state.round_resets.ante = 8
    state.current_round.discards_left = 0
    state.current_round.hands_left = 3
    state.current_round.hands_played = 2
    state.hand_cards = [
        PlayingCard(front_key=f"{s[0]}_{r}", suit=s, rank=r)
        for s, r in [("Clubs", "A"), ("Spades", "Q"), ("Diamonds", "Q"), ("Hearts", "J"),
                     ("Spades", "T"), ("Diamonds", "8"), ("Diamonds", "7"), ("Diamonds", "6"),
                     ("Hearts", "4"), ("Clubs", "2")]
    ]
    for key in ("j_banner", "j_ride_the_bus", "j_baseball", "j_card_sharp"):
        add_joker(state, key)
    bus = next(j for j in state.jokers if j.center_key == "j_ride_the_bus")
    bus.mult = 39
    state.hands["Pair"].update(level=2, chips=25, mult=3, played=20, played_this_round=0)
    agent = HeuristicAgent(shop_policy="search")
    baseline = agent._estimate_hand_score(state, (0,))
    action = agent.select_action(state, runner.sub_phase, runner.compute_mask(), round_score=0)
    played = subset_indices(decode_action(action).index)
    assert agent._estimate_hand_score(state, played) >= baseline
    from pylatro import get_poker_hand_info

    scoring = get_poker_hand_info(state, [state.hand_cards[i] for i in played])[3]
    assert all(c.rank not in {"J", "Q", "K"} for c in scoring)


def prepared_runner():
    runner = FastRunner(0, load_game_data(), deck_key="b_blue", raise_errors=True)
    runner.step(ActionRange.BLIND_PLAY)
    runner.state.hand_cards = [
        PlayingCard(front_key=f"{suit[0]}_{rank}", suit=suit, rank=rank)
        for suit, rank in [
            ("Spades", "K"),
            ("Hearts", "K"),
            ("Clubs", "Q"),
            ("Diamonds", "Q"),
            ("Hearts", "9"),
            ("Clubs", "7"),
            ("Spades", "4"),
            ("Diamonds", "2"),
        ]
    ]
    joker = add_joker(runner.state, "j_green_joker")
    joker.mult = 10
    runner.state.dollars = 25
    return runner


def test_growth_keeps_a_winning_hand_and_gains_permanent_mult():
    greedy = prepared_runner()
    agent = HeuristicAgent()
    greedy.step(agent.select_action(greedy.state, greedy.sub_phase, greedy.compute_mask(), round_score=0))
    assert greedy.sub_phase == SubPhase.SHOP
    greedy_mult = greedy.state.jokers[0].mult

    runner = prepared_runner()
    grower = HeuristicAgent(grow_scalers=True)
    runner.step(grower.select_action(runner.state, runner.sub_phase, runner.compute_mask(), round_score=0))
    assert runner.sub_phase == SubPhase.CHOOSE_ACTION
    assert runner.round_score < 300
    for _ in range(5):
        if runner.sub_phase == SubPhase.SHOP:
            break
        runner.step(
            grower.select_action(runner.state, runner.sub_phase, runner.compute_mask(), round_score=runner.round_score)
        )
    assert runner.sub_phase == SubPhase.SHOP
    assert runner.state.jokers[0].mult > greedy_mult


def test_mail_collects_cash_without_spending_the_winning_pair():
    from pylatro.instances import remove_joker

    runner = prepared_runner()
    remove_joker(runner.state, runner.state.jokers[0])
    add_joker(runner.state, "j_half")
    add_joker(runner.state, "j_blue_joker")
    add_joker(runner.state, "j_mail")
    runner.state.current_round.mail_card = {"rank": "2", "id": 2}
    before = runner.state.dollars
    agent = HeuristicAgent(shop_policy="search")
    action = agent.select_action(runner.state, runner.sub_phase, runner.compute_mask(), round_score=0)
    assert ActionRange.DISCARD_SUBSET_START <= action <= ActionRange.DISCARD_SUBSET_END
    runner.step(action)
    assert runner.state.dollars == before + 5
    assert runner.state.current_round.hands_played == 0
    for _ in range(5):
        if runner.sub_phase == SubPhase.SHOP:
            break
        runner.step(
            agent.select_action(runner.state, runner.sub_phase, runner.compute_mask(), round_score=runner.round_score)
        )
    assert runner.sub_phase == SubPhase.SHOP


def test_verdant_leaf_sells_income_joker_and_preserves_scoring_engine():
    from pylatro_agent.masks import compute_action_mask

    runner = FastRunner(0, load_game_data(), deck_key="b_blue", raise_errors=True)
    state = runner.state
    state.round_resets.ante = 8
    state.blind_on_deck = "Boss"
    state.round_resets.blind_choices["Boss"] = "bl_final_leaf"
    for key in ("j_blue_joker", "j_half", "j_golden"):
        add_joker(state, key)
    runner.step(ActionRange.BLIND_PLAY)
    assert all(card.debuff for card in state.hand_cards)
    full_mask = compute_action_mask(state, SubPhase.CHOOSE_ACTION)
    fast_mask = runner.compute_mask()
    assert (full_mask == fast_mask).all()
    action = HeuristicAgent().select_action(state, runner.sub_phase, fast_mask, round_score=0)
    assert action == ActionRange.SHOP_SELL_JOKER_START + 2
    runner.step(action)
    assert state.blind_disabled
    assert not any(card.debuff for card in state.deck_cards)
    assert state.joker_keys == ["j_blue_joker", "j_half"]
    assert state.current_round.hands_played == 0


def test_square_growth_does_not_replace_a_winning_final_hand():
    from pylatro.instances import remove_joker
    from pylatro_agent.subset_actions import subset_indices

    runner = prepared_runner()
    state = runner.state
    remove_joker(state, state.jokers[0])
    add_joker(state, "j_half")
    add_joker(state, "j_blue_joker")
    add_joker(state, "j_square").extra["chips"] = 200
    state.hands["Pair"].update(level=11, chips=160, mult=12)
    state.round_resets.ante = 8
    state.current_round.hands_left = 1
    state.current_round.discards_left = 0
    agent = HeuristicAgent()
    progress = agent._get_blind_target(state) - 10000
    action = agent.select_action(state, SubPhase.CHOOSE_ACTION, runner.compute_mask(), round_score=progress)
    assert ActionRange.PLAY_SUBSET_START <= action <= ActionRange.PLAY_SUBSET_END
    indices = subset_indices(action - ActionRange.PLAY_SUBSET_START)
    assert len(indices) <= 3
    assert agent._estimate_hand_score(state, indices) >= 10000


def test_baron_keeps_an_immediate_win_over_saving_kings_for_later():
    from pylatro_agent.subset_actions import subset_indices

    runner = prepared_runner()
    state = runner.state
    state.hand_cards[3].rank = "J"
    state.hand_cards[3].front_key = "D_J"
    state.jokers[0].mult = 50
    add_joker(state, "j_baron")
    state.hands["Pair"].update(level=5, chips=70, mult=6)
    state.round_resets.ante = 4
    state.current_round.hands_left = 2
    state.current_round.discards_left = 0
    agent = HeuristicAgent()
    remaining = 4000
    action = agent.select_action(
        state, SubPhase.CHOOSE_ACTION, runner.compute_mask(),
        round_score=agent._get_blind_target(state) - remaining,
    )
    assert ActionRange.PLAY_SUBSET_START <= action <= ActionRange.PLAY_SUBSET_END
    indices = subset_indices(action - ActionRange.PLAY_SUBSET_START)
    assert agent._estimate_hand_score(state, indices) >= remaining


def test_well_funded_growth_uses_a_spare_penultimate_hand():
    runner = prepared_runner()
    runner.state.current_round.hands_left = 2
    agent = HeuristicAgent(grow_scalers=True)
    action = agent.select_action(runner.state, runner.sub_phase, runner.compute_mask(), round_score=0)
    runner.step(action)
    assert runner.sub_phase == SubPhase.CHOOSE_ACTION
    assert runner.state.current_round.hands_left == 1
    action = agent.select_action(
        runner.state, runner.sub_phase, runner.compute_mask(), round_score=runner.round_score,
    )
    runner.step(action)
    assert runner.sub_phase == SubPhase.SHOP


def test_burnt_uses_the_first_discard_for_the_main_pair_without_mutating_state():
    import pickle

    from pylatro import add_joker, create_run_state, start_blind
    from pylatro.flow import discard_cards
    from pylatro_agent.action import ActionType, decode_action
    from pylatro_agent.constants import SubPhase
    from pylatro_agent.heuristic import HeuristicAgent
    from pylatro_agent.heuristic_growth import burnt_discard
    from pylatro_agent.masks import compute_action_mask
    from pylatro_agent.subset_actions import subset_indices

    state = create_run_state("burnt_plan", deck_key="b_blue")
    add_joker(state, "j_burnt")
    start_blind(state, "Small")
    by_key = {c.front_key: c for c in state.deck_cards}
    state.hand_cards = [by_key[k] for k in ("C_K", "D_K", "H_8", "S_8", "C_2", "D_3", "S_4", "H_6")]
    held = {c.reward_uid for c in state.hand_cards}
    state.draw_pile = [c for c in state.deck_cards if c.reward_uid not in held]
    agent = HeuristicAgent(shop_policy="search")
    before = pickle.dumps(state)
    mask = compute_action_mask(state, SubPhase.CHOOSE_ACTION)
    action = decode_action(burnt_discard(agent, state, mask))
    assert action.action_type == ActionType.DISCARD_SUBSET
    assert decode_action(agent.select_action(state, SubPhase.CHOOSE_ACTION, mask, round_score=0)) == action
    indices = subset_indices(action.index)
    assert len(indices) == 2
    assert agent._quick_hand_quality(state, [state.hand_cards[i] for i in indices]) == "Pair"
    assert pickle.dumps(state) == before
    discard_cards(state, indices)
    assert state.hands["Pair"]["level"] == 2
    assert burnt_discard(agent, state, compute_action_mask(state, SubPhase.CHOOSE_ACTION)) is None
