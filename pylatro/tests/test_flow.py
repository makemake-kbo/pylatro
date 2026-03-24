from pylatro import add_joker, create_run_state, discard_cards, play_cards, start_blind


def test_start_blind_draws_seeded_opening_hand() -> None:
    state = create_run_state("AAAAAAAA")
    drawn = start_blind(state)

    assert [card.front_key for card in drawn] == ["D_A", "S_4", "S_K", "S_2", "C_4", "S_8", "D_T", "H_5"]
    assert [card.front_key for card in state.hand_cards] == ["D_A", "S_K", "D_T", "S_8", "H_5", "S_4", "C_4", "S_2"]
    assert len(state.draw_pile) == 44
    assert state.current_round.hand_size == 8


def test_discard_redraws_to_hand_and_updates_round_state() -> None:
    state = create_run_state("AAAAAAAA")
    start_blind(state)

    result = discard_cards(state, state.hand_cards[:2])

    assert [card.front_key for card in result.discarded] == ["D_A", "S_K"]
    assert [card.front_key for card in result.drawn] == ["H_2", "S_7"]
    assert [card.front_key for card in state.hand_cards] == ["D_T", "S_8", "S_7", "H_5", "S_4", "C_4", "S_2", "H_2"]
    assert len(state.draw_pile) == 42
    assert len(state.discard_pile) == 2
    assert state.current_round.discards_left == 3
    assert state.current_round.discards_used == 1


def test_play_only_redraws_when_the_hand_empties() -> None:
    state = create_run_state("AAAAAAAA")
    start_blind(state)

    partial = play_cards(state, state.hand_cards[:5])

    assert partial.score.hand_name == "High Card"
    assert partial.score.total == 16
    assert [card.front_key for card in partial.played] == ["D_A", "S_K", "D_T", "S_8", "H_5"]
    assert partial.drawn == []
    assert [card.front_key for card in state.hand_cards] == ["S_4", "C_4", "S_2"]
    assert len(state.draw_pile) == 44
    assert len(state.discard_pile) == 5
    assert state.current_round.hands_left == 3
    assert state.current_round.hands_played == 1

    state = create_run_state("AAAAAAAA")
    start_blind(state)
    full = play_cards(state, state.hand_cards)

    assert full.score.hand_name == "Pair"
    assert [card.front_key for card in full.drawn] == ["H_2", "S_7", "D_Q", "D_6", "C_9", "C_Q", "C_J", "S_T"]
    assert [card.front_key for card in state.hand_cards] == ["C_Q", "D_Q", "C_J", "S_T", "C_9", "S_7", "D_6", "H_2"]
    assert len(state.draw_pile) == 36
    assert state.current_round.hands_left == 3
    assert state.current_round.hands_played == 1


def test_serpent_draws_three_cards_after_the_first_action() -> None:
    state = create_run_state("AAAAAAAA")
    state.round_resets.blind_choices["Boss"] = "bl_serpent"
    state.blind_on_deck = "Boss"
    start_blind(state, "Boss")

    result = discard_cards(state, state.hand_cards[:1])

    assert [card.front_key for card in result.drawn] == ["H_2", "S_7", "D_Q"]
    assert [card.front_key for card in state.hand_cards] == ["S_K", "D_Q", "D_T", "S_8", "S_7", "H_5", "S_4", "C_4", "S_2", "H_2"]
    assert len(state.draw_pile) == 41


def test_boss_draw_rules_match_seeded_face_down_behavior() -> None:
    house = create_run_state("AAAAAAAA")
    house.round_resets.blind_choices["Boss"] = "bl_house"
    house.blind_on_deck = "Boss"
    start_blind(house, "Boss")
    assert [card.face_down for card in house.hand_cards] == [True, True, True, True, True, True, True, True]

    mark = create_run_state("AAAAAAAA")
    mark.round_resets.blind_choices["Boss"] = "bl_mark"
    mark.blind_on_deck = "Boss"
    start_blind(mark, "Boss")
    assert [(card.front_key, card.face_down) for card in mark.hand_cards] == [
        ("D_A", False),
        ("S_K", True),
        ("D_T", False),
        ("S_8", False),
        ("H_5", False),
        ("S_4", False),
        ("C_4", False),
        ("S_2", False),
    ]

    wheel = create_run_state("AAAAAAAA")
    wheel.round_resets.blind_choices["Boss"] = "bl_wheel"
    wheel.blind_on_deck = "Boss"
    start_blind(wheel, "Boss")
    assert [(card.front_key, card.face_down) for card in wheel.hand_cards] == [
        ("D_A", False),
        ("S_K", False),
        ("D_T", False),
        ("S_8", False),
        ("H_5", False),
        ("S_4", False),
        ("C_4", True),
        ("S_2", True),
    ]

    fish = create_run_state("AAAAAAAA")
    fish.round_resets.blind_choices["Boss"] = "bl_fish"
    fish.blind_on_deck = "Boss"
    start_blind(fish, "Boss")
    redraw = play_cards(fish, fish.hand_cards)
    assert [(card.front_key, card.face_down) for card in redraw.drawn] == [
        ("H_2", True),
        ("S_7", True),
        ("D_Q", True),
        ("D_6", True),
        ("C_9", True),
        ("C_Q", True),
        ("C_J", True),
        ("S_T", True),
    ]


def test_cerulean_bell_and_crimson_heart_apply_on_draw() -> None:
    bell = create_run_state("AAAAAAAA")
    bell.round_resets.blind_choices["Boss"] = "bl_final_bell"
    bell.blind_on_deck = "Boss"
    start_blind(bell, "Boss")

    assert [(card.front_key, card.forced_selection) for card in bell.hand_cards] == [
        ("D_A", False),
        ("S_K", False),
        ("D_T", False),
        ("S_8", False),
        ("H_5", True),
        ("S_4", False),
        ("C_4", False),
        ("S_2", False),
    ]

    heart = create_run_state("AAAAAAAA")
    add_joker(heart, "j_joker")
    add_joker(heart, "j_abstract")
    heart.round_resets.blind_choices["Boss"] = "bl_final_heart"
    heart.blind_on_deck = "Boss"
    start_blind(heart, "Boss")

    redraw = play_cards(heart, heart.hand_cards)

    assert [card.front_key for card in redraw.drawn] == ["H_2", "S_7", "D_Q", "D_6", "C_9", "C_Q", "C_J", "S_T"]
    assert [joker.debuff for joker in heart.jokers] == [True, False]
