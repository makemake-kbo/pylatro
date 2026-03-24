from pylatro import cash_out, create_run_state, reroll_boss, select_blind, skip_blind


def test_skip_and_select_blind_follow_reference_fixture() -> None:
    state = create_run_state("AAAAAAAA")

    next_blind = skip_blind(state)
    assert next_blind == "Big"
    assert state.round_resets.blind_states == {
        "Small": "Skipped",
        "Big": "Select",
        "Boss": "Upcoming",
    }
    assert state.tags == ["tag_skip"]

    select_blind(state)
    assert state.round_resets.blind["key"] == "bl_big"
    assert state.round_resets.blind_states == {
        "Small": "Skipped",
        "Big": "Current",
        "Boss": "Upcoming",
    }
    assert state.current_round.hands_left == 4
    assert state.current_round.discards_left == 4


def test_boss_reroll_and_cash_out_regenerate_round_structure() -> None:
    state = create_run_state("AAAAAAAA")

    assert reroll_boss(state) == "bl_hook"
    assert state.dollars == -6
    assert state.round_resets.boss_rerolled is True

    state = create_run_state("AAAAAAAA")
    state.round_resets.blind_states["Boss"] = "Defeated"
    old_boss = state.round_resets.blind_choices["Boss"]
    cash_out(state)

    assert old_boss == "bl_manacle"
    assert state.round_resets.ante == 2
    assert state.round_resets.blind_ante == 2
    assert state.round_resets.blind_choices["Boss"] == "bl_house"
    assert state.round_resets.blind_states == {
        "Small": "Upcoming",
        "Big": "Upcoming",
        "Boss": "Upcoming",
    }
    assert state.blind_on_deck == "Small"
    assert state.current_voucher == "v_magic_trick"
    assert state.round_resets.blind_tags == {"Small": "tag_juggle", "Big": "tag_ethereal"}
