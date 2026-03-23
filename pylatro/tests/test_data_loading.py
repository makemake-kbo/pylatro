from pylatro import load_game_data


def test_reference_data_counts_and_ordering() -> None:
    data = load_game_data()

    assert len(data.blinds) == 30
    assert len(data.cards) == 52
    assert len(data.centers) == 299
    assert len(data.hands) == 12
    assert len(data.center_pools["Joker"]) == 150

    assert data.center_pools["Joker"][0]["key"] == "j_joker"
    assert data.center_pools["Joker"][-1]["key"] == "j_perkeo"
    assert [deck["key"] for deck in data.center_pools["Back"][:5]] == [
        "b_red",
        "b_blue",
        "b_yellow",
        "b_green",
        "b_black",
    ]
    assert data.center_pools["Voucher"][0]["key"] == "v_overstock_norm"
    assert data.center_pools["Voucher"][-1]["key"] == "v_palette"
    assert [joker["key"] for joker in data.joker_rarity_pools[4]] == [
        "j_caino",
        "j_triboulet",
        "j_yorick",
        "j_chicot",
        "j_perkeo",
    ]
    assert data.hands["Straight Flush"]["played"] == 0
