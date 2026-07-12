"""Tests for the seed-search spec validation and seed simulation."""

from __future__ import annotations

import random

import pytest

from pylatro.data import load_game_data
from pylatro.run import create_run_state
from pylatro.seedsearch import (
    SpecError,
    check_seed,
    parse_spec,
    random_seed,
    search_seeds,
    unlock_data,
)


@pytest.fixture(scope="module")
def data():
    return load_game_data()


# ---------------------------------------------------------------------------
# Spec validation


def test_voucher_requires_earlier_tier(data):
    with pytest.raises(SpecError, match="requires 'Blank'"):
        parse_spec({"ante1": {"voucher": "antimatter"}}, data)


def test_voucher_tier_order_enforced(data):
    with pytest.raises(SpecError, match="requires 'Blank'"):
        parse_spec({"ante1": {"voucher": "antimatter"}, "ante2": {"voucher": "blank"}}, data)
    # correct order parses
    parse_spec({"ante1": {"voucher": "blank"}, "ante2": {"voucher": "antimatter"}}, data)


def test_duplicate_voucher_rejected(data):
    with pytest.raises(SpecError, match="at most once"):
        parse_spec({"ante1": {"voucher": "blank"}, "ante2": {"voucher": "blank"}}, data)


def test_boss_blind_cannot_be_skipped(data):
    with pytest.raises(SpecError, match="cannot be skipped"):
        parse_spec({"ante1": {"boss_blind": {"skip": {}}}}, data)


def test_legendary_never_in_shop(data):
    with pytest.raises(SpecError, match="legendary"):
        parse_spec({"ante1": {"small": {"shop": {"contains": ["perkeo"]}}}}, data)


def test_pack_item_type_mismatch(data):
    with pytest.raises(SpecError, match="cannot appear in a Arcana pack"):
        parse_spec({"ante1": {"big": {"skip": {"pack": "mega arcana", "contains": ["blueprint"]}}}}, data)


def test_skip_pack_size_must_match_tag(data):
    with pytest.raises(SpecError, match="mega Arcana pack, not jumbo"):
        parse_spec({"ante1": {"small": {"skip": {"pack": "jumbo arcana"}}}}, data)


def test_voucher_unbuyable_when_ante1_fully_skipped(data):
    with pytest.raises(SpecError, match="no ante-1 shop"):
        parse_spec(
            {"ante1": {"voucher": "overstock", "small": {"skip": {}}, "big": {"skip": {}}}},
            data,
        )


def test_locked_voucher_rejected_in_locked_mode(data):
    spec = {"ante1": {"voucher": "overstock"}, "ante2": {"voucher": "overstock plus"}}
    parse_spec(spec, data)  # unlocked profile (default): fine
    with pytest.raises(SpecError, match="locked"):
        parse_spec({**spec, "unlocked": False}, data)


def test_unknown_item_suggests_alternatives(data):
    with pytest.raises(SpecError, match="Unknown item"):
        parse_spec({"ante1": {"small": {"shop": {"contains": ["bluepirnt"]}}}}, data)


def test_bad_within_count(data):
    with pytest.raises(SpecError, match=">= 1"):
        parse_spec({"ante1": {"small": {"shop": {"within_0": ["blueprint"]}}}}, data)


def test_voucher_not_a_shop_card(data):
    with pytest.raises(SpecError, match="ante-level 'voucher'"):
        parse_spec({"ante1": {"small": {"shop": {"contains": ["overstock"]}}}}, data)


def test_friendly_name_resolution(data):
    spec = parse_spec(
        {
            "ante1": {
                "bigblind": {"skip": {"pack": "megaarcana", "contains": ["hermit", "perkeo"]}},
                "voucher": "overstock",
            },
            "ante2": {"voucher": "overstockplus", "smallblind": {"shop": {"within_2": ["blueprint"]}}},
        },
        data,
    )
    assert spec.antes[1].voucher == "v_overstock_norm"
    assert spec.antes[2].voucher == "v_overstock_plus"
    skip = spec.antes[1].blinds["Big"].skip
    assert skip is not None and skip.tag_key == "tag_charm"
    assert {item.key for item in skip.pack.contains} == {"c_hermit", "j_perkeo"}


# ---------------------------------------------------------------------------
# Seed simulation against engine ground truth


def test_check_seed_matches_engine_voucher_and_boss(data):
    unlocked = unlock_data(data)
    state = create_run_state("TESTSEED", data=unlocked)
    spec = parse_spec(
        {"ante1": {"voucher": state.current_voucher, "boss": state.round_resets.blind_choices["Boss"]}},
        data,
    )
    match = check_seed(spec, "TESTSEED", data)
    assert match is not None
    other = "v_blank" if state.current_voucher != "v_blank" else "v_hone"
    bad_spec = parse_spec({"ante1": {"voucher": other}}, data)
    assert check_seed(bad_spec, "TESTSEED", data) is None


def test_check_seed_skip_tag_matches_engine(data):
    unlocked = unlock_data(data)
    state = create_run_state("TESTSEED", data=unlocked)
    tag = state.round_resets.blind_tags["Small"]
    spec = parse_spec({"ante1": {"small": {"skip": {"tag": tag}}}}, data)
    assert check_seed(spec, "TESTSEED", data) is not None
    wrong = "tag_coupon" if tag != "tag_coupon" else "tag_boss"
    bad = parse_spec({"ante1": {"small": {"skip": {"tag": wrong}}}}, data)
    assert check_seed(bad, "TESTSEED", data) is None


def test_search_finds_voucher_and_engine_agrees(data):
    spec = parse_spec({"ante1": {"voucher": "overstock"}}, data)
    found = search_seeds(spec, max_seeds=2000, matches=2, rng=random.Random(42), data=data)
    assert found, "expected a match within 2000 seeds (~6% hit rate)"
    unlocked = unlock_data(data)
    for match in found:
        state = create_run_state(match.seed, data=unlocked)
        assert state.current_voucher == "v_overstock_norm"


def test_search_voucher_tier_chain(data):
    spec = parse_spec(
        {"ante1": {"voucher": "overstock"}, "ante2": {"voucher": "overstock plus"}},
        data,
    )
    found = search_seeds(spec, max_seeds=30_000, matches=1, rng=random.Random(7), data=data)
    assert found, "expected a match within 30k seeds (~0.4% hit rate)"
    assert any("v_overstock_plus" in note for note in found[0].notes)


def test_search_skip_pack_contents(data):
    spec = parse_spec(
        {"ante1": {"big": {"skip": {"pack": "mega arcana", "contains": ["hermit"]}}}},
        data,
    )
    found = search_seeds(spec, max_seeds=5000, matches=1, rng=random.Random(11), data=data)
    assert found
    notes = " ".join(found[0].notes)
    assert "tag_charm" in notes
    assert "c_hermit" in notes


def test_search_shop_within_rolls(data):
    spec = parse_spec({"ante1": {"small": {"shop": {"within_4": ["j_joker"]}}}}, data)
    found = search_seeds(spec, max_seeds=2000, matches=1, rng=random.Random(3), data=data)
    assert found
    assert any("j_joker in shop on roll" in note for note in found[0].notes)


def test_known_perkeo_seed(data):
    # Found by search: skipping ante-1 big blind gives a Charm tag whose Mega
    # Arcana pack contains The Soul, which becomes Perkeo.
    spec = parse_spec(
        {"ante1": {"big": {"skip": {"pack": "mega arcana", "contains": ["perkeo"]}}}},
        data,
    )
    match = check_seed(spec, "9LB4Q6CE", data)
    assert match is not None
    assert any("j_perkeo" in note for note in match.notes)


def test_soul_alone_means_any_legendary(data):
    # "soul" = the pack contains The Soul, whatever legendary it becomes.
    spec = parse_spec({"ante1": {"big": {"skip": {"pack": "mega arcana", "contains": ["soul"]}}}}, data)
    assert check_seed(spec, "9LB4Q6CE", data) is not None
    # A specific different legendary must not match.
    other = parse_spec({"ante1": {"big": {"skip": {"pack": "mega arcana", "contains": ["yorick"]}}}}, data)
    assert check_seed(other, "9LB4Q6CE", data) is None


def test_nickname_aliases(data):
    from pylatro.seedsearch import Resolver

    resolver = Resolver(data)
    assert resolver.item("trib").key == "j_triboulet"
    assert resolver.item("chigoat").key == "j_chicot"


def test_full_example_spec_known_seed(data):
    # Found by a 3M-seed search over the reference spec from the feature request.
    spec = parse_spec(
        {
            "ante1": {
                "big": {"skip": {"pack": "mega arcana", "contains": ["hermit"]}},
                "voucher": "overstock",
            },
            "ante2": {"voucher": "overstock plus", "small": {"shop": {"within_2": ["blueprint"]}}},
        },
        data,
    )
    match = check_seed(spec, "47XZSXLH", data)
    assert match is not None
    notes = " ".join(match.notes)
    assert "v_overstock_norm" in notes
    assert "v_overstock_plus" in notes
    assert "j_blueprint" in notes


def test_random_seed_alphabet():
    rng = random.Random(0)
    for _ in range(50):
        seed = random_seed(rng)
        assert len(seed) == 8
        assert "O" not in seed and "0" not in seed
