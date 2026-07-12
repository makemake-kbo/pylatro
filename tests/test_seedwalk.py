"""Tests for the interactive seed-walk engine (pylatro.seedwalk)."""

from __future__ import annotations

import json

import pytest

from pylatro.data import load_game_data
from pylatro.seedsearch import SpecError, check_seed, parse_spec, unlock_data
from pylatro.seedwalk import SeedWalk
from pylatro_cli.seedsearch import _strip_jsonc  # comment stripper used by the CLI

# Documented seed: the ante-1 small-blind shop sells Photograph and Hanging Chad
# (see docs/seed_search.md).
PHOTOCHAD_SEED = "5W7A7UGZ"
# Skipping this seed's ante-1 small blind grants a Charm tag (a Mega Arcana pack).
CHARM_SKIP_SEED = "FNFFVJ2S"


@pytest.fixture(scope="module")
def data():
    # Share one unlocked GameData copy across the module so each SeedWalk skips
    # the per-instance unlock deepcopy.
    return unlock_data(load_game_data())


def walk(data, seed=PHOTOCHAD_SEED, **kw):
    return SeedWalk(seed, data=data, unlock_all=False, **kw)


# ---------------------------------------------------------------------------
# Position / phantom walk


def test_starts_at_ante1_blind_select(data):
    w = walk(data)
    assert w.ante == 1
    assert w.on_deck == "Small"
    assert not w.in_shop


def test_beat_blind_opens_shop(data):
    w = walk(data)
    w.beat_blind()
    assert w.in_shop
    assert w.shop_blind == "small"
    assert w.shop_ante == 1
    assert w.roll == 0
    keys = {c.center_key for c in w.state.shop.cards}
    assert {"j_photograph", "j_hanging_chad"} <= keys


def test_leave_shop_advances_to_next_blind(data):
    w = walk(data)
    w.beat_blind()
    w.leave_shop()
    assert not w.in_shop
    assert w.on_deck == "Big"
    assert w.ante == 1


def test_beating_boss_advances_ante(data):
    w = walk(data)
    for _ in range(3):  # Small, Big, Boss
        w.beat_blind()
        w.leave_shop()
    assert w.ante == 2
    assert w.on_deck == "Small"


def test_skip_returns_tag_and_advances(data):
    w = walk(data)
    tag = w.skip_blind()
    assert tag.startswith("tag_")
    assert w.on_deck == "Big"
    with pytest.raises(AssertionError):
        # Skipping does not open a shop.
        w.reroll_shop()


def test_cannot_skip_boss(data):
    w = walk(data)
    w.beat_blind()
    w.leave_shop()  # Big on deck
    w.beat_blind()
    w.leave_shop()  # Boss on deck
    assert w.on_deck == "Boss"
    with pytest.raises(ValueError):
        w.skip_blind()


# ---------------------------------------------------------------------------
# Parity with the seed searcher


def test_shop_matches_searcher(data):
    """The phantom walk's ante-1 small shop matches what the searcher checks."""
    w = walk(data)
    w.beat_blind()
    spec = parse_spec({"ante1": {"small": {"shop": {"contains": ["photo", "chad"]}}}}, data)
    assert check_seed(spec, PHOTOCHAD_SEED, data) is not None


def test_voucher_schedule_matches_forward_walk(data):
    w = walk(data)
    schedule = w.voucher_schedule(horizon=4)
    antes = [a for a, _ in schedule]
    assert antes == [1, 2, 3, 4]
    # Non-destructive: the real walk is untouched by the preview.
    assert w.ante == 1 and not w.in_shop
    # Ante-1 voucher matches the live state.
    assert schedule[0][1] == w.current_voucher


def test_rolls_until_finds_present_card_at_zero(data):
    w = walk(data)
    w.beat_blind()
    assert w.rolls_until("photo")[1] == 0
    assert w.rolls_until("chad")[1] == 0


def test_rolls_until_is_non_destructive(data):
    w = walk(data)
    w.beat_blind()
    before = [c.center_key for c in w.state.shop.cards]
    w.rolls_until("blueprint", cap=50)
    after = [c.center_key for c in w.state.shop.cards]
    assert before == after and w.roll == 0


def test_rolls_until_unknown_name_raises(data):
    w = walk(data)
    w.beat_blind()
    with pytest.raises(SpecError):
        w.rolls_until("not-a-real-joker-xyz")


# ---------------------------------------------------------------------------
# Buy blocking


def test_buying_joker_blocks_future_rolls(data):
    w = walk(data)
    w.beat_blind()
    idx = next(i for i, c in enumerate(w.state.shop.cards) if c.center_key == "j_photograph")
    w.buy_card(idx)
    # Photograph is now owned, so it can no longer roll (no Showman in play).
    assert w.rolls_until("photo", cap=300)[1] is None
    assert any(j.center_key == "j_photograph" for j in w.state.jokers)


# ---------------------------------------------------------------------------
# Report / pinning


def test_pinned_report_roundtrips_through_searcher(data):
    w = walk(data)
    w.beat_blind()
    for card in list(w.state.shop.cards):
        if card.center_key in ("j_photograph", "j_hanging_chad"):
            w.pin_shop_card(card.center_key)
    spec_dict = w.report.to_spec()
    assert spec_dict == {"ante1": {"small": {"shop": {"contains": ["j_hanging_chad", "j_photograph"]}}}}
    # The generated spec matches the very seed it was walked from.
    spec = parse_spec(spec_dict, data)
    assert check_seed(spec, PHOTOCHAD_SEED, data) is not None


def test_report_text_is_loadable_spec(data):
    """The // header comments are stripped by the spec loader, leaving valid JSON."""
    w = walk(data)
    w.beat_blind()
    w.pin_shop_card("j_photograph")
    w.pin_boss()
    w.pin_current_voucher()
    text = w.report.to_text()
    spec_dict = json.loads(_strip_jsonc(text))
    spec = parse_spec(spec_dict, data)
    assert check_seed(spec, PHOTOCHAD_SEED, data) is not None


def test_within_n_uses_reroll_depth(data):
    w = walk(data)
    w.beat_blind()
    w.reroll_shop()  # roll counter -> 1 (within_2 in notation)
    card = w.state.shop.cards[0]
    w.pin_shop_card(card.center_key)
    spec = w.report.to_spec()
    shop = spec["ante1"]["small"]["shop"]
    assert "within_2" in shop and card.center_key in shop["within_2"]


def test_skip_pack_pin_roundtrips(data):
    w = walk(data, seed=CHARM_SKIP_SEED)
    tag = w.blind_tag("Small")
    assert w.tag_pack_label(tag) == "mega arcana"
    w.skip_blind()
    pack = w.open_skip_pack("Small")
    assert w.state.pack is not None  # left open for the UI to display
    w.pin_skip("Small", tag, pack=w.tag_pack_label(tag), contains=[c.center_key for c in pack.cards])
    spec = w.report.to_spec()
    assert spec["ante1"]["small"]["skip"]["tag"] == "tag_charm"
    assert check_seed(parse_spec(spec, data), CHARM_SKIP_SEED, data) is not None


def test_remove_pin(data):
    w = walk(data)
    w.beat_blind()
    w.pin_shop_card("j_photograph")
    assert len(w.report.pins) == 1
    w.report.remove(0)
    assert w.report.pins == []
    assert w.report.to_spec() == {}
