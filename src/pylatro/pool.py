from __future__ import annotations

from typing import Any

from ._helpers import _as_dict, _calculate_cost, _has_showman, _mark_center_used
from .models import RunState, ShopCard

# Finisher (showdown) bosses appear every 8 antes in the real game, where
# win_ante is fixed at 8. Kept independent of state.win_ante so curriculum
# overrides (win_ante < 8) don't change the boss rotation.
SHOWDOWN_ANTE_INTERVAL = 8


def get_current_pool(
    state: RunState,
    card_type: str,
    rarity: float | None = None,
    legendary: bool | None = None,
    append: str | None = None,
) -> tuple[list[str], str]:
    if card_type == "Joker":
        rarity_roll = (
            rarity
            if rarity is not None
            else float(state.pseudorandom.pseudorandom(f"rarity{state.round_resets.ante}{append or ''}"))
        )
        # Joker rarity from the roll: >0.95 Rare (3), >0.7 Uncommon (2), else
        # Common (1); Legendary (4) only via explicit flag (Soul card, etc.).
        rarity_index = 4 if legendary else 3 if rarity_roll > 0.95 else 2 if rarity_roll > 0.7 else 1
        starting_pool = state.data.joker_rarity_pools[rarity_index]
        pool_key = f"Joker{rarity_index}{'' if legendary else append or ''}"
    else:
        starting_pool = state.data.center_pools[card_type]
        pool_key = f"{card_type}{append or ''}"

    pool: list[str] = []
    pool_size = 0
    for proto in starting_pool:
        add = False
        if card_type == "Enhanced":
            add = True
        elif card_type == "Demo":
            add = bool(proto.get("pos") and proto.get("config"))
        elif card_type == "Tag":
            add = (not proto.get("requires") or state.data.centers.get(proto["requires"], {}).get("discovered")) and (
                not proto.get("min_ante") or proto["min_ante"] <= state.round_resets.ante
            )
        elif not (state.used_jokers.get(proto["key"]) and not _has_showman(state)) and (
            proto.get("unlocked", True) or proto.get("rarity") == 4
        ):
            if proto.get("set") == "Voucher":
                if not state.used_vouchers.get(proto["key"]):
                    add = True
                    for required in proto.get("requires", []):
                        if not state.used_vouchers.get(required):
                            add = False
                    for voucher in state.shop.vouchers:
                        if voucher.center_key == proto["key"]:
                            add = False
            elif proto.get("set") == "Planet":
                config = _as_dict(proto.get("config"))
                add = not config.get("softlock", False) or state.hands[config["hand_type"]]["played"] > 0
            elif proto.get("enhancement_gate"):
                add = any(card.center_key == proto["enhancement_gate"] for card in state.deck_cards)
            else:
                add = True
            # Black Hole / The Soul never appear in normal pools; they only
            # spawn via the rare "soulable" roll in pack/shop generation.
            if proto["name"] in {"Black Hole", "The Soul"}:
                add = False

        if proto.get("no_pool_flag") and state.pool_flags.get(proto["no_pool_flag"]):
            add = False
        if proto.get("yes_pool_flag") and not state.pool_flags.get(proto["yes_pool_flag"]):
            add = False
        if add and not state.banned_keys.get(proto["key"]):
            pool.append(proto["key"])
            pool_size += 1
        else:
            pool.append("UNAVAILABLE")

    if pool_size == 0:
        fallback = {
            "Tarot": "c_strength",
            "Tarot_Planet": "c_strength",
            "Planet": "c_pluto",
            "Spectral": "c_incantation",
            "Joker": "j_joker",
            "Voucher": "v_blank",
            "Tag": "tag_handy",
        }
        pool = [fallback.get(card_type, "j_joker")]
    suffix = "" if legendary else str(state.round_resets.ante)
    return pool, f"{pool_key}{suffix}"


def _pick_pool_key(state: RunState, pool: list[str] | dict[str, Any], pool_key: str) -> str:
    center, _ = state.pseudorandom.pseudorandom_element(pool, state.pseudorandom.pseudoseed(pool_key))
    reroll = 1
    while center == "UNAVAILABLE":
        reroll += 1
        center, _ = state.pseudorandom.pseudorandom_element(
            pool,
            state.pseudorandom.pseudoseed(f"{pool_key}_resample{reroll}"),
        )
    return center


def get_next_voucher_key(state: RunState, from_tag: bool = False) -> str:
    pool, pool_key = get_current_pool(state, "Voucher")
    return _pick_pool_key(state, pool, "Voucher_fromtag" if from_tag else pool_key)


def get_next_tag_key(state: RunState, append: str | None = None) -> str:
    pool, pool_key = get_current_pool(state, "Tag", append=append)
    return _pick_pool_key(state, pool, pool_key)


def get_new_boss(state: RunState) -> str:
    eligible: dict[str, int | bool] = {}
    ante = max(1, state.round_resets.ante)

    # Upstream gates showdown bosses on ante % win_ante, which is always 8 in
    # the real game. Our win_ante can be lowered as a training curriculum
    # (e.g. 5), and that must not pull the finisher pool to earlier antes:
    # showdown bosses only ever appear on antes 8, 16, 24, ... and are the
    # only bosses eligible there.
    for key, blind in state.data.blinds.items():
        boss = blind.get("boss")
        if not boss:
            continue
        if (
            not boss.get("showdown")
            and boss["min"] <= ante
            and (ante % SHOWDOWN_ANTE_INTERVAL != 0 or state.round_resets.ante < 2)
        ) or (
            boss.get("showdown")
            and ante % SHOWDOWN_ANTE_INTERVAL == 0
            and state.round_resets.ante >= 2
        ):
            eligible[key] = True

    for key in list(eligible):
        if state.banned_keys.get(key):
            del eligible[key]

    # Prefer the least-used eligible bosses so the rotation doesn't repeat one
    # until the others have been seen. 100 is just a sentinel above any real
    # use count (bosses start at 0 uses).
    min_use = 100
    for key, uses in state.bosses_used.items():
        if key in eligible:
            eligible[key] = uses
            min_use = min(min_use, uses)

    eligible = {key: value for key, value in eligible.items() if value == min_use}
    _, boss = state.pseudorandom.pseudorandom_element(eligible, state.pseudorandom.pseudoseed("boss"))
    assert isinstance(boss, str)
    state.bosses_used[boss] += 1
    return boss


def get_pack(state: RunState, key: str | None = None, pack_type: str | None = None) -> str:
    if not state.first_shop_buffoon and not state.banned_keys.get("p_buffoon_normal_1"):
        state.first_shop_buffoon = True
        return f"p_buffoon_normal_{int(state.pseudorandom.random_without_seed(1, 2))}"

    cumulative = 0.0
    for proto in state.data.center_pools["Booster"]:
        if (not pack_type or pack_type == proto["kind"]) and not state.banned_keys.get(proto["key"]):
            cumulative += proto.get("weight", 1)

    poll = (
        float(
            state.pseudorandom.pseudorandom(
                state.pseudorandom.pseudoseed(f"{key or 'pack_generic'}{state.round_resets.ante}")
            )
        )
        * cumulative
    )
    current = 0.0
    for proto in state.data.center_pools["Booster"]:
        if state.banned_keys.get(proto["key"]):
            continue
        if pack_type and pack_type != proto["kind"]:
            continue
        weight = proto.get("weight", 1)
        current += weight
        if current >= poll:
            return proto["key"]
    raise RuntimeError("Booster selection failed")


def poll_edition(
    state: RunState,
    key: str = "edition_generic",
    mod: float = 1,
    no_negative: bool = False,
    guaranteed: bool = False,
) -> dict[str, bool] | None:
    # Base spawn rates: negative 0.3%, polychrome 0.6%, holo 2%, foil 4%,
    # checked from rarest to most common. `state.edition_rate` (vouchers) and
    # `mod` scale the non-negative odds. `guaranteed` packs (e.g. a Foil pack)
    # multiply the base rates by 25 so an edition almost always lands.
    edition_poll = float(state.pseudorandom.pseudorandom(state.pseudorandom.pseudoseed(key)))
    if guaranteed:
        if edition_poll > 1 - 0.003 * 25 and not no_negative:
            return {"negative": True}
        if edition_poll > 1 - 0.006 * 25:
            return {"polychrome": True}
        if edition_poll > 1 - 0.02 * 25:
            return {"holo": True}
        if edition_poll > 1 - 0.04 * 25:
            return {"foil": True}
        return None

    if edition_poll > 1 - 0.003 * mod and not no_negative:
        return {"negative": True}
    if edition_poll > 1 - 0.006 * state.edition_rate * mod:
        return {"polychrome": True}
    if edition_poll > 1 - 0.02 * state.edition_rate * mod:
        return {"holo": True}
    if edition_poll > 1 - 0.04 * state.edition_rate * mod:
        return {"foil": True}
    return None


def _apply_joker_stickers(state: RunState, center: dict[str, Any], source: str | None) -> tuple[bool, bool, bool]:
    eternal = False
    perishable = False
    rental = False

    if state.modifiers.get("all_eternal") and center.get("eternal_compat"):
        eternal = True

    if source in {"shop", "pack"}:
        eternal_key = ("packetper" if source == "pack" else "etperpoll") + str(state.round_resets.ante)
        eternal_poll = float(state.pseudorandom.pseudorandom(eternal_key))
        if (
            state.modifiers.get("enable_eternals_in_shop")
            and eternal_poll > 0.7
            and center.get("eternal_compat")
            and not perishable
        ):
            eternal = True
        elif (
            state.modifiers.get("enable_perishables_in_shop")
            and 0.4 < eternal_poll <= 0.7
            and center.get("perishable_compat")
            and not eternal
        ):
            perishable = True

        rental_key = ("packssjr" if source == "pack" else "ssjr") + str(state.round_resets.ante)
        if state.modifiers.get("enable_rentals_in_shop") and float(state.pseudorandom.pseudorandom(rental_key)) > 0.7:
            rental = True

    return eternal, perishable, rental


def create_card_spec(
    state: RunState,
    card_type: str,
    forced_key: str | None = None,
    append: str | None = None,
    *,
    source: str | None = None,
    soulable: bool = False,
) -> ShopCard:
    requested_type = card_type
    # "Soulable" packs have a ~0.3% chance (poll > 0.997) per card to instead
    # spawn the hidden Soul (legendary joker) or Black Hole (level-up-all) card.
    if not forced_key and soulable and not state.banned_keys.get("c_soul"):
        if (
            card_type in {"Tarot", "Spectral", "Tarot_Planet"}
            and not (state.used_jokers.get("c_soul") and not _has_showman(state))
            and float(state.pseudorandom.pseudorandom(f"soul_{card_type}{state.round_resets.ante}")) > 0.997
        ):
            forced_key = "c_soul"
        if (
            card_type in {"Planet", "Spectral"}
            and not (state.used_jokers.get("c_black_hole") and not _has_showman(state))
            and float(state.pseudorandom.pseudorandom(f"soul_{card_type}{state.round_resets.ante}")) > 0.997
        ):
            forced_key = "c_black_hole"

    if card_type == "Base":
        forced_key = "c_base"

    if forced_key and not state.banned_keys.get(forced_key):
        center_key = forced_key
        center = state.data.centers[center_key]
        card_type = str(center["set"]) if center.get("set") not in {None, "Default"} else requested_type
    else:
        pool, pool_key = get_current_pool(state, card_type, append=append)
        center_key = _pick_pool_key(state, pool, pool_key)
        center = state.data.centers[center_key]

    front_key = None
    if card_type in {"Base", "Enhanced"}:
        _, front_key = state.pseudorandom.pseudorandom_element(
            state.data.cards,
            state.pseudorandom.pseudoseed(f"front{append or ''}{state.round_resets.ante}"),
        )
        if isinstance(front_key, dict):
            raise AssertionError("Front selection should return a card key")

    edition = None
    if card_type == "Joker":
        eternal, perishable, rental = _apply_joker_stickers(state, center, source)
        edition = poll_edition(state, key=f"edi{append or ''}{state.round_resets.ante}")
    else:
        eternal = False
        perishable = False
        rental = False

    _mark_center_used(state, center_key)

    return ShopCard(
        center_key=center_key,
        card_type=card_type,
        cost=_calculate_cost(state, center, edition=edition, rental=rental),
        base_cost=int(center.get("cost", 1) or 1),
        front_key=front_key if isinstance(front_key, str) else None,
        edition=edition,
        seal=None,
        eternal=eternal,
        perishable=perishable,
        rental=rental,
    )
