"""Convert validated Balatro snapshots into the existing RunState model."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from pylatro import get_blind_amount, load_game_data
from pylatro.models import (
    ConsumableInstance,
    JokerInstance,
    PackState,
    PlayingCard,
    RunState,
    ShopCard,
)
from pylatro_agent.constants import SubPhase

from .protocol import DecisionRequest, ProtocolError


def _obj(value: object, field: str) -> dict[str, Any]:
    # Steamodded's bundled rxi/json library encodes an empty Lua table as [],
    # even when the table is semantically an object.
    if value == []:
        return {}
    if not isinstance(value, Mapping):
        raise ProtocolError(f"state.{field} must be an object")
    return {str(key): item for key, item in value.items()}


def _items(value: object, field: str) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ProtocolError(f"state.{field} must be an array")
    return [_obj(item, f"{field}[{index}]") for index, item in enumerate(value)]


def _integer(value: object, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    return default


def _number(value: object, default: float = 0.0) -> float:
    return float(value) if isinstance(value, (int, float)) else default


def _edition(value: object) -> dict[str, bool] | None:
    if isinstance(value, str) and value:
        return {value.removeprefix("e_"): True}
    if isinstance(value, Mapping):
        result = {str(k).removeprefix("e_"): bool(v) for k, v in value.items() if v}
        return result or None
    return None


def _rank(value: object) -> str:
    rank = str(value or "A")
    return {"Ace": "A", "Jack": "J", "Queen": "Q", "King": "K"}.get(rank, rank)


@dataclass(slots=True)
class LiveState:
    state: RunState
    sub_phase: SubPhase
    round_score: int
    hand_ids: list[str]
    joker_ids: list[str]
    consumable_ids: list[str]
    shop_ids: list[str]
    pack_ids: list[str]


class SnapshotAdapter:
    """State adapter that refuses unknown vanilla object keys."""

    def __init__(self) -> None:
        self.data = load_game_data()

    def adapt(self, request: DecisionRequest) -> LiveState:
        raw = request.state
        unsupported = raw.get("unsupported", [])
        if unsupported:
            names = ", ".join(str(item) for item in unsupported)
            raise ProtocolError(f"unsupported gameplay content: {names}")

        state = RunState(
            data=self.data,
            seed=str(raw.get("seed") or request.session_id),
            stake=max(1, _integer(raw.get("stake"), 1)),
            deck_key=str(raw.get("deck_key") or "b_red"),
        )
        if state.deck_key not in self.data.centers:
            raise ProtocolError(f"unknown deck key {state.deck_key!r}")

        state.dollars = _integer(raw.get("dollars"))
        state.bankrupt_at = _integer(raw.get("bankrupt_at"))
        state.interest_cap = _integer(raw.get("interest_cap"), 25)
        state.skips = _integer(raw.get("skips"))
        state.round = _integer(raw.get("round"))
        state.win_ante = _integer(raw.get("win_ante"), 8)
        state.blind_on_deck = str(raw.get("blind_on_deck") or "Small")
        state.blind_disabled = bool(raw.get("blind_disabled", False))
        state.blind_triggered = bool(raw.get("blind_triggered", False))
        state.tags = [str(item) for item in raw.get("tags", [])]

        state.starting_params.hand_size = _integer(raw.get("hand_size"), 8)
        state.starting_params.joker_slots = _integer(raw.get("joker_slots"), 5)
        state.starting_params.consumable_slots = _integer(raw.get("consumable_slots"), 2)
        state.current_round.hand_size = state.starting_params.hand_size
        state.current_round.hands_left = _integer(raw.get("hands_left"))
        state.current_round.discards_left = _integer(raw.get("discards_left"))
        state.current_round.hands_played = _integer(raw.get("hands_played"))
        state.current_round.discards_used = _integer(raw.get("discards_used"))
        state.current_round.reroll_cost = _integer(raw.get("reroll_cost"), 5)
        state.current_round.free_rerolls = _integer(raw.get("free_rerolls"))

        state.round_resets.ante = max(1, _integer(raw.get("ante"), 1))
        state.round_resets.blind_ante = _integer(raw.get("blind_ante"), state.round_resets.ante)
        state.round_resets.hands = _integer(raw.get("round_hands"), state.current_round.hands_left)
        state.round_resets.discards = _integer(raw.get("round_discards"), state.current_round.discards_left)
        state.round_resets.reroll_cost = state.current_round.reroll_cost
        state.round_resets.boss_rerolled = bool(raw.get("boss_rerolled", False))
        state.round_resets.blind_choices.update(
            {str(k): str(v) for k, v in _obj(raw.get("blind_choices", {}), "blind_choices").items()}
        )
        state.round_resets.blind_states.update(
            {str(k): str(v) for k, v in _obj(raw.get("blind_states", {}), "blind_states").items()}
        )
        state.round_resets.blind_tags.update(
            {str(k): str(v) for k, v in _obj(raw.get("blind_tags", {}), "blind_tags").items()}
        )
        blind = _obj(raw.get("blind", {}), "blind")
        if blind:
            blind_key = str(blind.get("key") or "")
            if blind_key and blind_key not in self.data.blinds:
                raise ProtocolError(f"unknown blind key {blind_key!r}")
            canonical = dict(self.data.blinds.get(blind_key, {}))
            canonical.update(blind)
            live_chips = _integer(blind.get("chips"), 0)
            canonical["chips"] = live_chips
            # The unchanged tokenizer computes base(ante, stake) * blind.mult.
            # Express the visible target as an effective multiplier so its
            # existing scalar slot receives the exact live value.
            if live_chips > 0:
                base_chips = get_blind_amount(
                    state.round_resets.ante, min(state.stake, 3)
                )
                if base_chips > 0:
                    canonical["mult"] = live_chips / base_chips
            state.round_resets.blind = canonical

        card_piles = _obj(raw.get("cards", {}), "cards")
        hand_raw = _items(card_piles.get("hand", []), "cards.hand")
        draw_raw = _items(card_piles.get("draw", []), "cards.draw")
        discard_raw = _items(card_piles.get("discard", []), "cards.discard")
        deck_raw = _items(card_piles.get("deck", []), "cards.deck")
        state.hand_cards = [self._playing_card(card) for card in hand_raw]
        state.draw_pile = [self._playing_card(card) for card in draw_raw]
        state.discard_pile = [self._playing_card(card) for card in discard_raw]
        state.deck_cards = [self._playing_card(card) for card in deck_raw]
        if not state.deck_cards:
            state.deck_cards = list(state.hand_cards) + list(state.draw_pile) + list(state.discard_pile)

        jokers_raw = _items(raw.get("jokers", []), "jokers")
        state.jokers = [self._joker(item) for item in jokers_raw]
        state.joker_keys = [joker.center_key for joker in state.jokers]
        consumables_raw = _items(raw.get("consumables", []), "consumables")
        state.consumables = [self._consumable(item) for item in consumables_raw]
        state.consumable_keys = [item.center_key for item in state.consumables]
        state.used_vouchers = {
            str(key): bool(value) for key, value in _obj(raw.get("vouchers", {}), "vouchers").items()
        }
        self._validate_keys(state.used_vouchers, "voucher")

        hands = _obj(raw.get("hands", {}), "hands")
        for name, live_hand in hands.items():
            if name not in state.hands:
                raise ProtocolError(f"unknown poker hand {name!r}")
            if isinstance(live_hand, Mapping):
                state.hands[name].update(live_hand)

        shop = _obj(raw.get("shop", {}), "shop")
        shop_cards = _items(shop.get("cards", []), "shop.cards")
        vouchers = _items(shop.get("vouchers", []), "shop.vouchers")
        boosters = _items(shop.get("boosters", []), "shop.boosters")
        state.shop.cards = [self._shop_card(item) for item in shop_cards]
        state.shop.vouchers = [self._shop_card(item) for item in vouchers]
        state.shop.boosters = [self._shop_card(item) for item in boosters]

        pack_raw = raw.get("pack")
        pack_items: list[dict[str, Any]] = []
        if isinstance(pack_raw, Mapping):
            pack_obj = _obj(pack_raw, "pack")
            pack_items = _items(pack_obj.get("cards", []), "pack.cards")
            booster_key = str(pack_obj.get("booster_key") or "")
            if booster_key:
                self._require_center(booster_key)
            state.pack = PackState(
                booster_key=booster_key,
                state_name=str(pack_obj.get("state_name") or "SHOP"),
                choices_remaining=_integer(pack_obj.get("choices_remaining"), 1),
                cards=[self._shop_card(item) for item in pack_items],
            )

        state.won = bool(raw.get("won", False))
        return LiveState(
            state=state,
            sub_phase={
                "blind_select": SubPhase.BLIND_SELECT,
                "hand_play": SubPhase.CHOOSE_ACTION,
                "shop": SubPhase.SHOP,
                "booster_pack": SubPhase.BOOSTER_PACK,
                "terminal": SubPhase.BLIND_SELECT,
            }[request.phase],
            round_score=_integer(raw.get("round_score")),
            hand_ids=self._ids(hand_raw, "hand"),
            joker_ids=self._ids(jokers_raw, "joker"),
            consumable_ids=self._ids(consumables_raw, "consumable"),
            shop_ids=self._ids(shop_cards + vouchers + boosters, "shop item"),
            pack_ids=self._ids(pack_items, "pack card"),
        )

    @staticmethod
    def _ids(items: list[dict[str, Any]], kind: str) -> list[str]:
        result = []
        for index, item in enumerate(items):
            value = item.get("id")
            if not isinstance(value, str) or not value:
                raise ProtocolError(f"{kind} at index {index} has no stable id")
            result.append(value)
        if len(result) != len(set(result)):
            raise ProtocolError(f"duplicate {kind} ids")
        return result

    def _require_center(self, key: str) -> dict[str, Any]:
        if key not in self.data.centers:
            raise ProtocolError(f"unknown vanilla center key {key!r}")
        return self.data.centers[key]

    def _validate_keys(self, values: Mapping[str, object], kind: str) -> None:
        for key, enabled in values.items():
            if enabled:
                center = self._require_center(key)
                if kind == "voucher" and center.get("set") != "Voucher":
                    raise ProtocolError(f"{key!r} is not a vanilla voucher")

    def _playing_card(self, raw: dict[str, Any]) -> PlayingCard:
        front_key = str(raw.get("front_key") or "")
        if front_key not in self.data.cards:
            raise ProtocolError(f"unknown vanilla playing-card key {front_key!r}")
        front = self.data.cards[front_key]
        center_key = str(raw.get("center_key") or "c_base")
        self._require_center(center_key)
        return PlayingCard(
            front_key=front_key,
            suit=str(raw.get("suit") or front.get("suit") or "Spades"),
            rank=_rank(raw.get("rank") or front.get("value") or front.get("rank")),
            center_key=center_key,
            edition_key=(str(raw["edition"]) if raw.get("edition") else None),
            seal=(str(raw["seal"]) if raw.get("seal") else None),
            perma_bonus=_integer(raw.get("perma_bonus")),
            debuff=bool(raw.get("debuff", False)),
            face_down=bool(raw.get("face_down", False)),
            forced_selection=bool(raw.get("forced_selection", False)),
            times_played=_integer(raw.get("times_played")),
        )

    def _joker(self, raw: dict[str, Any]) -> JokerInstance:
        key = str(raw.get("center_key") or "")
        center = self._require_center(key)
        if center.get("set") != "Joker":
            raise ProtocolError(f"{key!r} is not a vanilla joker")
        values = {
            field: raw.get(field)
            for field in (
                "mult",
                "h_mult",
                "h_x_mult",
                "h_dollars",
                "p_dollars",
                "t_mult",
                "t_chips",
                "x_mult",
                "h_size",
                "d_size",
                "extra",
                "extra_value",
                "type",
                "hands_played_at_create",
                "invis_rounds",
                "caino_xmult",
                "yorick_discards",
                "loyalty_remaining",
                "driver_tally",
                "stone_tally",
                "steel_tally",
                "to_do_poker_hand",
                "blueprint_compat",
                "money",
                "sell_cost",
                "nine_tally",
            )
        }
        integer_fields = {
            "mult",
            "h_mult",
            "h_dollars",
            "p_dollars",
            "t_mult",
            "t_chips",
            "h_size",
            "d_size",
            "extra_value",
            "hands_played_at_create",
            "invis_rounds",
            "yorick_discards",
            "loyalty_remaining",
            "driver_tally",
            "stone_tally",
            "steel_tally",
            "money",
            "sell_cost",
            "nine_tally",
        }
        float_fields = {"h_x_mult", "x_mult", "caino_xmult"}
        for field in integer_fields:
            if values[field] is not None:
                values[field] = _integer(values[field])
            else:
                values.pop(field)
        for field in float_fields:
            if values[field] is not None:
                values[field] = _number(values[field])
            else:
                values.pop(field)
        values = {key: value for key, value in values.items() if value is not None}
        return JokerInstance(
            center_key=key,
            edition=_edition(raw.get("edition")),
            eternal=bool(raw.get("eternal", False)),
            perishable=bool(raw.get("perishable", False)),
            perish_tally=(_integer(raw["perish_tally"]) if raw.get("perish_tally") is not None else None),
            rental=bool(raw.get("rental", False)),
            debuff=bool(raw.get("debuff", False)),
            **values,
        )

    def _consumable(self, raw: dict[str, Any]) -> ConsumableInstance:
        key = str(raw.get("center_key") or "")
        center = self._require_center(key)
        if not center.get("consumeable"):
            raise ProtocolError(f"{key!r} is not a vanilla consumable")
        return ConsumableInstance(
            center_key=key,
            edition=_edition(raw.get("edition")),
            extra_value=_integer(raw.get("extra_value")),
            sell_cost=_integer(raw.get("sell_cost"), 1),
        )

    def _shop_card(self, raw: dict[str, Any]) -> ShopCard:
        key = str(raw.get("center_key") or "")
        center = self._require_center(key)
        card_type = str(raw.get("card_type") or center.get("set") or "")
        return ShopCard(
            center_key=key,
            card_type=card_type,
            cost=_integer(raw.get("cost")),
            base_cost=_integer(raw.get("base_cost"), _integer(raw.get("cost"))),
            front_key=(str(raw["front_key"]) if raw.get("front_key") else None),
            edition=_edition(raw.get("edition")),
            seal=(str(raw["seal"]) if raw.get("seal") else None),
            eternal=bool(raw.get("eternal", False)),
            perishable=bool(raw.get("perishable", False)),
            rental=bool(raw.get("rental", False)),
            shop_voucher=bool(raw.get("shop_voucher", card_type == "Voucher")),
        )
