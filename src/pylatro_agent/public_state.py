"""Public-information v2: an immutable projection, never a copied live world.

Only this module's capture boundary reads an executor. Consumers receive JSON
bytes with an explicit allowlist. Deck membership is a multiset; concealed hand
slots carry no identity. Simulated RNG and card IDs are freshly constructed.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import asdict, dataclass
from random import Random

from pylatro.models import (
    ConsumableInstance,
    CurrentRound,
    JokerInstance,
    PackState,
    PlayingCard,
    RoundResets,
    RunState,
    ShopCard,
    ShopState,
    StartingParams,
)
from pylatro_agent.constants import SubPhase
from pylatro_agent.history import CardSnapshot, OmittedSummary, PlayEvent, PlayHistoryTracker, RoundHistory
from pylatro_agent.masks import compute_action_mask

OBSERVATION_INTERPRETATION = "public-information-v2"
PUBLIC_FIELDS = (
    "stake",
    "deck_key",
    "modifiers",
    "probabilities",
    "bosses_used",
    "pool_flags",
    "used_vouchers",
    "banned_keys",
    "current_boss_streak",
    "base_reroll_cost",
    "edition_rate",
    "joker_rate",
    "tarot_rate",
    "planet_rate",
    "spectral_rate",
    "playing_card_rate",
    "interest_cap",
    "discount_percent",
    "inflation",
    "pack_size",
    "skips",
    "dollars",
    "bankrupt_at",
    "blind_on_deck",
    "tags",
    "hands",
    "win_ante",
    "round",
    "won",
    "interest_amount",
    "perishable_rounds",
    "rental_rate",
    "consumeable_buffer",
    "joker_buffer",
    "max_jokers",
    "starting_deck_size",
    "ecto_minus",
    "tag_tally",
    "hands_played",
    "unused_discards",
    "last_tarot_planet",
    "previous_round",
    "round_bonus",
    "dollar_buffer",
    "last_hand_played",
    "consumeable_usage",
    "consumeable_usage_total",
    "cards_played",
    "first_shop_buffoon",
    "blind_disabled",
    "blind_triggered",
    "blind_prepped",
    "eye_hands",
    "mouth_only_hand",
)
CARD_IDENTITY = (
    "front_key",
    "suit",
    "rank",
    "center_key",
    "edition_key",
    "seal",
    "perma_bonus",
    "played_this_ante",
    "times_played",
)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def card_record(card, *, composition=False):
    if composition:
        return {key: getattr(card, key) for key in CARD_IDENTITY}
    if card.face_down:
        return {"unknown": True, "forced_selection": bool(card.forced_selection)}
    record = asdict(card)
    record.pop("reward_uid")
    return record


def public_history(history):
    rounds = []
    for rnd in history.rounds:
        record = asdict(rnd)
        # Historical captures predate reveal. Conservatively retain unknown
        # identities for face-down cards rather than infer from private data.
        for event in record["events"]:
            if record["key"] and record["key"][-1] in {"Amber Acorn", "bl_final_acorn"}:
                event["joker_keys"] = sorted(event["joker_keys"])
            event["cards"] = [
                asdict(CardSnapshot("", "", "c_base", None, None, 0, False, True)) if card["face_down"] else card
                for card in event["cards"]
            ]
        rounds.append(record)
    return rounds


def restore_history(records):
    tracker = PlayHistoryTracker()
    rounds = []
    for record in records:
        record = deepcopy(record)
        record["key"] = tuple(record["key"]) if record["key"] is not None else None
        record["omitted"] = OmittedSummary(**record["omitted"])
        events = []
        for event in record["events"]:
            event["cards"] = tuple(CardSnapshot(**c) for c in event["cards"])
            event["joker_keys"] = tuple(event["joker_keys"])
            events.append(PlayEvent(**event))
        record["events"] = events
        rounds.append(RoundHistory(**record))
    tracker._rounds = rounds
    tracker._current_key = rounds[-1].key
    return tracker


@dataclass(frozen=True, slots=True)
class PublicGameSnapshot:
    payload: bytes

    @property
    def key(self):
        return hashlib.sha256(self.payload).hexdigest()

    @property
    def values(self):
        return json.loads(self.payload)

    @property
    def sub_phase(self):
        return SubPhase(self.values["sub_phase"])

    @classmethod
    def capture(cls, state, sub_phase, round_score=0, history=None):
        fields = {name: deepcopy(getattr(state, name)) for name in PUBLIC_FIELDS}
        fields.update(
            {
                name: asdict(getattr(state, name))
                for name in ("starting_params", "current_round", "round_resets", "shop")
            }
        )
        # Only this ante's revealed blinds are carried. Future vouchers are
        # preselected privately by the engine, so derive them from visible offers.
        fields["round_resets"]["blind_choices"] = {
            k: v for k, v in fields["round_resets"]["blind_choices"].items() if k in {"Small", "Big", "Boss"}
        }
        fields["jokers"] = [asdict(j) for j in state.jokers]
        fields["joker_order_unknown"] = concealed_joker_order(state, sub_phase)
        if fields["joker_order_unknown"]:
            fields["jokers"].sort(key=canonical)
        owned = {j.center_key for j in state.jokers}
        defaults = CurrentRound()
        for target, joker in (
            ("idol_card", "j_idol"),
            ("mail_card", "j_mail"),
            ("ancient_card", "j_ancient"),
            ("castle_card", "j_castle"),
        ):
            if joker not in owned:
                fields["current_round"][target] = getattr(defaults, target)
        fields["consumables"] = [asdict(c) for c in state.consumables]
        fields["pack"] = asdict(state.pack) if sub_phase == SubPhase.BOOSTER_PACK and state.pack else None
        fields["deck"] = sorted((card_record(c, composition=True) for c in state.deck_cards), key=canonical)
        for name in ("hand_cards", "discard_pile", "play_cards"):
            fields[name] = [card_record(c) for c in getattr(state, name)]
        fields["draw_count"] = len(state.draw_pile)
        fields["sub_phase"] = str(sub_phase)
        fields["round_score"] = round_score
        fields["history"] = public_history(history or PlayHistoryTracker())
        fields["interpretation"] = OBSERVATION_INTERPRETATION
        return cls(canonical(fields).encode())


def concealed_joker_order(state, sub_phase):
    # The engine implements Acorn's shuffle but omits its visual concealment.
    # Public mode conservatively treats order as unknown for the whole blind.
    return (
        sub_phase == SubPhase.CHOOSE_ACTION
        and not state.blind_disabled
        and (state.round_resets.blind or {}).get("name") == "Amber Acorn"
    )


def simulation_state(snapshot, data, sample=0):
    """Construct one public conditional world, without replacement.

    No live RunState or RNG can enter here. The same sample index yields the
    same world for every candidate. Only the simulation executor sees its RNG.
    """
    value = snapshot.values
    seed = "PUBLIC_V2_" + hashlib.sha256(snapshot.payload + str(sample).encode()).hexdigest()
    state = RunState(data=data, seed=seed)
    for key in PUBLIC_FIELDS:
        setattr(state, key, value[key])
    for key, typ in (
        ("starting_params", StartingParams),
        ("current_round", CurrentRound),
        ("round_resets", RoundResets),
    ):
        setattr(state, key, typ(**value[key]))
    state.shop = ShopState(
        **{
            **value["shop"],
            **{key: [ShopCard(**c) for c in value["shop"][key]] for key in ("cards", "vouchers", "boosters")},
        }
    )
    state.pack = (
        PackState(**{**value["pack"], "cards": [ShopCard(**c) for c in value["pack"]["cards"]]})
        if value["pack"]
        else None
    )
    state.jokers = [JokerInstance(**j) for j in value["jokers"]]
    state.joker_keys = [j.center_key for j in state.jokers]
    state.consumables = [ConsumableInstance(**c) for c in value["consumables"]]
    state.consumable_keys = [c.center_key for c in state.consumables]
    state.used_jokers = {key: True for key in state.joker_keys + state.consumable_keys}
    state.current_voucher = state.shop.vouchers[0].center_key if state.shop.vouchers else None
    if state.current_voucher is None:
        from pylatro.pool import get_next_voucher_key

        state.current_voucher = get_next_voucher_key(state)
    pool = [PlayingCard(**c, reward_uid=i + 1) for i, c in enumerate(value["deck"])]
    state.deck_cards = list(pool)
    rng = Random(seed)
    if value["joker_order_unknown"]:
        rng.shuffle(state.jokers)
        state.joker_keys = [joker.center_key for joker in state.jokers]
    known = {}
    # Remove all visibly known identities before assigning any concealed slot.
    for name in ("hand_cards", "discard_pile", "play_cards"):
        for i, record in enumerate(value[name]):
            if record.get("unknown"):
                continue
            match = next((c for c in pool if all(getattr(c, k) == record[k] for k in CARD_IDENTITY)), None)
            if match is None:
                # Destroyed played cards can remain in the visible play area.
                match = PlayingCard(**record, reward_uid=10000 + len(known))
            else:
                pool.remove(match)
                for k, v in record.items():
                    setattr(match, k, v)
            known[name, i] = match
    rng.shuffle(pool)
    for name in ("hand_cards", "discard_pile", "play_cards"):
        cards = []
        for i, record in enumerate(value[name]):
            if record.get("unknown"):
                if not pool:
                    raise ValueError("Concealed slots exceed the public unseen pool")
                card = pool.pop()
                card.face_down = True
                card.forced_selection = record["forced_selection"]
                from pylatro.flow import _debuff_card

                _debuff_card(state, card)
            else:
                card = known[name, i]
            cards.append(card)
        setattr(state, name, cards)
    state.draw_pile = pool[: value["draw_count"]]
    return state


def public_mask(state, sub_phase):
    mask = compute_action_mask(state, sub_phase, public_information=True)
    # Aura legality depends on a card's edition. A concealed edition cannot
    # justify a legal target, regardless of its identity in a sampled world.
    from pylatro_agent.constants import (
        CONSUMABLE_ACTIONS_PER_SLOT,
        CONSUMABLE_HAND_SUBSET_OFFSET,
        CONSUMABLE_JOKER_OFFSET,
        MAX_CONSUMABLE_SLOTS,
        MAX_HAND_SIZE,
        MAX_JOKER_SLOTS,
        ActionRange,
    )
    from pylatro_agent.subset_actions import consumable_subset_index

    for slot, consumable in enumerate(state.consumables[:MAX_CONSUMABLE_SLOTS]):
        if state.data.centers[consumable.center_key]["name"] == "Aura":
            for i, card in enumerate(state.hand_cards[:MAX_HAND_SIZE]):
                if card.face_down:
                    action = (
                        ActionRange.CONSUMABLE_FLAT_START
                        + slot * CONSUMABLE_ACTIONS_PER_SLOT
                        + CONSUMABLE_HAND_SUBSET_OFFSET
                        + consumable_subset_index((i,))
                    )
                    if action < len(mask):
                        mask[action] = 0
    if concealed_joker_order(state, sub_phase):
        mask[ActionRange.SHOP_SELL_JOKER_START : ActionRange.SHOP_SELL_JOKER_END + 1] = 0
        for slot in range(MAX_CONSUMABLE_SLOTS):
            start = ActionRange.CONSUMABLE_FLAT_START + slot * CONSUMABLE_ACTIONS_PER_SLOT + CONSUMABLE_JOKER_OFFSET
            mask[start : start + MAX_JOKER_SLOTS] = 0
    return mask


def observation(snapshot, data, tokenizer):
    state = simulation_state(snapshot, data)
    mask = public_mask(state, snapshot.sub_phase)
    # Derived candidates/risk are deterministic public-sample estimates. Token
    # piles are canonicalized so neural inputs never encode future draw order.
    state.draw_pile.sort(key=lambda c: canonical(card_record(c, composition=True)))
    obs = tokenizer.tokenize(
        state,
        snapshot.sub_phase,
        action_mask=mask,
        round_score=snapshot.values["round_score"],
        history=restore_history(snapshot.values["history"]),
    )
    from pylatro_agent.constants import DECK_MAX, DECK_START

    offset = 0
    for name in ("hand_cards", "draw_pile", "discard_pile"):
        for i, card in enumerate(getattr(state, name)):
            if offset + i >= DECK_MAX:
                break
            if card.face_down:
                token = obs.tokens[DECK_START + offset + i]
                token[:5] = 0
                token[6] = token[8] = 0
        offset += len(getattr(state, name))
    return obs
