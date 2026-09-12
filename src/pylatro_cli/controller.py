"""GameController, pure data layer wrapping the pylatro engine. No Textual imports."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import floor
from typing import TYPE_CHECKING

import pylatro
from pylatro.blind import blind_multiplier

if TYPE_CHECKING:
    from pylatro import (
        ConsumableInstance,
        DiscardResult,
        GameData,
        JokerInstance,
        PackState,
        PlayResult,
        RunState,
        UseConsumableResult,
    )
    from pylatro.models import PlayingCard, ShopCard


class GamePhase(StrEnum):
    MENU = "menu"
    BLIND_SELECT = "blind_select"
    HAND_PLAY = "hand_play"
    SHOP = "shop"
    BOOSTER_PACK = "booster_pack"
    GAME_OVER = "game_over"
    GAME_WON = "game_won"


@dataclass
class GameController:
    data: GameData
    state: RunState | None = None
    phase: GamePhase = GamePhase.MENU
    round_score: int = 0

    # ── Run lifecycle ──

    def new_run(self, seed: str, stake: int = 1, deck_key: str = "b_red") -> None:
        self.state = pylatro.create_run_state(seed, stake, deck_key, data=self.data)
        self.phase = GamePhase.BLIND_SELECT
        self.round_score = 0

    # ── Blind selection ──

    def select_blind(self, blind_type: str = "Small") -> list[PlayingCard]:
        assert self.state is not None
        drawn = pylatro.start_blind(self.state, blind_type)
        self.round_score = 0
        self.phase = GamePhase.HAND_PLAY
        return drawn

    def skip_blind(self) -> str:
        assert self.state is not None
        return pylatro.skip_blind(self.state)

    def reroll_boss(self) -> str:
        assert self.state is not None
        return pylatro.reroll_boss(self.state)

    # ── Hand play ──

    def play_selected(self, indices: list[int]) -> PlayResult:
        assert self.state is not None
        cards = [self.state.hand_cards[i] for i in sorted(indices)]
        result = pylatro.play_cards(self.state, cards)
        self.round_score += result.score.total
        # If blind beaten, caller transitions to shop; otherwise draw and check loss.
        if self.blind_beaten():
            result.blue_seals_activated = sum(
                card.seal == "Blue" and not card.debuff
                for card in result.score.held_cards
            )
            result.blue_planets_generated = [
                consumable.center_key
                for consumable in pylatro.resolve_blue_seals(
                    self.state,
                    result.score.held_cards,
                    result.score.hand_name,
                )
            ]
            result.held_gold_count, result.held_gold_payout = pylatro.resolve_held_gold_cards(
                self.state,
                result.score.held_cards,
            )
        else:
            pylatro.draw_to_hand(self.state)
            if self.state.current_round.hands_left <= 0:
                if pylatro.check_mr_bones(self.state, self.round_score, self.blind_target()):
                    self.round_score = 0
                else:
                    self.phase = GamePhase.GAME_OVER
        return result

    def discard_selected(self, indices: list[int]) -> DiscardResult:
        assert self.state is not None
        cards = [self.state.hand_cards[i] for i in sorted(indices)]
        return pylatro.discard_cards(self.state, cards)

    # ── Blind target ──

    def blind_target(self) -> int:
        assert self.state is not None
        blind = self.state.round_resets.blind
        if blind is None:
            return 0
        ante = self.state.round_resets.ante
        scaling = self.state.stake if self.state.stake <= 3 else 3
        base = pylatro.get_blind_amount(ante, scaling)
        mult = blind_multiplier(self.state, blind)
        return floor(base * mult)

    def blind_beaten(self) -> bool:
        return self.round_score >= self.blind_target()

    # ── Shop transition ──

    def cash_out(self) -> None:
        assert self.state is not None
        # Mark the beaten blind as defeated and advance to the next
        blind_order = ("Small", "Big", "Boss")
        for i, bt in enumerate(blind_order):
            if self.state.round_resets.blind_states.get(bt) == "Current":
                self.state.round_resets.blind_states[bt] = "Defeated"
                if i + 1 < len(blind_order):
                    self.state.round_resets.blind_states[blind_order[i + 1]] = "Select"
                    self.state.blind_on_deck = blind_order[i + 1]
                break
        pylatro.cash_out(self.state)
        if self.state.round_resets.ante > self.state.win_ante:
            self.phase = GamePhase.GAME_WON
            return
        self.phase = GamePhase.SHOP

    def enter_shop(self) -> None:
        assert self.state is not None
        pylatro.populate_shop(self.state)

    # ── Shop actions ──

    def buy_card(self, index: int) -> ShopCard:
        assert self.state is not None
        return pylatro.buy_shop_card(self.state, index)

    def buy_voucher(self, voucher_key: str) -> None:
        assert self.state is not None
        pylatro.redeem_voucher(self.state, voucher_key)

    def reroll(self) -> list[ShopCard]:
        assert self.state is not None
        return pylatro.reroll_shop(self.state)

    def sell_joker(self, index: int) -> None:
        assert self.state is not None
        pylatro.sell_owned_joker(self.state, index)
        if self.phase == GamePhase.HAND_PLAY and self.blind_beaten():
            self.cash_out()
            if self.phase != GamePhase.GAME_WON:
                self.enter_shop()

    def sell_consumable(self, index: int) -> None:
        assert self.state is not None
        pylatro.sell_owned_consumable(self.state, index)

    def leave_shop(self) -> list[str]:
        assert self.state is not None
        result = pylatro.finish_shop(self.state)
        self.phase = GamePhase.BLIND_SELECT
        return result

    # ── Booster packs ──

    def open_pack(self, index: int) -> PackState:
        assert self.state is not None
        pack = pylatro.open_booster_pack(self.state, index)
        self.phase = GamePhase.BOOSTER_PACK
        return pack

    def claim_from_pack(
        self, index: int, *, hand_targets: tuple[int, ...] | None = None,
        joker_targets: tuple[int, ...] = (),
    ) -> ShopCard:
        assert self.state is not None
        result = pylatro.claim_pack_card(self.state, index, hand_targets=hand_targets, joker_targets=joker_targets)
        if self.state.pack is None:
            self.phase = GamePhase.SHOP
        return result

    def close_current_pack(self, *, skipped: bool = True) -> None:
        assert self.state is not None
        pylatro.close_pack(self.state, skipped=skipped)
        self.phase = GamePhase.SHOP

    # ── Consumables ──

    def can_use(
        self,
        consumable: int | str | ConsumableInstance,
        *,
        hand_targets: tuple[int, ...] = (),
        joker_targets: tuple[int, ...] = (),
    ) -> bool:
        assert self.state is not None
        return pylatro.can_use_consumable(
            self.state, consumable, hand_targets=hand_targets, joker_targets=joker_targets
        )

    def use_consumable_on(
        self,
        consumable: int | str | ConsumableInstance,
        *,
        hand_targets: tuple[int, ...] = (),
        joker_targets: tuple[int, ...] = (),
    ) -> UseConsumableResult:
        assert self.state is not None
        return pylatro.use_consumable(
            self.state, consumable, hand_targets=hand_targets, joker_targets=joker_targets
        )

    # ── Display helpers ──

    def hand_evaluation(self, indices: list[int]) -> tuple[str, str, list[PlayingCard]] | None:
        """Preview what poker hand the selected cards form. Returns (name, display_name, scoring_cards)."""
        assert self.state is not None
        if not indices:
            return None
        cards = [self.state.hand_cards[i] for i in sorted(indices)]
        hand_name, display_name, _all_hands, scoring = pylatro.get_poker_hand_info(self.state, cards)
        return hand_name, display_name, scoring

    def hand_chips_mult(self, hand_name: str) -> tuple[int, int]:
        """Current chips and mult for a poker hand name.

        The engine keeps hands[name]["chips"/"mult"] up to date with the hand's
        level (_level_up_hand recomputes them), so no level math is needed here.
        """
        assert self.state is not None
        hand_data = self.state.hands.get(hand_name)
        if hand_data is None:
            return 0, 0
        return hand_data.get("chips", 0), hand_data.get("mult", 0)

    def card_display_info(self, card: PlayingCard) -> dict:
        """Return display-friendly info for a playing card."""
        return {
            "rank": card.rank,
            "suit": card.suit,
            "debuff": card.debuff,
            "face_down": card.face_down,
            "edition": card.edition_key,
            "seal": card.seal,
            "center_key": card.center_key,
        }

    def joker_display_info(self, joker: JokerInstance) -> dict:
        """Return display-friendly info for a joker."""
        center = self.data.centers.get(joker.center_key, {})
        return {
            "name": center.get("name", joker.center_key),
            "description": center.get("description", ""),
            "rarity": center.get("rarity", 1),
            "cost": center.get("cost", 0),
            "sell_cost": joker.sell_cost,
            "edition": joker.edition,
            "eternal": joker.eternal,
            "perishable": joker.perishable,
            "rental": joker.rental,
            "debuff": joker.debuff,
        }

    def consumable_display_info(self, consumable: ConsumableInstance) -> dict:
        """Return display-friendly info for a consumable."""
        center = self.data.centers.get(consumable.center_key, {})
        return {
            "name": center.get("name", consumable.center_key),
            "description": center.get("description", ""),
            "set": center.get("set", ""),
            "sell_cost": consumable.sell_cost,
            "edition": consumable.edition,
        }
