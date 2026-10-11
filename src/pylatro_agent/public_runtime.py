"""Executor and controllers for the public-information simulator runtime."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import fields

from pylatro_agent.constants import MAX_HAND_SIZE, MAX_JOKER_SLOTS, SubPhase
from pylatro_agent.heuristic import HeuristicAgent
from pylatro_agent.joker_layout import NO_ORDER_DECISION, OrderDecision, OrderObjective, _score_order, order_candidates
from pylatro_agent.public_state import (
    PublicGameSnapshot,
    observation,
    public_mask,
    restore_history,
    simulation_state,
)
from pylatro_agent.training.fast_runner import FastRunner
from pylatro_cli.controller import GamePhase


@contextmanager
def addressable_hand(state):
    """Bound heuristic proposals to the existing action vocabulary.

    This view is used only on an independent public proposal state. Engine
    transitions and scoring retain the complete hand, including held overflow.
    """
    full_hand = state.hand_cards
    if len(full_hand) > MAX_HAND_SIZE:
        state.hand_cards = full_hand[:MAX_HAND_SIZE]
    try:
        yield
    finally:
        state.hand_cards = full_hand


def public_joker_order(snapshot, data, indices):
    if snapshot.values["joker_order_unknown"]:
        return NO_ORDER_DECISION
    state = simulation_state(snapshot, data)
    count = min(len(state.jokers), MAX_JOKER_SLOTS)
    if count < 2:
        return NO_ORDER_DECISION
    candidates = order_candidates(state, count)
    if len(candidates) < 2:
        return NO_ORDER_DECISION
    remaining = max(0, HeuristicAgent()._get_blind_target(state) - snapshot.values["round_score"])
    worlds = [simulation_state(snapshot, data, sample) for sample in range(4)]
    scored = []
    for order in candidates:
        outcomes = [_score_order(world, indices, order) for world in worlds]
        # Completion and cash are estimates over public samples, never a
        # promise of a clear from the executor's private RNG realization.
        clears = sum(chips >= remaining for chips, _ in outcomes) / 4
        chips = sum(c for c, _ in outcomes) / 4
        cash = sum(d for _, d in outcomes) / 4
        scored.append((order, clears, chips, cash))
    chosen = max(scored, key=lambda row: (row[1], row[3] if row[1] else 0, row[2]))
    return OrderDecision(
        order=None if chosen[0] == candidates[0] else chosen[0],
        objective=OrderObjective.SCORE,
        chips=int(chosen[2]),
        dollars=int(chosen[3]),
        chips_forgone=0,
        dollars_gained=0,
        clears=False,
    )


class PublicRunner(FastRunner):
    """The executor is the only owner of a live or simulated engine state."""

    def __init__(self, seed, data, **kwargs):
        # The historical Cython runner accepts integer seeds only. Public
        # studies reserve a new string namespace without changing that API.
        super().__init__(seed if isinstance(seed, int) else 0, data, **kwargs)
        if not isinstance(seed, int):
            self._ctrl.new_run(str(seed), stake=kwargs.get("stake", 1), deck_key=kwargs.get("deck_key", "b_blue"))
            self._state = self._ctrl.state
            self._state.win_ante = kwargs.get("win_ante", 8)

    def snapshot(self):
        return PublicGameSnapshot.capture(self.state, self.sub_phase, self.round_score, self.history)

    def compute_mask(self):
        state = simulation_state(self.snapshot(), self.state.data)
        return public_mask(state, self.sub_phase)

    def _claim_pack(self, index):
        from pylatro.shop import legal_pack_consumable_targets, pack_consumable_use_targets

        public = simulation_state(self.snapshot(), self.state.data)
        card = public.pack.cards[index]
        if public.data.centers[card.center_key].get("consumeable"):
            targets = pack_consumable_use_targets(
                public, card.center_key, edition=card.edition
            ) or legal_pack_consumable_targets(public, card.center_key, edition=card.edition)
            if targets is None:
                raise RuntimeError("No public pack target")
            return self._ctrl.claim_from_pack(index, hand_targets=targets[0], joker_targets=targets[1])
        return self._ctrl.claim_from_pack(index)

    def _order_jokers(self, indices):
        decision = public_joker_order(self.snapshot(), self.state.data, indices)
        if decision.order is not None:
            count = len(decision.order)
            self.state.jokers[:count] = [self.state.jokers[i] for i in decision.order]
            self.state.joker_keys = [j.center_key for j in self.state.jokers]
        return decision

    @classmethod
    def from_snapshot(cls, snapshot, data, sample=0):
        # Constructing an independent run initializes all runner bookkeeping;
        # replace its generated state with the public conditional world.
        runner = cls(0, data, deck_key="b_blue", stake=1, raise_errors=True)
        runner._state = runner._ctrl.state = simulation_state(snapshot, data, sample)
        runner._sub_phase = snapshot.sub_phase
        runner._ctrl.phase = {
            SubPhase.BLIND_SELECT: GamePhase.BLIND_SELECT,
            SubPhase.CHOOSE_ACTION: GamePhase.HAND_PLAY,
            SubPhase.SHOP: GamePhase.SHOP,
            SubPhase.BOOSTER_PACK: GamePhase.BOOSTER_PACK,
        }[runner.sub_phase]
        runner._round_score = runner._ctrl.round_score = snapshot.values["round_score"]
        runner._history = restore_history(snapshot.values["history"])
        return runner


class PublicHeuristic:
    """Nonrecursive continuation; each decision gets a new public projection."""

    def __init__(self, data):
        self.data = data

    def __call__(self, snapshot):
        state = simulation_state(snapshot, self.data)
        agent = HeuristicAgent(shop_policy="legacy", grow_scalers=True)
        # The ordinary heuristic's emergency fallback uses global NumPy RNG.
        # A stable public legal fallback avoids consuming policy randomness.
        agent._random_valid = lambda mask: int(mask.nonzero()[0][0])
        mask = public_mask(state, snapshot.sub_phase)
        with addressable_hand(state):
            action = int(
                agent.select_action(state, snapshot.sub_phase, mask, round_score=snapshot.values["round_score"])
            )
        if not mask[action]:
            raise RuntimeError("Public continuation proposed an illegal action")
        return action


class PublicNeuralPolicy:
    def __init__(self, checkpoint, data, *, sample=False, policy_seed=0, temperature=1.0):
        import torch

        from pylatro_agent.agent import AgentConfig, BalatroAgent
        from pylatro_agent.checkpoint import load_checkpoint_payload
        from pylatro_agent.tokenizer import Tokenizer
        from pylatro_agent.vocab import build_vocab

        torch.set_num_threads(1)
        self.data, self.sample, self.temperature = data, sample, temperature
        vocab = build_vocab(data)
        self.tokenizer = Tokenizer(vocab)
        payload = load_checkpoint_payload(checkpoint, torch.device("cpu"))
        self.model = BalatroAgent(AgentConfig(**payload["agent_config"]), vocab).float().eval()
        self.model.load_state_dict(payload["state_dict"], strict=True)
        self.model.requires_grad_(False)
        self.reset_rng(policy_seed)

    def reset_rng(self, seed):
        import torch

        self.rng_state = torch.Generator(device="cpu").manual_seed(seed).get_state()

    def __call__(self, snapshot):
        import torch

        obs = observation(snapshot, self.data, self.tokenizer)
        floating = {"scalars", "action_mask", "history_event_features", "history_omitted"}
        batch = {
            f.name: torch.as_tensor(
                getattr(obs, f.name), dtype=torch.float32 if f.name in floating else torch.long
            ).unsqueeze(0)
            for f in fields(obs)
        }
        with torch.inference_mode(), torch.random.fork_rng(devices=[]):
            torch.set_rng_state(self.rng_state)
            distribution, _ = self.model.action_distribution(**batch, temperature=self.temperature)
            action = int((distribution.sample() if self.sample else distribution.mode()).item())
            self.rng_state = torch.get_rng_state()
        if not obs.action_mask[action]:
            raise RuntimeError("Public neural policy proposed an illegal action")
        return action
