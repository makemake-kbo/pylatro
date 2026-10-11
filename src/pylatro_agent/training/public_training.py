"""Training adapter using exactly the player's public observation and executor.

The only live state belongs to PublicRunner. Retained teaching states are
immutable public projections selected before outcomes are known.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import fields

from pylatro_agent.constants import SubPhase
from pylatro_agent.public_runtime import PublicRunner
from pylatro_agent.public_state import observation
from pylatro_agent.tokenizer import Tokenizer
from pylatro_cli.controller import GamePhase


def selection_key(seed, decision):
    return hashlib.sha256(f"{seed}:{decision}".encode()).hexdigest()


def observation_dict(snapshot, data, tokenizer):
    obs = observation(snapshot, data, tokenizer)
    return {f.name: getattr(obs, f.name) for f in fields(obs)}


class PublicTrainingEnv:
    def __init__(
        self,
        seed,
        data,
        vocab,
        *,
        deck_key="b_blue",
        stake=1,
        win_ante=8,
        max_steps=2048,
        capture_states=False,
        deadline=None,
        **kwargs,
    ):
        self.seed, self.data = seed, data
        self.tokenizer = Tokenizer(vocab)
        self.options = dict(deck_key=deck_key, stake=stake, win_ante=win_ante, max_steps=max_steps, raise_errors=True)
        self.capture_states, self.deadline = capture_states, deadline
        self.runner = None

    @property
    def state(self):
        return self.runner.state

    @property
    def _sub_phase(self):
        return self.runner.sub_phase

    def reset(self, *, seed):
        self.seed = seed
        self.runner = PublicRunner(seed, self.data, **self.options)
        self.decisions = 0
        self.selected = {"early": [], "late": []}
        self._cleared_boss_antes = set()
        return observation_dict(self.runner.snapshot(), self.data, self.tokenizer), {}

    def step(self, action):
        if self.deadline is not None and time.time() >= self.deadline:
            raise TimeoutError("Study deadline reached during collection")
        runner = self.runner
        snapshot = runner.snapshot()
        if not runner.compute_mask()[action]:
            raise ValueError("Illegal public training action")
        ante = int(snapshot.values["round_resets"]["ante"])
        if self.capture_states and snapshot.sub_phase == SubPhase.CHOOSE_ACTION and 1 <= ante <= 8:
            bucket = self.selected["early" if ante <= 3 else "late"]
            row = dict(
                seed=self.seed,
                decision=self.decisions,
                ante=ante,
                selection_key=selection_key(self.seed, self.decisions),
                snapshot=snapshot.values,
            )
            bucket.append(row)
            bucket.sort(key=lambda r: r["selection_key"])
            del bucket[2:]
        runner.step(int(action))
        self.decisions += 1
        terminal = runner.done and runner.phase in (GamePhase.GAME_OVER, GamePhase.GAME_WON)
        truncated = runner.done and not terminal
        if snapshot.values["blind_on_deck"] == "Boss" and runner.sub_phase == SubPhase.SHOP:
            self._cleared_boss_antes.add(ante)
        obs = observation_dict(runner.snapshot(), self.data, self.tokenizer)
        return (
            obs,
            10.0 if terminal and runner.won else 0.0,
            terminal,
            truncated,
            dict(
                ante=runner.max_ante,
                won=runner.won,
                round_score=runner.round_score,
                stalled=truncated,
                dollars=runner.state.dollars,
                blind_on_deck=runner.state.blind_on_deck,
            ),
        )

    def selected_public_states(self):
        return self.selected["early"] + self.selected["late"]

    def close(self):
        pass
