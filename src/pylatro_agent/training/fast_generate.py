"""Optimized training data generation using FastRunner.

Uses a single-pass approach: FastRunner drives game logic (no Gymnasium
overhead) and observations are built inline only when needed.  When
``min_ante`` is set, games are played to completion first without
observations, and only qualifying games are re-run with observation building
(two-pass fallback for high ``min_ante`` thresholds).
"""

from __future__ import annotations

import logging
import multiprocessing
import os
from typing import TYPE_CHECKING, Any

from pylatro import GameData, load_game_data
from pylatro_cli.controller import GamePhase

from ..heuristic import HeuristicAgent
from ..reward import default_reward
from ..tokenizer import Tokenizer
from ..vocab import Vocab, build_vocab
from .fast_runner import FastRunner, _blind_target

if TYPE_CHECKING:
    import numpy as np


logger = logging.getLogger(__name__)

_shared_counter: Any = None
_shared_target: int = 0


def _init_worker(counter: Any, target: int) -> None:
    global _shared_counter, _shared_target
    _shared_counter = counter
    _shared_target = target


def _discounted_returns(rewards: list[float], gamma: float) -> list[float]:
    returns = [0.0] * len(rewards)
    running = 0.0
    for idx in range(len(rewards) - 1, -1, -1):
        running = rewards[idx] + gamma * running
        returns[idx] = running
    return returns


def _capture_info(runner: FastRunner, *, stalled: bool = False) -> dict[str, Any]:
    state = runner.state
    ctrl_phase = runner.phase
    pack_cr = state.pack.choices_remaining if state.pack is not None else 0
    shop_n = len(state.shop.cards) + len(state.shop.vouchers) + len(state.shop.boosters)
    return {
        "ante": state.round_resets.ante,
        "round_score": runner.round_score,
        "blind_beaten": runner.phase == GamePhase.HAND_PLAY and runner._ctrl.blind_beaten(),
        "blind_on_deck": state.blind_on_deck or "",
        "blind_target": _blind_target(state),
        "hands_left": state.current_round.hands_left,
        "discards_left": state.current_round.discards_left,
        "dollars": state.dollars,
        "in_shop": ctrl_phase == GamePhase.SHOP,
        "phase": ctrl_phase,
        "sub_phase": runner.sub_phase,
        "joker_count": len(state.jokers),
        "consumable_count": len(state.consumables),
        "shop_item_count": shop_n,
        "pack_choices_remaining": pack_cr,
        "blind_just_beaten": runner.blind_just_beaten,
        "progress_made": False,
        "steps_since_progress": runner.steps_since_progress,
        "stalled": stalled,
    }


def _info_signature(info: dict[str, Any]) -> tuple[Any, ...]:
    """Mirror BalatroEnv progress detection on captured info snapshots."""
    return (
        info.get("ante", 0),
        info.get("blind_on_deck", ""),
        info.get("blind_target", 0),
        info.get("round_score", 0),
        info.get("hands_left", 0),
        info.get("discards_left", 0),
        info.get("dollars", 0),
        info.get("phase", ""),
        info.get("sub_phase", ""),
        info.get("joker_count", 0),
        info.get("consumable_count", 0),
        info.get("shop_item_count", 0),
        info.get("pack_choices_remaining", 0),
    )


def _build_obs(runner: FastRunner, tokenizer: Tokenizer) -> dict[str, np.ndarray]:
    mask = runner.compute_mask()
    raw = tokenizer.tokenize(
        runner.state,
        runner.sub_phase,
        selected_cards=runner.selected_cards,
        action_mask=mask.copy(),
    )
    return {
        "tokens": raw.tokens,
        "token_types": raw.token_types,
        "scalars": raw.scalars,
        "attention_mask": raw.attention_mask,
        "action_mask": raw.action_mask,
        "selected_cards": raw.selected_cards,
    }


def _run_game_single_pass(
    seed: int,
    data: GameData,
    tokenizer: Tokenizer,
    agent: HeuristicAgent,
    gamma: float,
) -> tuple[list[dict[str, Any]], int, bool]:
    runner = FastRunner(seed, data)
    records: list[dict[str, Any]] = []
    rewards: list[float] = []
    steps_since_progress = 0
    won = False

    prev_info = _capture_info(runner)
    prev_sig = _info_signature(prev_info)
    obs = _build_obs(runner, tokenizer)

    while not runner.done:
        mask = runner.compute_mask()
        action = agent.select_action(
            runner.state,
            runner.sub_phase,
            mask,
            selected_cards=runner.selected_cards,
            pending_action=runner.pending_action,
            pending_consumable_slot=runner.pending_consumable_slot,
        )

        current_obs = obs
        current_prev_info = prev_info

        runner.step(action)

        state = runner.state
        terminated = runner.done
        won = runner.phase == GamePhase.GAME_WON
        curr_info = _capture_info(runner)
        curr_sig = _info_signature(curr_info)
        progress_made = curr_sig != prev_sig
        if progress_made:
            steps_since_progress = 0
        else:
            steps_since_progress += 1

        stalled = False
        if not terminated and steps_since_progress >= 2000:
            terminated = True
            stalled = True

        curr_info["stalled"] = stalled
        curr_info["blind_just_beaten"] = runner.blind_just_beaten
        curr_info["progress_made"] = progress_made
        curr_info["steps_since_progress"] = steps_since_progress

        reward = default_reward(state, current_prev_info, curr_info, terminated, won)
        rewards.append(float(reward))

        records.append({"obs": current_obs, "action": action, "reward": float(reward)})

        if terminated:
            break

        prev_info = curr_info
        prev_sig = curr_sig
        obs = _build_obs(runner, tokenizer)

    return_targets = _discounted_returns(rewards, gamma)
    for rec, rt in zip(records, return_targets, strict=True):
        rec["won"] = won
        rec["max_ante"] = runner.max_ante
        rec["return_target"] = rt

    return records, runner.max_ante, won


def _run_game_fast_no_obs(
    seed: int, data: GameData, agent: HeuristicAgent,
) -> tuple[int, bool]:
    runner = FastRunner(seed, data)
    while not runner.done:
        mask = runner.compute_mask()
        action = agent.select_action(
            runner.state,
            runner.sub_phase,
            mask,
            selected_cards=runner.selected_cards,
            pending_action=runner.pending_action,
            pending_consumable_slot=runner.pending_consumable_slot,
        )
        runner.step(action)
    return runner.max_ante, runner.won


def _generate_games_worker(args: tuple) -> list[dict[str, Any]]:
    seed_start, min_ante, gamma = args
    data = load_game_data()
    vocab = build_vocab(data)
    tokenizer = Tokenizer(vocab=vocab)
    agent = HeuristicAgent()
    records: list[dict[str, Any]] = []
    seed = seed_start

    while True:
        with _shared_counter.get_lock():
            if _shared_counter.value >= _shared_target:
                break

        if min_ante > 2:
            max_ante, won = _run_game_fast_no_obs(seed, data, agent)
            seed += 1
            if max_ante < min_ante and not won:
                continue
            with _shared_counter.get_lock():
                if _shared_counter.value >= _shared_target:
                    break
                _shared_counter.value += 1
            game_records, _, _ = _run_game_single_pass(
                seed - 1, data, tokenizer, agent, gamma,
            )
            records.extend(game_records)
        else:
            with _shared_counter.get_lock():
                if _shared_counter.value >= _shared_target:
                    break
                _shared_counter.value += 1
            game_records, _, _ = _run_game_single_pass(
                seed, data, tokenizer, agent, gamma,
            )
            seed += 1
            records.extend(game_records)

    return records


def _get_num_workers(num_workers: int) -> int:
    if num_workers > 0:
        return num_workers
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def generate_training_data(
    num_games: int,
    data: GameData | None = None,
    vocab: Vocab | None = None,
    min_ante: int = 5,
    gamma: float = 0.995,
    num_workers: int = 0,
) -> list[dict[str, Any]]:
    num_workers = _get_num_workers(num_workers)
    logger.info(
        "Fast-generating %d games across %d workers (min_ante=%d, gamma=%.3f)",
        num_games, num_workers, min_ante, gamma,
    )

    worker_args = [(i * 1_000_000, min_ante, gamma) for i in range(num_workers)]

    if num_workers == 1:
        global _shared_counter, _shared_target
        _shared_counter = multiprocessing.Value("i", 0)
        _shared_target = num_games
        records = _generate_games_worker(worker_args[0])
    else:
        counter = multiprocessing.Value("i", 0)
        with multiprocessing.Pool(
            num_workers, initializer=_init_worker, initargs=(counter, num_games)
        ) as pool:
            results = pool.map(_generate_games_worker, worker_args)
        records = []
        for r in results:
            records.extend(r)

    logger.info("Fast-generated %d training records from %d games", len(records), num_games)
    return records
