"""Optimized training data generation using FastRunner.

Uses a single-pass approach: FastRunner drives game logic (no Gymnasium
overhead) and observations are built inline only when needed.  When
``min_ante`` is set, games are played to completion first without
observations, and only qualifying games are re-run with observation building
(two-pass fallback for high ``min_ante`` thresholds).
"""

from __future__ import annotations

import gc
import logging
import math
import multiprocessing
import os
import threading
import time
from typing import TYPE_CHECKING, Any

from pylatro import GameData, load_game_data
from pylatro_cli.controller import GamePhase

from ..action import decode_action
from ..heuristic import HeuristicAgent
from ..reward import default_reward
from ..survival import compute_ante_survival_targets
from ..tokenizer import Tokenizer
from ..vocab import Vocab, build_vocab
from .fast_runner import FastRunner, _blind_target

if TYPE_CHECKING:
    import numpy as np


logger = logging.getLogger(__name__)

_shared_counter: Any = None
_shared_total_attempted: Any = None
_shared_busy: Any = None
_shared_target: int = 0


def _init_worker(counter: Any, total_attempted: Any, busy: Any, target: int) -> None:
    global _shared_counter, _shared_total_attempted, _shared_busy, _shared_target
    _shared_counter = counter
    _shared_total_attempted = total_attempted
    _shared_busy = busy
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
    shop_items = list(state.shop.cards) + list(state.shop.vouchers) + list(state.shop.boosters)
    pack_cards = state.pack.cards if state.pack is not None else ()
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
        "reroll_cost": state.current_round.reroll_cost,
        "free_rerolls": state.current_round.free_rerolls,
        "joker_keys": tuple(state.joker_keys),
        "consumable_keys": tuple(state.consumable_keys),
        "last_tarot_planet": state.last_tarot_planet or "",
        "shop_keys": tuple(item.center_key for item in shop_items),
        "shop_item_details": tuple(_shop_item_detail(state, item) for item in shop_items),
        "pack_booster_key": state.pack.booster_key if state.pack is not None else "",
        "pack_card_keys": tuple(card.center_key for card in pack_cards),
        "pack_state_name": state.pack.state_name if state.pack is not None else "",
        "pack_card_details": tuple(_pack_card_detail(state, card) for card in pack_cards),
        "pack_choices_remaining": pack_cr,
        "blind_just_beaten": runner.blind_just_beaten,
        "progress_made": False,
        "steps_since_progress": runner.steps_since_progress,
        "stalled": stalled,
    }


def _info_signature(info: dict[str, Any]) -> tuple[Any, ...]:
    return (
        info.get("ante", 0),
        info.get("blind_on_deck", ""),
        info.get("blind_target", 0),
        info.get("round_score", 0),
        info.get("hands_left", 0),
        info.get("discards_left", 0),
        info.get("dollars", 0),
        info.get("phase", ""),
        info.get("reroll_cost", 0),
        info.get("free_rerolls", 0),
        info.get("joker_keys", ()),
        info.get("consumable_keys", ()),
        info.get("last_tarot_planet", ""),
        info.get("shop_keys", ()),
        info.get("pack_booster_key", ""),
        info.get("pack_card_keys", ()),
        info.get("pack_choices_remaining", 0),
    )


def _pack_card_detail(state, card) -> dict[str, object]:
    front_key = card.front_key or ""
    front = state.data.cards.get(front_key, {}) if front_key else {}
    return {
        "center_key": card.center_key,
        "front_key": front_key,
        "rank": front_key[2:] if len(front_key) > 2 else "",
        "suit": str(front.get("suit", "")),
        "seal": card.seal or "",
        "edition": card.edition or {},
    }


def _shop_item_detail(state, card) -> dict[str, object]:
    center = state.data.centers.get(card.center_key, {})
    return {
        "center_key": card.center_key,
        "card_type": getattr(card, "card_type", ""),
        "pack_state_name": _pack_state_name_for_shop_card(center),
    }


def _pack_state_name_for_shop_card(center: dict) -> str:
    name = str(center.get("name", ""))
    if "Arcana" in name:
        return "TAROT_PACK"
    if "Celestial" in name:
        return "PLANET_PACK"
    if "Spectral" in name:
        return "SPECTRAL_PACK"
    if "Standard" in name:
        return "STANDARD_PACK"
    if "Buffoon" in name:
        return "BUFFOON_PACK"
    return ""


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
        decoded = decode_action(action)
        curr_info["action_type"] = decoded.action_type
        curr_info["action_index"] = decoded.index
        curr_info["action_detail"] = decoded.detail

        reward = default_reward(state, current_prev_info, curr_info, terminated, won)
        rewards.append(float(reward))

        records.append({"obs": current_obs, "action": action, "reward": float(reward)})

        if terminated:
            break

        prev_info = curr_info
        prev_sig = curr_sig
        obs = _build_obs(runner, tokenizer)

    return_targets = _discounted_returns(rewards, gamma)
    survival_target, survival_mask = compute_ante_survival_targets(runner.max_ante, won)
    for rec, rt in zip(records, return_targets, strict=True):
        rec["won"] = won
        rec["max_ante"] = runner.max_ante
        rec["return_target"] = rt
        rec["ante_survival_target"] = survival_target
        rec["ante_survival_mask"] = survival_mask

    return records, runner.max_ante, won


def _run_game_fast_no_obs(
    seed: int,
    data: GameData,
    agent: HeuristicAgent,
) -> tuple[int, bool]:
    runner = FastRunner(seed, data, max_steps=2000)
    while not runner.done:
        mask = runner.compute_mask()
        action = agent.select_action(
            runner.state,
            runner.sub_phase,
            mask,
            selected_cards=runner.selected_cards,
            pending_action=runner.pending_action,
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
    games_since_gc = 0

    while True:
        with _shared_counter.get_lock():
            if _shared_counter.value >= _shared_target:
                break

        if min_ante > 2:
            max_ante, won = _run_game_fast_no_obs(seed, data, agent)
            with _shared_total_attempted.get_lock():
                _shared_total_attempted.value += 1
            seed += 1
            games_since_gc += 1
            if max_ante < min_ante and not won:
                if games_since_gc >= 500:
                    gc.collect()
                    games_since_gc = 0
                continue
            with _shared_counter.get_lock():
                if _shared_counter.value >= _shared_target:
                    break
                _shared_counter.value += 1
            with _shared_busy.get_lock():
                _shared_busy.value += 1
            game_records, _, _ = _run_game_single_pass(
                seed - 1,
                data,
                tokenizer,
                agent,
                gamma,
            )
            with _shared_busy.get_lock():
                _shared_busy.value -= 1
            records.extend(game_records)
            gc.collect()
            games_since_gc = 0
        else:
            with _shared_counter.get_lock():
                if _shared_counter.value >= _shared_target:
                    break
                _shared_counter.value += 1
            with _shared_total_attempted.get_lock():
                _shared_total_attempted.value += 1
            with _shared_busy.get_lock():
                _shared_busy.value += 1
            game_records, _, _ = _run_game_single_pass(
                seed,
                data,
                tokenizer,
                agent,
                gamma,
            )
            with _shared_busy.get_lock():
                _shared_busy.value -= 1
            seed += 1
            records.extend(game_records)
            games_since_gc += 1
            if games_since_gc >= 50:
                gc.collect()
                games_since_gc = 0

    return records


def _get_num_workers(num_workers: int) -> int:
    if num_workers > 0:
        return num_workers
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def _format_eta(eta_secs: float) -> str:
    if math.isinf(eta_secs) or math.isnan(eta_secs) or eta_secs < 0:
        return "--"
    if eta_secs < 60:
        return f"{eta_secs:.0f}s"
    if eta_secs < 3600:
        return f"{eta_secs / 60:.1f}m"
    return f"{eta_secs / 3600:.1f}h"


def _progress_reporter(
    counter: Any,
    total_attempted: Any,
    busy: Any,
    target: int,
    stop_event: threading.Event,
    interval_seconds: int = 60,
) -> None:
    start_time = time.monotonic()
    prev_valid = 0
    prev_attempted = 0
    prev_time = start_time

    while not stop_event.is_set():
        stop_event.wait(timeout=interval_seconds)
        if stop_event.is_set():
            break

        now = time.monotonic()
        valid = counter.value
        attempted = total_attempted.value
        busy_count = busy.value
        elapsed_since_last = now - prev_time

        if elapsed_since_last < 0.1:
            continue

        valid_rate = (valid - prev_valid) / elapsed_since_last
        attempted_rate = (attempted - prev_attempted) / elapsed_since_last
        pct_valid = (valid / max(attempted, 1)) * 100

        remaining = target - valid
        if remaining <= 0:
            eta_str = "done"
        elif valid_rate > 0:
            eta_str = _format_eta(remaining / valid_rate)
        else:
            eta_str = "--"

        status = "collecting" if valid >= target and busy_count > 0 else ""
        busy_str = f" | single_pass: {busy_count}" if busy_count > 0 else ""
        status_str = f" | {status}" if status else ""

        logger.info(
            "Progress: %d/%d valid games (%.1f%% of %d attempted) | "
            "valid: %.1f games/s | attempted: %.1f games/s | ETA: %s%s%s",
            valid,
            target,
            pct_valid,
            attempted,
            valid_rate,
            attempted_rate,
            eta_str,
            busy_str,
            status_str,
        )

        prev_valid = valid
        prev_attempted = attempted
        prev_time = now


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
        num_games,
        num_workers,
        min_ante,
        gamma,
    )

    worker_args = [(i * 1_000_000, min_ante, gamma) for i in range(num_workers)]

    stop_event = threading.Event()

    if num_workers == 1:
        global _shared_counter, _shared_total_attempted, _shared_busy, _shared_target
        _shared_counter = multiprocessing.Value("i", 0)
        _shared_total_attempted = multiprocessing.Value("i", 0)
        _shared_busy = multiprocessing.Value("i", 0)
        _shared_target = num_games
        progress_thread = threading.Thread(
            target=_progress_reporter,
            args=(_shared_counter, _shared_total_attempted, _shared_busy, num_games, stop_event),
            daemon=True,
        )
        progress_thread.start()
        records = _generate_games_worker(worker_args[0])
    else:
        counter = multiprocessing.Value("i", 0)
        total_attempted = multiprocessing.Value("i", 0)
        busy = multiprocessing.Value("i", 0)
        progress_thread = threading.Thread(
            target=_progress_reporter,
            args=(counter, total_attempted, busy, num_games, stop_event),
            daemon=True,
        )
        progress_thread.start()
        t_pool_start = time.monotonic()
        with multiprocessing.Pool(
            num_workers,
            initializer=_init_worker,
            initargs=(counter, total_attempted, busy, num_games),
        ) as pool:
            results = pool.map(_generate_games_worker, worker_args)
        t_pool_elapsed = time.monotonic() - t_pool_start
        logger.info("Pool.map completed in %.1fs, collecting results", t_pool_elapsed)
        records = []
        for r in results:
            records.extend(r)

    stop_event.set()
    progress_thread.join(timeout=5)
    logger.info(
        "Fast-generated %d training records from %d requested games",
        len(records),
        num_games,
    )
    return records
