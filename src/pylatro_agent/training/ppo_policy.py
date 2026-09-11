"""Shared policy-distribution and temperature helpers for PPO and evaluation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from ..action import ActionType, decode_action
from ..action_grammar import ActionGrammarDistribution, ActionGrammarOutput
from ..agent import BalatroAgent
from ..constants import (
    NUM_ACTIONS,
)

if TYPE_CHECKING:
    from .ppo_config import (
        PPOConfig,
    )

logger = logging.getLogger(__name__)


_HISTORY_SIGNATURE_CACHE: dict[type, bool] = {}


_ACTION_TYPES = tuple(ActionType)


_ACTION_TYPE_TO_INDEX = {action_type: idx for idx, action_type in enumerate(_ACTION_TYPES)}


_ACTION_ID_TO_TYPE_INDEX = torch.tensor(
    [_ACTION_TYPE_TO_INDEX[decode_action(action_id).action_type] for action_id in range(NUM_ACTIONS)],
    dtype=torch.long,
)


def _unwrap_model(model: nn.Module) -> nn.Module:
    """Return the underlying model when wrapped for multi-GPU training."""
    return model.module if isinstance(model, nn.DataParallel) else model


def _grammar_distribution(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    temperature: float | torch.Tensor = 1.0,
):
    import inspect

    base_model = _unwrap_model(model)
    model_type = type(base_model)
    supports_history = _HISTORY_SIGNATURE_CACHE.get(model_type)
    if supports_history is None:
        parameters = inspect.signature(base_model.action_distribution).parameters
        supports_history = "history_events" in parameters or any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
        )
        _HISTORY_SIGNATURE_CACHE[model_type] = supports_history
    history_kwargs = {
        key: batch[key]
        for key in (
            "history_events",
            "history_event_features",
            "history_cards",
            "history_card_mask",
            "history_jokers",
            "history_joker_mask",
            "history_event_mask",
            "history_round_mask",
            "history_omitted",
        )
        if supports_history and key in batch
    }
    if isinstance(model, nn.DataParallel) and isinstance(base_model, BalatroAgent):
        raw_output, value_dict = model(
            batch["tokens"],
            batch["token_types"],
            batch["scalars"],
            batch["attention_mask"],
            batch["action_mask"],
            temperature=temperature,
            return_raw_outputs=True,
            **history_kwargs,
        )
        grammar_output = ActionGrammarOutput(**raw_output)
        return (
            ActionGrammarDistribution(
                grammar_output,
                batch["action_mask"],
                temperature=temperature,
                tokens=batch["tokens"],
                hand_ar_mixture_eps=base_model.config.hand_ar_mixture_eps,
            ),
            value_dict,
        )
    return base_model.action_distribution(
        batch["tokens"],
        batch["token_types"],
        batch["scalars"],
        batch["attention_mask"],
        batch["action_mask"],
        temperature=temperature,
        **history_kwargs,
    )


def _critic_predictions(model, batch, temperature=1.0):
    """Predict values with a frozen encoder and no unused policy-head work."""
    if isinstance(_unwrap_model(model), BalatroAgent):
        inputs = {key: batch[key] for key in (
            "tokens", "token_types", "scalars", "attention_mask", "action_mask",
            "history_events", "history_event_features", "history_cards", "history_card_mask",
            "history_jokers", "history_joker_mask", "history_event_mask", "history_round_mask", "history_omitted",
        ) if key in batch}
        _, values = model(**inputs, critic_only=True)
        return values
    # Lightweight test/custom policies may only expose action_distribution.
    _, values = _grammar_distribution(model, batch, temperature=temperature)
    return values


def _policy_temperature_for_scalars(
    scalars: torch.Tensor,
    config: PPOConfig,
) -> float | torch.Tensor:
    """Return the on-policy temperature for each observation row.

    Only active hand-play states are sharpened. Ante 1 is protected regardless
    of the analytic risk estimate; later hands are sharpened when immediate
    death probability crosses the configured threshold. Because this function
    is used for rollout collection, PPO minibatches, SIL, and KL diagnostics,
    old and new log-probabilities remain distributions over the same policy.
    """

    danger_temperature = config.danger_rollout_temperature
    if danger_temperature is None:
        return config.rollout_temperature

    base = torch.full(
        (scalars.shape[0],),
        float(config.rollout_temperature),
        dtype=scalars.dtype,
        device=scalars.device,
    )
    # Tokenizer scalar layout: ante=2, sub_phase=7 (CHOOSE_ACTION=1),
    # immediate_death_probability=12.
    active_hand = scalars[:, 7].round().eq(1)
    opening_ante = scalars[:, 2] <= 1.0
    immediate_danger = scalars[:, 12] >= float(config.danger_death_probability_threshold)
    sharpen = active_hand & (opening_ante | immediate_danger)
    danger = torch.full_like(base, min(float(danger_temperature), float(config.rollout_temperature)))
    return torch.where(sharpen, danger, base)
