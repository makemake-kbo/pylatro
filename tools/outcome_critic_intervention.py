"""Explicit experimental hook for a critic trained only on completed episodes.

The core reference stays frozen during the study. This hook is scoped to one
training call and records the effective loss protocol in every saved checkpoint.
"""

from contextlib import contextmanager
from dataclasses import replace

from pylatro_agent.training import ppo

COMPLETE_CRITIC_PROTOCOL = dict(
    version="complete_episode_critic_v1",
    value_definition="10 * P(win), zero return residual",
    ppo_return_loss_coeff=0.,
    ppo_outcome_loss_coeff=0.,
    ppo_critic_parameters="Frozen with grad=None; Adam weights, moments and counters are untouched",
    replay_outcome_loss_coeff=.1,
    replay_updates_per_rollout=30,
    replay_physical_batch=256,
    replay_sampling="Row-uniform complete recent student episodes; no teacher labels",
    replay_gradient_scope="Private outcome tower only; encoder and actor remain unchanged during replay",
    note="A declared critic-treatment bundle. It does not isolate the return loss, label mask and replay exposure.",
)


@contextmanager
def complete_episode_critic_updates(enabled=True):
    if not enabled:
        yield
        return
    original_update = ppo._run_ppo_update
    original_save = ppo._save_checkpoint

    def update(**kwargs):
        config = kwargs["config"]
        if config.value_loss_coeff != 0 or config.gamma != 1:
            raise ValueError("The complete-episode critic requires gamma one and no return residual loss")
        parameters = list(kwargs["model"].value_head.parameters())
        flags = [parameter.requires_grad for parameter in parameters]
        for parameter in parameters:
            parameter.grad = None
            parameter.requires_grad_(False)
        try:
            return original_update(**{**kwargs, "config": replace(config, outcome_loss_coeff=0.)})
        finally:
            for parameter, flag in zip(parameters, flags, strict=True):
                parameter.requires_grad_(flag)

    def save(**kwargs):
        extra = {**(kwargs.get("extra") or {}), "outcome_study_critic_protocol": COMPLETE_CRITIC_PROTOCOL}
        return original_save(**{**kwargs, "extra": extra})

    ppo._run_ppo_update, ppo._save_checkpoint = update, save
    try:
        yield
    finally:
        ppo._run_ppo_update, ppo._save_checkpoint = original_update, original_save
