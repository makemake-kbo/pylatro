"""Auditable transfers between policy-head variants without changing observations."""

from __future__ import annotations

from dataclasses import replace

from .agent import BalatroAgent


def add_sale_residual(source: BalatroAgent, vocab, variant: str) -> tuple[BalatroAgent, dict]:
    if source.config.action_head_variant != "legacy_v1":
        raise ValueError("A sale-head transfer must start from the legacy architecture")
    if variant not in ("pooled_sale_residual_v1", "entity_sale_residual_v1"):
        raise ValueError("Unknown sale residual variant")
    target = BalatroAgent(replace(source.config, action_head_variant=variant), vocab)
    incompatible = target.load_state_dict(source.state_dict(), strict=False)
    expected = {
        f"action_grammar_head.sale_residual.{name}" for name in target.action_grammar_head.sale_residual.state_dict()
    }
    if set(incompatible.missing_keys) != expected or incompatible.unexpected_keys:
        raise ValueError(f"Unexpected architecture transfer mismatch: {incompatible}")
    return target, {
        "from_variant": source.config.action_head_variant,
        "to_variant": variant,
        "initialized_keys": sorted(expected),
        "initializer": "random hidden layer; zero output weight and bias",
        "observation_semantics_changed": False,
    }


def add_subset_residual(source: BalatroAgent, vocab) -> tuple[BalatroAgent, dict]:
    variants = {"legacy_v1": "subset_residual_v1", "entity_sale_residual_v1": "entity_sale_subset_residual_v1"}
    if source.config.action_head_variant not in variants:
        raise ValueError("A subset transfer requires a legacy or entity-sale policy without a subset residual")
    variant = variants[source.config.action_head_variant]
    target = BalatroAgent(replace(source.config, action_head_variant=variant), vocab)
    incompatible = target.load_state_dict(source.state_dict(), strict=False)
    expected = {
        f"action_grammar_head.subset_residual.{name}"
        for name in target.action_grammar_head.subset_residual.state_dict()
    }
    if set(incompatible.missing_keys) != expected or incompatible.unexpected_keys:
        raise ValueError(f"Unexpected subset architecture transfer mismatch: {incompatible}")
    return target, {
        "from_variant": source.config.action_head_variant, "to_variant": variant,
        "initialized_keys": sorted(expected), "initializer": "random hidden layers; zero output weight and bias",
        "observation_semantics_changed": False,
        "sampling": (
            "exact normalized subset categorical; zero energy preserves probability but changes RNG consumption"
        ),
    }
