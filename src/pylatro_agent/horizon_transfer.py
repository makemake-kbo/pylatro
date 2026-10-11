"""Evaluation intervention: change early decisions with a common later policy."""

from __future__ import annotations

from dataclasses import asdict

from torch import nn

from .action_grammar import ActionGrammarDistribution, ActionGrammarOutput
from .constants import CURRENT_ANTE_SCALAR_INDEX


class EarlyAntePolicy(nn.Module):
    """Use an early policy through a fixed Ante, then the control continuation.

    Routes raw outputs before constructing ONE distribution, so observing the
    second policy never consumes extra sampling randomness. The inherited
    value predictions describe the component policies, not the hybrid value.
    """

    def __init__(self, early, continuation, *, through_ante=3):
        super().__init__()
        if asdict(early.config) != asdict(continuation.config):
            raise ValueError("Early-transfer intervention requires identical architecture/sampling configurations")
        if type(through_ante) is not int or not 1 <= through_ante < 8:
            raise ValueError("Early-transfer cutoff must precede the Ante-8 win objective")
        self.early = early
        self.continuation = continuation
        self.config = continuation.config
        self.through_ante = through_ante
        self.route_counts = {"early": 0, "continuation": 0}

    def action_distribution(
        self, tokens, token_types, scalars, attention_mask, action_mask, temperature=1.0, **history
    ):
        early_rows = scalars[:, CURRENT_ANTE_SCALAR_INDEX] <= self.through_ante
        inputs = dict(
            tokens=tokens,
            token_types=token_types,
            scalars=scalars,
            attention_mask=attention_mask,
            action_mask=action_mask,
            **history,
        )
        raw, values = {}, {}
        for name, model, selected in (
            ("early", self.early, early_rows),
            ("continuation", self.continuation, ~early_rows),
        ):
            indices = selected.nonzero().flatten()
            count = len(indices)
            self.route_counts[name] += count
            if not count:
                continue
            part_raw, part_values = model(
                **{key: value[indices] for key, value in inputs.items() if value is not None},
                return_raw_outputs=True,
            )
            for destination, part in ((raw, part_raw), (values, part_values)):
                for key, tensor in part.items():
                    if key not in destination:
                        destination[key] = tensor.new_empty((len(tokens), *tensor.shape[1:]))
                    destination[key][indices] = tensor
        return ActionGrammarDistribution(
            ActionGrammarOutput(**raw),
            action_mask,
            temperature=temperature,
            tokens=tokens,
            hand_ar_mixture_eps=self.config.hand_ar_mixture_eps,
        ), values
