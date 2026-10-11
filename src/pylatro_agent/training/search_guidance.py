"""Two independent paired-world batches supply conservative actor corrections."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np
import torch

from pylatro_agent.constants import ActionRange as AR
from pylatro_agent.constants import SubPhase
from pylatro_agent.hybrid import HybridPolicy, PlannerConfig, PlanningDeadline, bounded
from pylatro_agent.public_state import PublicGameSnapshot, canonical, public_mask, simulation_state

from .ppo_observations import _obs_dicts_to_batch
from .ppo_policy import _grammar_distribution, _policy_temperature_for_scalars
from .public_training import observation_dict


@dataclass(frozen=True)
class GuidanceRecipe:
    coefficient: float = 0.02
    labels_per_group: int = 64
    per_stratum: int = 2
    samples: int = 16
    sample_starts: tuple = (0, 10000)
    seconds: float = 10.0
    minimum_clear_gain: float = 2 / 16
    hand_candidates: int = 6
    hand_steps: int = 60
    version: str = "public-search-guidance-v1"

    def metadata(self):
        # JSON-normalize tuples so round trips are compared strictly.
        import json

        return json.loads(canonical(asdict(self)))


RECIPE = GuidanceRecipe()


def restore_snapshot(row):
    return PublicGameSnapshot(canonical(row["snapshot"]).encode())


def proposals(model, records, data, tokenizer):
    """Greedy collecting-policy proposals; do not sample or consume policy RNG."""
    result = []
    with torch.no_grad():
        for offset in range(0, len(records), 32):
            rows = records[offset : offset + 32]
            batch = _obs_dicts_to_batch(
                [observation_dict(restore_snapshot(r), data, tokenizer) for r in rows], torch.device("cpu")
            )
            dist, _ = _grammar_distribution(model, batch, temperature=1.0)
            actions = dist.mode().cpu().tolist()
            result.extend({**row, "proposal": int(action)} for row, action in zip(rows, actions, strict=True))
    return result


def filter_correction(proposal, candidates, batches, mask):
    if len(candidates) < 2 or len(set(candidates)) != len(candidates) or candidates[0] != proposal:
        return None, "unsupported"
    if any(not 0 <= a < len(mask) or not mask[a] for a in candidates):
        return None, "illegal"
    winners, gains = [], []
    if len(batches) != 2:
        return None, "incomplete"
    for batch in batches:
        values = batch.get("values", [])
        if not batch.get("complete") or batch.get("samples") != RECIPE.samples or len(values) != len(candidates):
            return None, "incomplete"
        if any(len(v) != 4 or not np.isfinite(v).all() for v in values):
            return None, "unsupported"
        ranked = [tuple(v) for v in values]
        best = max(ranked)
        indices = [i for i, value in enumerate(ranked) if value == best]
        if len(indices) != 1:
            return None, "tied"
        winner = indices[0]
        if winner == 0:
            return None, "proposal_retained"
        gain = ranked[winner][0] - ranked[0][0]
        if gain < RECIPE.minimum_clear_gain:
            return None, "small_clear_gain"
        winners.append(candidates[winner])
        gains.append(gain)
    if winners[0] != winners[1]:
        return None, "disagreement"
    return dict(target=winners[0], weight=min(gains)), "accepted"


def teach(record, data, *, deadline=None):
    started = time.monotonic()
    snapshot = restore_snapshot(record)
    proposal = record["proposal"]
    mask = public_mask(simulation_state(snapshot, data), snapshot.sub_phase)
    result = {**record, "batches": [], "candidates": [], "accepted": False}
    if snapshot.sub_phase != SubPhase.CHOOSE_ACTION or not AR.PLAY_SUBSET_START <= proposal <= AR.DISCARD_SUBSET_END:
        result.update(reason="unsupported", seconds=time.monotonic() - started)
        return result
    plans = None
    for start in RECIPE.sample_starts:
        if deadline is not None and time.time() >= deadline:
            raise TimeoutError("Study deadline reached during labels")
        planner = HybridPolicy(
            lambda _: proposal,
            data,
            PlannerConfig(mode="hand", samples=RECIPE.samples, seconds=RECIPE.seconds, sample_start=start),
        )
        completed, stop = [], None
        try:
            seconds = RECIPE.seconds if deadline is None else min(RECIPE.seconds, deadline - time.time())
            with bounded(seconds):
                if plans is None:
                    plans = planner._hand_plans(snapshot, proposal)
                    result["candidates"] = [p.actions[0] for p in plans]
                for sample in range(start, start + RECIPE.samples):
                    paired = [planner._trial(snapshot, plan, sample, True) for plan in plans]
                    if not all(done for _, done in paired):
                        stop = "continuation_cap"
                        break
                    completed.append([score for score, _ in paired])
        except PlanningDeadline:
            stop = "deadline"
        except (RuntimeError, ValueError, IndexError) as exc:
            stop = "unsupported:" + type(exc).__name__
        values = np.mean(np.asarray(completed), axis=0).tolist() if completed else []
        result["batches"].append(
            dict(
                sample_start=start,
                samples=len(completed),
                values=values,
                complete=len(completed) == RECIPE.samples and stop is None,
                stop=stop,
            )
        )
        # Failure cannot be rescued by choosing a different state or candidates.
        if plans is None:
            break
    target, reason = filter_correction(proposal, result["candidates"], result["batches"], mask)
    result.update(reason=reason, accepted=target is not None, seconds=time.monotonic() - started)
    if target is not None:
        result.update(target)
    return result


class SearchGuidance:
    """One fresh cohort's public labels, using an independently checkpointed RNG."""

    def __init__(self, records, data, tokenizer, rng, coefficient=RECIPE.coefficient):
        self.records = [r for r in records if r["accepted"]]
        self.observations = [observation_dict(restore_snapshot(r), data, tokenizer) for r in self.records]
        self.rng, self.coefficient = rng, coefficient
        self.losses = []
        self.draws = 0

    def loss(self, model, config, device):
        if not self.records or self.coefficient == 0:
            return None
        indices = self.rng.choice(
            len(self.records), RECIPE.labels_per_group, replace=len(self.records) < RECIPE.labels_per_group
        )
        batch = _obs_dicts_to_batch([self.observations[i] for i in indices], device)
        actions = torch.tensor([self.records[i]["target"] for i in indices], dtype=torch.long, device=device)
        weights = torch.tensor([self.records[i]["weight"] for i in indices], dtype=torch.float32, device=device)
        dist, _ = _grammar_distribution(
            model, batch, temperature=_policy_temperature_for_scalars(batch["scalars"], config)
        )
        # No value-head output is used. Gradients flow to actor and shared trunk.
        loss = -(dist.log_prob(actions) * weights).sum() / weights.sum()
        self.losses.append(float(loss.detach().cpu()))
        self.draws += len(indices)
        return self.coefficient * loss
