"""Public-only trace inspection; none of these diagnostics control gameplay."""

from __future__ import annotations

import gzip
import hashlib
import json
from collections import Counter

from pylatro_agent.constants import ActionRange as AR
from pylatro_agent.constants import SubPhase
from pylatro_agent.heuristic import HeuristicAgent
from pylatro_agent.public_state import PublicGameSnapshot, canonical, simulation_state


def public_resources(snapshot, data):
    """Derive diagnostics from the same allowlisted projection seen by a policy."""
    v = snapshot.values
    state = simulation_state(snapshot, data)
    target = max(1, HeuristicAgent()._get_blind_target(state))
    return dict(
        blind_id=[v["round_resets"]["ante"], v["round"], v["blind_on_deck"]],
        blind=v["round_resets"].get("blind"),
        target=target,
        score=v["round_score"],
        deficit=max(0, target - v["round_score"]),
        score_fraction=v["round_score"] / target,
        hands=v["current_round"]["hands_left"],
        discards=v["current_round"]["discards_left"],
        dollars=v["dollars"],
        consumables=v["consumables"],
        jokers=v["jokers"],
        hand_levels=v["hands"],
        restrictions=dict(
            disabled=v["blind_disabled"],
            triggered=v["blind_triggered"],
            eye_hands=v["eye_hands"],
            mouth_only_hand=v["mouth_only_hand"],
            joker_order_unknown=v["joker_order_unknown"],
            concealed_hand=sum(bool(c.get("unknown")) for c in v["hand_cards"]),
            forced_cards=sum(bool(c.get("forced_selection")) for c in v["hand_cards"]),
        ),
    )


def trace_records(path):
    with gzip.open(path, "rt") as f:
        for line in f:
            yield json.loads(line)


def inspect_trace(path):
    """Retain final-blind evidence and the first eligible hand snapshot only."""
    entry = final = first = None
    blind_id = None
    overrides = plays = discards = 0
    trajectory = hashlib.sha256()
    total = 0
    for row in trace_records(path):
        total += 1
        trajectory.update(canonical([row["snapshot_key"], row["selection"]]).encode())
        if row["snapshot"]["sub_phase"] != str(SubPhase.CHOOSE_ACTION):
            continue
        current = row["public_resources"]["blind_id"]
        if current != blind_id:
            blind_id, entry, first = current, row["public_resources"], None
            overrides = plays = discards = 0
        eligible = AR.PLAY_SUBSET_START <= row["proposal"] <= AR.DISCARD_SUBSET_END
        if eligible and first is None:
            first = dict(
                snapshot=row["snapshot"],
                proposal=row["proposal"],
                selection=row["selection"],
                diagnostics=row["diagnostics"],
                step=row["step"],
                snapshot_key=row["snapshot_key"],
            )
        overrides += int(row["diagnostics"]["changed"])
        plays += int(AR.PLAY_SUBSET_START <= row["selection"] <= AR.PLAY_SUBSET_END)
        discards += int(AR.DISCARD_SUBSET_START <= row["selection"] <= AR.DISCARD_SUBSET_END)
        final = row["observed_transition"]["resources"]
    return dict(
        entry=entry,
        final=final,
        first_eligible=first,
        overrides=overrides,
        plays=plays,
        discards=discards,
        decisions=total,
        trajectory_sha256=trajectory.hexdigest(),
    )


def historical_groups(base, hybrid):
    if len(base) != 400 or len(hybrid) != 400:
        raise ValueError("Diagnosis requires all 400 completed historical pairs")
    groups = {k: [] for k in ("regressions", "shared_losses", "rescues", "shared_wins")}
    if [r["seed"] for r in base] != [r["seed"] for r in hybrid] or len({r["seed"] for r in base}) != 400:
        raise ValueError("Historical pairs must be complete and unique")
    pairs = []
    for b, h in zip(base, hybrid, strict=True):
        if not all(r.get("complete") and not r.get("illegal_actions") for r in (b, h)):
            raise ValueError("Historical integrity failure")
        group = {
            (True, False): "regressions",
            (False, False): "shared_losses",
            (False, True): "rescues",
            (True, True): "shared_wins",
        }[bool(b["won"]), bool(h["won"])]
        groups[group].append(b["seed"])
        fields = ("won", "ante", "steps", "changes", "searches", "fallbacks", "paired_sample_sets", "seconds")
        pairs.append(
            dict(seed=b["seed"], group=group, baseline={k: b[k] for k in fields}, incumbent={k: h[k] for k in fields})
        )
    if len(groups["regressions"]) != 32:
        raise ValueError("Expected the confirmed incumbent's 32 regressions")
    subset = {
        k: sorted(v, key=lambda s: hashlib.sha256(s.encode()).hexdigest())[:n]
        for (k, v), n in zip(groups.items(), (32, 16, 8, 8), strict=True)
    }
    loss_antes = {
        arm: dict(Counter(str(r["ante"]) for r in rows if not r["won"]))
        for arm, rows in (("baseline", base), ("incumbent", hybrid))
    }
    return dict(
        counts={k: len(v) for k, v in groups.items()},
        pairs=pairs,
        diagnostic_subset=subset,
        loss_antes=loss_antes,
        limits="Historical compact records lack blind states; causal diagnosis uses descriptive replays.",
    )


def replay_difference(historical, replay):
    fields = ("won", "ante", "steps", "changes", "searches", "fallbacks", "paired_sample_sets")
    return {
        key: dict(historical=historical[key], replay=replay[key]) for key in fields if historical[key] != replay[key]
    }


def restore_snapshot(record):
    return PublicGameSnapshot(canonical(record).encode())
