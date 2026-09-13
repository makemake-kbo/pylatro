"""Explicit panels, accounting, and measurements for the second transfer study."""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import signal
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .action import ActionType, decode_action
from .action_grammar import ACTION_TYPE_TO_GRAMMAR_INDEX
from .training.supervised import SupervisedConfig, _collate_batch, _grammar_distribution
from .transfer import write_json

TRAIN_SEEDS = tuple(range(4000000, 4001000))
VALIDATION_SEEDS = tuple(range(4100000, 4100200))
DEV_SEEDS = tuple(range(120000, 120200))
CONFIRM_SEEDS = tuple(range(130000, 130400))
ARMS = {"control": (0.5, 1.0), "hand_weight": (0.5, 2.0), "pure_ar": (1.0, 1.0), "combined": (1.0, 2.0)}
CHECKPOINT = Path("checkpoints/supervised_5k_v68_20260911_181359/supervised_epoch7.pt")
HF_DATA = Path("cache/datasets/5k-balatro-games")
STAGE_LIMITS = {"diagnostics": 5400, "training": 12600, "development": 5400, "confirmation": 9000}


def read_json(path):
    return json.loads(Path(path).read_text())


def disable_dropout(model):
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
        elif isinstance(module, torch.nn.MultiheadAttention):
            module.dropout = 0.0
    model.config.dropout = 0.0


def memorization_name(initialization, mixture):
    name = f"memorize_{initialization}_mixture{mixture}"
    return name + ("_dropout0_v2" if initialization == "epoch7" else "")


def atomic_json(path, payload):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    write_json(temp, payload)
    temp.replace(path)


def stage_limits(root):
    path = Path(root) / "budget_adjustments.json"
    limits = read_json(path)["limits"] if path.exists() else dict(STAGE_LIMITS)
    if set(limits) != set(STAGE_LIMITS) or sum(limits.values()) != 32400 or limits["confirmation"] < 9000:
        raise ValueError("Stage allocation must preserve nine hours and the confirmation reserve")
    return limits


def reserve_diagnostic_completion(root, estimate):
    """Borrow only unused, completed training allocation; never the confirmation reserve."""
    ledger = read_json(Path(root) / "compute.json")
    if any(job["status"] == "running" for job in ledger["jobs"]):
        raise RuntimeError("Budget allocation cannot change during an accelerator job")
    limits = stage_limits(root)
    used = {stage: sum(j["elapsed_seconds"] for j in ledger["jobs"] if j["stage"] == stage) for stage in limits}
    needed = max(0, math.ceil(1.2 * estimate + 5 - (limits["diagnostics"] - used["diagnostics"])))
    if not needed:
        return
    expected = {f"train/{arm}/{seed}" for arm in ARMS for seed in (0, 1)}
    completed = {j["name"] for j in ledger["jobs"] if j["status"] == "complete"}
    if not expected <= completed or needed > limits["training"] - used["training"]:
        raise TimeoutError("No completed training allocation is available for the remaining diagnostics")
    limits["training"] -= needed
    limits["diagnostics"] += needed
    path = Path(root) / "budget_adjustments.json"
    history = read_json(path)["history"] if path.exists() else []
    history.append(
        dict(
            from_stage="training",
            to_stage="diagnostics",
            seconds=needed,
            used_seconds=used,
            reason="Finish corrected diagnostics and full validation with the 20% admission margin",
        )
    )
    atomic_json(path, dict(limits=limits, history=history))


def reserve_evaluation_capacity(root, stage, estimate):
    """Move unused completed-stage allocation forward, without spending the reserve early."""
    if stage not in ("development", "confirmation"):
        return
    root = Path(root)
    ledger = read_json(root / "compute.json")
    if any(job["status"] == "running" for job in ledger["jobs"]):
        raise RuntimeError("Budget allocation cannot change during an accelerator job")
    limits = stage_limits(root)
    used = {key: sum(j["elapsed_seconds"] for j in ledger["jobs"] if j["stage"] == key) for key in limits}
    needed = max(0, math.ceil(1.2 * estimate + 5 - (limits[stage] - used[stage])))
    if not needed:
        return
    training_jobs = {f"train/{arm}/{seed}" for arm in ARMS for seed in (0, 1)}
    completed_jobs = {j["name"] for j in ledger["jobs"] if j["status"] == "complete"}
    finished = {
        "diagnostics": (root / "offline_final_complete.json").exists(),
        "training": training_jobs <= completed_jobs,
        "development": (root / "development_complete.json").exists() and stage == "confirmation",
    }
    donors = [key for key, complete in finished.items() if complete and key != stage]
    available = {key: max(0, math.floor(limits[key] - used[key])) for key in donors}
    if sum(available.values()) < needed:
        raise TimeoutError(f"No completed-stage allocation is available for {stage} evaluation")
    path = root / "budget_adjustments.json"
    history = read_json(path)["history"] if path.exists() else []
    for donor, seconds in available.items():
        transfer = min(seconds, needed)
        if transfer:
            limits[donor] -= transfer
            limits[stage] += transfer
            history.append(
                dict(
                    from_stage=donor,
                    to_stage=stage,
                    seconds=transfer,
                    used_seconds=used,
                    reason="Use unused completed-stage allocation for planned evaluation; preserve nine-hour total",
                )
            )
            needed -= transfer
    atomic_json(path, dict(limits=limits, history=history))


@contextlib.contextmanager
def accelerator_job(root, name, stage, estimate):
    """One accelerator process at a time; failed/partial jobs are charged too."""
    import fcntl

    with (Path(root) / "accelerator.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("An accelerator job is already running") from exc
        try:
            with _accounted_accelerator_job(root, name, stage, estimate) as remaining:
                yield remaining
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


@contextlib.contextmanager
def _accounted_accelerator_job(root, name, stage, estimate):
    path = Path(root) / "compute.json"
    ledger = read_json(path) if path.exists() else {"limit_seconds": 32400, "jobs": []}
    if any(job["status"] == "running" for job in ledger["jobs"]):
        raise RuntimeError("An accelerator job is already running or requires explicit crash accounting")
    if not any(job["name"] == "administrative_overhead_reserve" for job in ledger["jobs"]):
        ledger["jobs"].append(
            dict(name="administrative_overhead_reserve", stage="diagnostics", status="complete", elapsed_seconds=120.0)
        )
    used = sum(job["elapsed_seconds"] for job in ledger["jobs"])
    stage_used = sum(job["elapsed_seconds"] for job in ledger["jobs"] if job["stage"] == stage)
    reserve = 0 if stage == "confirmation" else 9000
    remaining = min(stage_limits(root)[stage] - stage_used, 32400 - used - reserve)
    if 1.2 * estimate > remaining:
        raise TimeoutError(f"Cannot start {name}: estimate with 20% margin exceeds {remaining:.1f}s remaining")
    job = dict(
        name=name,
        stage=stage,
        status="running",
        elapsed_seconds=0,
        pid=os.getpid(),
        started_unix=time.time(),
        maximum_seconds=remaining,
    )
    from .transfer import source_hashes

    sources = source_hashes()
    digest = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    snapshot = Path(root) / "source_snapshots" / f"{digest}.json"
    snapshot.parent.mkdir(exist_ok=True)
    if not snapshot.exists():
        write_json(snapshot, sources)
    job["source_snapshot"] = str(snapshot)
    ledger["jobs"].append(job)
    atomic_json(path, ledger)
    started = time.monotonic()

    def timeout(signum, frame):
        raise TimeoutError(f"Accelerator budget reached in {name}")

    previous = signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, max(1, remaining - 5))
    try:
        yield remaining - 5
        torch.mps.synchronize()
        job["status"] = "complete"
    except BaseException as exc:
        job.update(status="failed", error=repr(exc))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        job["elapsed_seconds"] = time.monotonic() - started
        ledger["elapsed_seconds"] = sum(row["elapsed_seconds"] for row in ledger["jobs"])
        atomic_json(path, ledger)


def hand_details(dist, actions):
    """Conditional probabilities and the actual valid candidate support."""
    if getattr(getattr(dist, "output", None), "hand_subset_energies", None) is not None:
        raise ValueError("Legacy mixture ceilings and responsibilities do not apply to subset residual policies")
    size = len(actions)
    result = {
        name: torch.zeros(size, device=actions.device)
        for name in (
            "is_hand",
            "candidate_covered",
            "has_candidates",
            "candidate_probability",
            "ar_probability",
            "conditional_probability",
            "conditional_accuracy",
            "nll_floor",
            "ar_responsibility",
            "candidate_count",
            "single_candidate_conflict",
        )
    }
    full_logp = dist.log_prob(actions)
    for is_play, family in ((True, ActionType.PLAY_SUBSET), (False, ActionType.DISCARD_SUBSET)):
        from .action import ActionRange

        base = int(ActionRange.PLAY_SUBSET_START if is_play else ActionRange.DISCARD_SUBSET_START)
        end = int(ActionRange.PLAY_SUBSET_END if is_play else ActionRange.DISCARD_SUBSET_END)
        selected = (actions >= base) & (actions <= end)
        _, valid, mapped = dist._candidate_distribution(is_play=is_play)
        matches = (mapped == actions[:, None]) & valid
        has = valid.any(-1)
        covered = matches.any(-1)
        ordered = mapped.masked_fill(~valid, -1).sort(dim=-1).values
        distinct = torch.cat(
            (torch.ones_like(ordered[:, :1], dtype=torch.bool), ordered[:, 1:] != ordered[:, :-1]), dim=-1
        )
        candidate_count = ((ordered >= 0) & distinct).sum(-1)
        # Reconstruct distributions from the same outputs without a second neural forward.
        from .action_grammar import ActionGrammarDistribution

        components = [
            ActionGrammarDistribution(
                dist.output, dist.action_mask, tokens=dist.tokens, temperature=dist.temperature, hand_ar_mixture_eps=eps
            )
            for eps in (0.0, 1.0)
        ]
        from .action_grammar import _masked_log_softmax

        macro = _masked_log_softmax(dist._t(dist.output.macro_logits), dist.macro_mask)[
            :, ACTION_TYPE_TO_GRAMMAR_INDEX[family]
        ]
        cand = (components[0].log_prob(actions) - macro).exp()
        cand = torch.where(has & covered, cand, torch.zeros_like(cand))
        ar = (components[1].log_prob(actions) - macro).exp()
        cond = (full_logp - macro).exp()
        cond = torch.nan_to_num(cond, nan=0.0, posinf=0.0)
        effective_eps = torch.where(has, dist.hand_ar_mixture_eps, 1.0)
        floor = torch.where(has & ~covered, -effective_eps.clamp_min(1e-30).log(), 0.0)
        values = dict(
            is_hand=torch.ones_like(cond),
            candidate_covered=covered,
            has_candidates=has,
            candidate_probability=cand,
            ar_probability=ar,
            conditional_probability=cond,
            conditional_accuracy=dist._hand_probabilities(is_play=is_play).argmax(-1) == actions - base,
            nll_floor=floor,
            ar_responsibility=effective_eps * ar / cond.clamp_min(1e-30),
            candidate_count=candidate_count,
            single_candidate_conflict=(candidate_count == 1) & ~covered,
        )
        for key, value in values.items():
            result[key] = torch.where(selected, value.float(), result[key])
    return result


def summarize_rows(rows):
    groups = defaultdict(list)
    for row in rows:
        groups["overall"].append(row)
        groups["family/" + row["family"]].append(row)
        if row["is_hand"] and "candidate_covered" in row:
            groups["hand/covered" if row["candidate_covered"] else "hand/uncovered"].append(row)
        groups[f"ante/{row['ante']}"].append(row)
        groups[f"outcome/{row['won']}"].append(row)
    metrics = [key for key in rows[0] if key not in {"seed", "step", "family", "ante", "won"}]
    result = {}
    hand_keys = {
        "candidate_covered",
        "has_candidates",
        "candidate_probability",
        "ar_probability",
        "conditional_probability",
        "conditional_accuracy",
        "ar_responsibility",
        "candidate_count",
        "single_candidate_conflict",
    }
    for name, group in groups.items():
        games = defaultdict(list)
        for row in group:
            games[row["seed"]].append(row)
        weighted = {}
        for metric in metrics:
            relevant = [row for row in group if metric not in hand_keys or row["is_hand"]]
            by_game = [
                [row[metric] for row in records if metric not in hand_keys or row["is_hand"]]
                for records in games.values()
            ]
            weighted[metric] = {
                "decision": float(np.mean([r[metric] for r in relevant])) if relevant else None,
                "game": float(np.mean([np.mean(v) for v in by_game if v])) if relevant else None,
            }
        result[name] = dict(decisions=len(group), games=len(games), metrics=weighted)
    return result


def measure(model, records, device, output, *, detailed=False, batch_size=320):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    model.eval()
    started = time.monotonic()
    manifest = dict(
        kind="offline_action_prediction",
        status="running",
        agent_config=asdict(model.config),
        device=str(device),
        temperature=1.0,
        policy_mode="evaluation",
        batch_size=batch_size,
        decisions=len(records),
        detailed=detailed,
    )
    write_json(output / "manifest.json", manifest)
    with torch.inference_mode(), (output / "predictions.jsonl").open("x", buffering=1) as stream:
        for start in range(0, len(records), batch_size):
            chunk = records[start : start + batch_size]
            batch = _collate_batch(chunk, device, SupervisedConfig())
            dist, _ = _grammar_distribution(model, batch)
            actions = batch["actions"]
            logp, predicted = dist.log_prob(actions), dist.mode()
            macro_predicted = dist.action_type_probs.argmax(-1).cpu().numpy()
            if not torch.isfinite(logp).all() or (logp < -1e7).any():
                raise RuntimeError("Nonfinite or unreachable teacher action")
            details = hand_details(dist, actions) if detailed else {}
            entropy = dist.entropy() if detailed else torch.zeros_like(logp)
            converted = {key: value.cpu().numpy() for key, value in details.items()}
            logp, predicted, entropy = logp.cpu().numpy(), predicted.cpu().numpy(), entropy.cpu().numpy()
            for index, record in enumerate(chunk):
                family = decode_action(int(record["action"])).action_type
                predicted_family = decode_action(int(predicted[index])).action_type
                row = dict(
                    seed=int(record["seed"]),
                    step=int(record["step"]),
                    family=family.value,
                    ante=int(record["obs"]["scalars"][2]),
                    won=bool(record.get("won", False)),
                    nll=-float(logp[index]),
                    probability=float(np.exp(logp[index])),
                    accuracy=float(predicted[index] == record["action"]),
                    macro_accuracy=float(predicted_family == family),
                    marginal_macro_accuracy=float(macro_predicted[index] == ACTION_TYPE_TO_GRAMMAR_INDEX[family]),
                    is_hand=float(family in (ActionType.PLAY_SUBSET, ActionType.DISCARD_SUBSET)),
                )
                if detailed:
                    row.update({key: float(value[index]) for key, value in converted.items()})
                    row.update(entropy=float(entropy[index]), excess_nll=row["nll"] - row["nll_floor"])
                rows.append(row)
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            if start % (batch_size * 20) == 0:
                print("measure", str(output), start + len(chunk), len(records), flush=True)
    summary = dict(
        strata=summarize_rows(rows),
        elapsed_seconds=time.monotonic() - started,
        decisions=len(rows),
        record_ids_sha256=hashlib.sha256(json.dumps([(r["seed"], r["step"]) for r in rows]).encode()).hexdigest(),
    )
    write_json(output / "summary.json", summary)
    manifest.update(
        status="complete", elapsed_seconds=summary["elapsed_seconds"], record_ids_sha256=summary["record_ids_sha256"]
    )
    write_json(output / "manifest.json", manifest)
    return summary


def balanced_probe(records, per_family=160):
    rng = np.random.default_rng(1729)
    selected = []
    hashes = set()
    for family in (ActionType.PLAY_SUBSET, ActionType.DISCARD_SUBSET):
        eligible = [r for r in records if decode_action(int(r["action"])).action_type == family]
        count = 0
        for index in rng.permutation(len(eligible)):
            row = eligible[index]
            digest = observation_digest(row["obs"])
            if digest in hashes:
                continue
            hashes.add(digest)
            selected.append(row)
            count += 1
            if count == per_family:
                break
        if count != per_family:
            raise ValueError("Insufficient distinct hand observations")
    return selected


def observation_digest(obs):
    digest = hashlib.sha256()
    for key, value in sorted(obs.items()):
        array = np.ascontiguousarray(value)
        digest.update(key.encode())
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def collision_audit(records):
    groups = defaultdict(list)
    for row in records:
        groups[observation_digest(row["obs"])].append((int(row["seed"]), int(row["step"]), int(row["action"])))
    return dict(
        records=len(records),
        unique_observations=len(groups),
        conflicting=[
            dict(observation_sha256=key, labels=values)
            for key, values in groups.items()
            if len({v[2] for v in values}) > 1
        ],
    )
