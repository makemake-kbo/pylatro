"""Conservative, supervisor-scoped PT/PPO health reviews and stop requests.

No model loading or GPU work. Reads event files directly, independently of the
dashboard, and never restarts training or deletes checkpoints. Run under the
same private supervisor as the job, with its expected launcher PID pinned.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path

logger = logging.getLogger(__name__)
# Points are (wall_time, step, value). Deliberately exclude diagnostic NaNs such
# as explained variance when milestone returns have zero variance.
CORE_TAGS = {
    "supervised": ("train/loss", "train/action_loss", "train/value_loss", "train/grad_norm"),
    "ppo": ("ppo/policy_loss", "ppo/approx_kl", "ppo/entropy_raw", "critic/return_huber"),
}
REVIEW_TAGS = (
    "epoch/loss",
    "epoch/accuracy",
    "train/accuracy",
    "ppo/minibatch_fraction",
    "ppo/entropy_normalized",
    "eval/win_rate",
    "recent_100/stall_rate",
    "recent_100/final_ante_mean",
    "curriculum/fresh_win_rate",
    "curriculum/archive_win_rate",
    "curriculum/survive_ante5",
    "curriculum/survive_ante6",
    "archive/states",
)


@dataclass(frozen=True)
class MonitorPolicy:
    poll_seconds: int = 60
    stall_seconds: int = 1800
    stop_grace_seconds: int = 600
    min_free_gib: float = 2.0
    eval_drop_floor: float = 0.02
    eval_drop_fraction: float = 0.5
    eval_patience: int = 3
    kl_limit: float = 0.25
    kl_patience: int = 3
    stall_rate_limit: float = 0.8
    stall_rate_patience: int = 10
    stall_rate_min_update: int = 20


def _ordered(points):
    # Repeated polling (or duplicate event writes) must not count as more evals.
    return sorted({p[1]: p for p in points}.values(), key=lambda p: p[1])


def quality_stop_reason(phase: str, series: dict, policy: MonitorPolicy) -> str | None:
    for tag in CORE_TAGS[phase]:
        if any(not math.isfinite(p[2]) for p in series.get(tag, ())):
            return f"Nonfinite core training metric: {tag}"
    if phase == "supervised":
        losses = _ordered(series.get("epoch/loss", ()))
        accuracies = {p[1]: p[2] for p in series.get("epoch/accuracy", ())}
        best_loss, best_acc, streak = math.inf, 0.0, 0
        for _, step, loss in losses:
            acc = accuracies.get(step)
            bad = acc is not None and loss >= 1.5 * best_loss and acc <= best_acc - 0.15
            streak = streak + 1 if bad else 0
            best_loss = min(best_loss, loss)
            if acc is not None:
                best_acc = max(best_acc, acc)
        if streak >= 2:
            return "PT loss rose >=50% and accuracy fell >=15 points for two complete epochs"
        return None

    best, streak = 0.0, 0
    for _, _, rate in _ordered(series.get("eval/win_rate", ())):
        if not math.isfinite(rate):
            continue  # Missing eval is not evidence of poor play.
        best = max(best, rate)
        threshold = max(policy.eval_drop_floor, best * policy.eval_drop_fraction)
        streak = streak + 1 if best - rate >= threshold - 1e-9 else 0
    if streak >= policy.eval_patience:
        return f"Fresh-run win rate regressed by >=max(2 percentage points, 50% of best) for {streak} evals"

    for tag, limit, patience, min_step in (
        ("ppo/approx_kl", policy.kl_limit, policy.kl_patience, 1),
        ("recent_100/stall_rate", policy.stall_rate_limit, policy.stall_rate_patience, policy.stall_rate_min_update),
    ):
        points = [p for p in _ordered(series.get(tag, ())) if p[1] >= min_step]
        tail = points[-patience:]
        if (
            len(tail) == patience
            and all(math.isfinite(p[2]) and p[2] >= limit for p in tail)
            and all(b[1] == a[1] + 1 for a, b in pairwise(tail))
        ):
            return f"{tag} >= {limit} for {patience} consecutive updates"
    # Zero wins alone, low entropy alone, shaped return, and archive-start wins
    # are deliberately not stopping criteria for a sparse Ante-8 experiment.
    return None


def eta_seconds(points, target: int) -> float | None:
    points = _ordered(points)
    if len(points) < 2:
        return None
    # A PPO window >=20 updates includes periodic eval/checkpoint overhead;
    # before that, label the estimate preliminary in the review.
    first, last = points[max(0, len(points) - 51)], points[-1]
    if last[1] <= first[1] or last[0] <= first[0]:
        return None
    return max(0, target - last[1]) * (last[0] - first[0]) / (last[1] - first[1])


class EventReader:
    def __init__(self):
        self.readers = {}

    def read(self, path: Path, phase: str) -> dict:
        if not path.exists():
            return {}
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        if path not in self.readers:
            self.readers[path] = EventAccumulator(str(path), size_guidance={"scalars": 0})
        reader = self.readers[path]
        reader.Reload()
        tags = set(reader.Tags()["scalars"])
        return {
            tag: [(v.wall_time, v.step, v.value) for v in reader.Scalars(tag)]
            for tag in (*CORE_TAGS[phase], *REVIEW_TAGS)
            if tag in tags
        }


def _process(config: Path) -> tuple[str, int]:
    result = subprocess.run(
        ["/usr/local/bin/supervisorctl", "-c", str(config), "status", "training"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    match = re.fullmatch(r"training\s+(\w+)(?:\s+pid (\d+),[^\n]*|[^\n]*)", result.stdout.strip())
    if not match:
        raise RuntimeError(f"Cannot establish training identity: {result.stdout} {result.stderr}")
    return match[1], int(match[2] or 0)


def _force_stop(config: Path, expected_pid: int) -> None:
    _, pid = _process(config)
    if pid != expected_pid:
        raise RuntimeError("Training PID changed; refusing to stop another process")
    subprocess.run(
        ["/usr/local/bin/supervisorctl", "-c", str(config), "stop", "training"],
        check=True,
        timeout=150,
    )


def _write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def monitor(root: Path, supervisor: Path, expected_pid: int, policy: MonitorPolicy) -> None:
    manifest = json.loads((root / "manifest.json").read_text())
    reader = EventReader()
    request_path = root / "monitor_stop_request.json"
    status_path = root / "monitor_status.json"
    started = time.time()
    _write_json(
        root / "monitor_policy.json",
        {
            **asdict(policy),
            "expected_pid": expected_pid,
            "supervisor": str(supervisor),
            "scope": "Stop this training pipeline only; never stop or destroy the rented instance",
            "review_kind": "Automated metric rules, not continuous conversational model review",
        },
    )
    while True:
        try:
            now = time.time()
            state, pid = _process(supervisor)
            if pid and pid != expected_pid:
                _write_json(status_path, {"state": "identity_changed", "pid": pid, "time": now})
                return
            if state not in ("RUNNING", "STARTING", "STOPPING"):
                _write_json(
                    status_path,
                    {
                        "state": "stopped" if request_path.exists() else "training_exited",
                        "supervisor_state": state,
                        "time": now,
                        "pipeline": json.loads((root / "status.json").read_text()),
                    },
                )
                logger.info("Training exited (%s); monitor finished", state)
                return
            phase = json.loads((root / "status.json").read_text())["stage"]
            if phase not in CORE_TAGS:
                time.sleep(policy.poll_seconds)
                continue
            series = reader.read(root / "events" / phase, phase)
            reason = quality_stop_reason(phase, series, policy)
            free = shutil.disk_usage(root).free / 1024**3
            if free < policy.min_free_gib:
                reason = f"Checkpoint disk reserve low: {free:.2f} GiB free"
            newest = max(
                [
                    started,
                    (root.parent / "training.log").stat().st_mtime,
                    (root / "status.json").stat().st_mtime,
                    *[p[0] for values in series.values() for p in values],
                ]
            )
            hung = now - newest > policy.stall_seconds
            if hung:
                reason = f"No metric or log progress for {policy.stall_seconds / 60:.0f} minutes"
            if reason and not request_path.exists():
                _write_json(request_path, {"reason": reason, "time": now, "phase": phase})
                logger.error("Stop requested: %s", reason)
            if request_path.exists():
                request = json.loads(request_path.read_text())
                # Running PT has no cooperative request hook. PPO saves at its
                # next completed update; forced stop is the bounded fallback.
                if phase == "supervised" or hung or now - request["time"] > policy.stop_grace_seconds:
                    _force_stop(supervisor, expected_pid)
            key = "train/loss" if phase == "supervised" else "ppo/policy_loss"
            points = series.get(key, [])
            if phase == "supervised":
                sup = manifest["supervised"]
                # The generator logs actual record count as a separate scalar.
                acc = reader.readers.get(root / "events" / phase)
                n = acc.Scalars("data/num_records")[-1].value if acc else 0
                target = math.ceil(n / sup["batch_size"]) * sup["max_epochs"] - 1
            else:
                target = manifest["ppo"]["total_updates"]
            report = {
                "state": "stop_requested" if request_path.exists() else "watching",
                "time": now,
                "phase": phase,
                "pid": pid,
                "latest_step": points[-1][1] if points else None,
                "phase_eta_seconds": eta_seconds(points, target),
                "eta_preliminary": phase == "ppo" and (not points or points[-1][1] < 20),
                "free_disk_gib": round(free, 2),
                "metrics": {
                    tag: values[-1][2] if math.isfinite(values[-1][2]) else None
                    for tag, values in series.items()
                    if values
                },
            }
            _write_json(status_path, report)
            with (root / "monitor_reviews.jsonl").open("a") as handle:
                handle.write(json.dumps(report, allow_nan=False) + "\n")
            logger.info(
                "%s step=%s ETA=%s minutes state=%s",
                phase,
                report["latest_step"],
                round(report["phase_eta_seconds"] / 60, 1) if report["phase_eta_seconds"] else None,
                report["state"],
            )
        except Exception:
            # An unreadable dashboard/event file is not proof of bad learning.
            # Surface the monitoring error without killing unrelated work.
            logger.exception("Monitor poll failed; no new quality decision this poll")
        time.sleep(policy.poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--supervisor-config", type=Path, required=True)
    parser.add_argument("--expected-pid", type=int, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    monitor(args.run_dir.resolve(), args.supervisor_config.resolve(), args.expected_pid, MonitorPolicy())


if __name__ == "__main__":
    main()
