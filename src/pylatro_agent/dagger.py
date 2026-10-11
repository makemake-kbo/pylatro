"""Iterative DAgger from the V68 heuristic on public-information-v2 student states.

Each round plays complete games in which every executed action is the
teacher's with probability beta and the student's otherwise, labels every
visited public decision with the V68 teacher, adds those labels to the
aggregate dataset, fine-tunes the student on the whole aggregate, and
evaluates the result on one fixed development panel. Round 0 evaluates the
starting checkpoint on the same panel, so every later round is a paired
comparison against it.

Labels are teacher actions on states the student reaches. Game outcomes only
supervise the critic, as the value of the executed (mixture) policy; they
never weight or select imitation labels.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import torch

from .action import decode_action
from .checkpoint import save_checkpoint
from .constants import NUM_ACTIONS
from .heuristic import HeuristicAgent
from .public_runtime import addressable_hand
from .public_state import OBSERVATION_INTERPRETATION, public_mask, simulation_state
from .reward import reward_checkpoint_metadata
from .survival import terminal_outcome_class
from .training.evaluation_envs import EvaluationEnvs, _Environments
from .training.ppo_observations import _OBSERVATION_SPEC, _obs_dicts_to_batch
from .training.ppo_policy import _grammar_distribution
from .training.public_training import PublicTrainingEnv
from .transfer import _bootstrap, load_policy, paired_statistics, sha256, summarize
from .value_head import outcome_nll

NAMESPACE = "dagger-v68-v1"
TEACHER = {"shop_policy": "search", "grow_scalers": True}
TEACHER_VIEWS = ("full", "public")
STUDENT_ACTIONS = ("sample", "greedy")
PANEL_MODES = ("greedy", "sample")
# Fields that may change when resuming an existing run directory. Everything
# else defines the experiment and must match the saved configuration.
RESUMABLE_FIELDS = frozenset({"rounds", "workers", "device", "collection_device", "inference_batch", "eval_every"})
# Teacher failures are recorded and skipped, but a run that loses more than
# this fraction of labels in a round is broken, not unlucky.
MAX_LABEL_FAILURE_FRACTION = 0.01


@dataclass(frozen=True)
class DaggerConfig:
    rounds: int = 8
    games_per_round: int = 512
    # Round r (1-based) executes the teacher's action with probability
    # beta_initial * beta_decay ** (r - 1), and the student's otherwise.
    beta_initial: float = 0.5
    beta_decay: float = 0.5
    # How the student acts during collection: "sample" explores more states;
    # "greedy" visits exactly the states the deployed policy reaches.
    student_action: str = "sample"
    temperature: float = 1.0
    # "full" labels with V68 as measured (live engine state, including exact
    # scoring randomness and face-down identities). "public" labels the same
    # public projection the student observes.
    teacher_view: str = "full"
    panel_games: int = 1000
    panel_modes: tuple[str, ...] = ("greedy",)
    eval_every: int = 1
    updates_per_round: int = 1000
    batch_size: int = 256
    micro_batch_size: int = 64
    lr: float = 3e-5
    weight_decay: float = 0.01
    warmup_updates: int = 100
    max_grad_norm: float = 1.0
    outcome_loss_coeff: float = 0.1
    shard_rows: int = 2048
    pool_shards: int = 8
    workers: int = 8
    inference_batch: int = 64
    max_episode_steps: int = 2048
    deck_key: str = "b_blue"
    stake: int = 1
    win_ante: int = 8
    seed: int = 0
    device: str = "cpu"
    # Inference device for collection and evaluation; defaults to ``device``.
    collection_device: str | None = None

    def __post_init__(self):
        if self.rounds < 1 or self.games_per_round < 1 or self.panel_games < 1 or self.updates_per_round < 1:
            raise ValueError("Rounds, games, panel size and updates must be positive")
        if not (0.0 <= self.beta_initial <= 1.0 and 0.0 <= self.beta_decay <= 1.0):
            raise ValueError("beta_initial and beta_decay must lie in [0, 1]")
        if self.student_action not in STUDENT_ACTIONS:
            raise ValueError(f"student_action must be one of {STUDENT_ACTIONS}")
        if self.teacher_view not in TEACHER_VIEWS:
            raise ValueError(f"teacher_view must be one of {TEACHER_VIEWS}")
        if not self.panel_modes or any(mode not in PANEL_MODES for mode in self.panel_modes):
            raise ValueError(f"panel_modes must be a nonempty subset of {PANEL_MODES}")
        if self.batch_size < 1 or self.micro_batch_size < 1 or self.batch_size % self.micro_batch_size:
            raise ValueError("batch_size must be a positive multiple of micro_batch_size")
        if not (self.temperature > 0 and self.lr > 0 and self.max_grad_norm > 0):
            raise ValueError("temperature, lr and max_grad_norm must be positive")
        if self.outcome_loss_coeff < 0 or self.warmup_updates < 0 or self.eval_every < 1:
            raise ValueError("outcome_loss_coeff and warmup_updates must be nonnegative; eval_every positive")
        if self.shard_rows < 1 or self.pool_shards < 1 or self.workers < 0 or self.inference_batch < 1:
            raise ValueError("Invalid shard, pool, worker or inference batch size")

    def beta(self, round_index):
        return self.beta_initial * self.beta_decay ** (round_index - 1)

    def to_json(self):
        return {**asdict(self), "panel_modes": list(self.panel_modes)}

    @classmethod
    def from_json(cls, values):
        known = {f.name for f in fields(cls)}
        return cls(**{**{k: v for k, v in values.items() if k in known}, "panel_modes": tuple(values["panel_modes"])})


# ---------------------------------------------------------------------------
# Seeds


def hashed_seeds(label, count, *, exclude=()):
    """Fresh int32 game seeds from a private namespace.

    The compiled executor accepts signed 32-bit integers. Large hashed values
    avoid the low integer ranges used by historical studies.
    """
    used, seeds, cursor = set(exclude), [], 0
    while len(seeds) < count:
        digest = hashlib.sha256(f"{NAMESPACE}:{label}:{cursor}".encode()).digest()
        seed = 1_000_000_000 + int.from_bytes(digest[:8], "big") % 1_147_483_647
        cursor += 1
        if seed not in used:
            seeds.append(seed)
            used.add(seed)
    return seeds


def load_exclusions(paths):
    """Every integer in the given JSON/JSONL files, e.g. seed lists or study ``seeds.json`` panels.

    Excluding an integer that was never a seed costs nothing, so no schema is assumed.
    """
    seeds = set()

    def gather(value):
        if isinstance(value, bool):
            return
        if isinstance(value, int):
            seeds.add(value)
        elif isinstance(value, dict):
            for item in value.values():
                gather(item)
        elif isinstance(value, list):
            for item in value:
                gather(item)

    for path in paths:
        text = Path(path).read_text()
        for value in (
            [json.loads(line) for line in text.splitlines() if line]
            if str(path).endswith(".jsonl")
            else [json.loads(text)]
        ):
            gather(value)
    return seeds


class SeedPlan:
    """The fixed panel plus per-round training seeds, persisted on first use.

    Exclusions are stored once beside the plan. Lists supplied on a resume
    are added for future rounds, and seeds already drawn must avoid them too.
    """

    def __init__(self, path, config, exclusions=()):
        self.path = Path(path)
        self.exclusion_path = self.path.with_name("excluded_seeds.json")
        saved_exclusions = set(read_json(self.exclusion_path)) if self.exclusion_path.exists() else set()
        self.exclusions = saved_exclusions | set(exclusions)
        if self.path.exists():
            saved = read_json(self.path)
            self.panel = saved["panel"]
            self.rounds = {int(k): v for k, v in saved["rounds"].items()}
            if len(self.panel) != config.panel_games:
                raise ValueError("Saved panel size differs from the configuration")
            self._check()
        else:
            self.panel = hashed_seeds("panel", config.panel_games, exclude=self.exclusions)
            self.rounds = {}
        # Persist new exclusions only once they are known to be consistent.
        if self.exclusions != saved_exclusions:
            atomic_json(self.exclusion_path, sorted(self.exclusions))
        if not self.path.exists():
            self._save()
        self.games_per_round = config.games_per_round

    def round(self, index):
        if index not in self.rounds:
            used = self.exclusions | set(self.panel) | {s for seeds in self.rounds.values() for s in seeds}
            self.rounds[index] = hashed_seeds(f"train:{index}", self.games_per_round, exclude=used)
            self._save()
        return self.rounds[index]

    def _check(self):
        flat = self.panel + [s for seeds in self.rounds.values() for s in seeds]
        if len(set(flat)) != len(flat) or set(flat) & self.exclusions:
            raise ValueError("DAgger seed collision with another panel or an excluded seed")

    def _save(self):
        self._check()
        atomic_json(
            self.path,
            {
                "namespace": NAMESPACE,
                "panel": self.panel,
                "rounds": {str(k): v for k, v in sorted(self.rounds.items())},
                "excluded_seed_count": len(self.exclusions),
            },
        )


# ---------------------------------------------------------------------------
# Teacher labels (worker side)


def teacher_label(env, teacher, view):
    """Return the V68 action for a live public game without changing it.

    The label is legal under the public mask the student acts with. ``full``
    queries a deep copy of the engine state; ``public`` queries the student's
    public projection.
    """
    runner = env.runner
    if view == "full":
        state = copy.deepcopy(runner.state, {id(runner.state.data): runner.state.data})
        mask = runner.compute_mask()
    elif view == "public":
        state = simulation_state(runner.snapshot(), runner.state.data)
        mask = public_mask(state, runner.sub_phase)
    else:
        raise ValueError(f"Unknown teacher view {view!r}")
    started = time.perf_counter()
    try:
        # Bound card-slot proposals to the action vocabulary for oversized hands.
        with addressable_hand(state):
            action = int(teacher.select_action(state, runner.sub_phase, mask, round_score=runner.round_score))
    except Exception as exc:  # One failed label must not end a long collection round.
        return {"action": -1, "seconds": time.perf_counter() - started, "error": f"{type(exc).__name__}: {exc}"}
    seconds = time.perf_counter() - started
    if not (0 <= action < NUM_ACTIONS and mask[action]):
        return {"action": -1, "seconds": seconds, "error": f"illegal teacher action {action}"}
    return {"action": action, "seconds": seconds}


def _dagger_teacher(self, requests):
    teachers = self.__dict__.setdefault("_dagger_teachers", {})
    for seed in [s for s in teachers if s not in self.envs]:
        del teachers[seed]
    return [
        (seed, teacher_label(self.envs[seed], teachers.setdefault(seed, HeuristicAgent(**TEACHER)), view))
        for seed, view in requests
    ]


def _install_commands():
    _Environments.dagger_teacher = _dagger_teacher


def _dagger_worker(connection):
    from .training.evaluation_envs import _worker

    _install_commands()
    _worker(connection)


class DaggerEnvs(EvaluationEnvs):
    """Public training envs whose workers also answer V68 label requests."""

    def __init__(self, workers, **kwargs):
        from .training import evaluation_envs

        original = evaluation_envs._worker
        evaluation_envs._worker = _dagger_worker
        try:
            _install_commands()
            super().__init__(workers, environment_class=PublicTrainingEnv, **kwargs)
        finally:
            evaluation_envs._worker = original

    def request_labels(self, requests):
        """Dispatch label requests without waiting, so inference overlaps them."""
        if self.local is not None:
            return dict(self.local.dagger_teacher(requests))
        shards = {}
        for seed, view in requests:
            shards.setdefault(self.owners[seed], []).append((seed, view))
        for index, shard in shards.items():
            self.connections[index].send(("dagger_teacher", shard))
        return list(shards)

    def receive_labels(self, pending):
        if isinstance(pending, dict):
            return pending
        labels = {}
        for index in pending:
            labels.update(self._receive(index))
        return labels


def make_envs(config, data, vocab):
    return DaggerEnvs(
        config.workers,
        data=data,
        vocab=vocab,
        deck_key=config.deck_key,
        stake=config.stake,
        win_ante=config.win_ante,
        max_steps=config.max_episode_steps,
    )


# ---------------------------------------------------------------------------
# Rollouts


def play(model, envs, seeds, config, *, device, mode, beta=0.0, rng=None, teacher_view=None, on_decision=None):
    """Play complete games with central batched inference; return outcomes in seed order.

    With ``teacher_view`` set, every decision is labelled by V68 and the
    teacher's action is executed with probability ``beta``. ``on_decision``
    receives each decision before it is executed; game outcomes come back via
    the return value and ``on_decision(..., outcome=...)`` at game end.
    """
    if mode not in PANEL_MODES:
        raise ValueError(f"Unknown play mode {mode!r}")
    if beta > 0 and (teacher_view is None or rng is None):
        raise ValueError("Teacher execution requires labels and an RNG")
    pending = iter(seeds)
    live, steps, outcomes = {}, {}, {}

    def start():
        seed = next(pending, None)
        if seed is not None:
            obs, _phase = envs.reset(seed)
            live[seed], steps[seed] = obs, 0

    for _ in range(min(config.inference_batch, len(seeds))):
        start()
    model.eval()
    while live:
        order = list(live)
        handle = envs.request_labels([(seed, teacher_view) for seed in order]) if teacher_view else None
        with torch.inference_mode():
            batch = _obs_dicts_to_batch([live[seed] for seed in order], device)
            # Greedy play and label likelihoods always use temperature 1: with
            # the candidate/AR hand mixture, the argmax depends on temperature.
            dist, _ = _grammar_distribution(model, batch)
            greedy = dist.mode()
            if mode == "greedy":
                student = greedy
            elif config.temperature == 1.0:
                student = dist.sample()
            else:
                student = _grammar_distribution(model, batch, temperature=config.temperature)[0].sample()
            greedy_actions = greedy.cpu().numpy()
            labels = envs.receive_labels(handle) if teacher_view else None
            teacher_logp = None
            if labels is not None:
                # Failed labels borrow the (legal) greedy action; their likelihood is never used.
                targets = [labels[seed]["action"] for seed in order]
                targets = torch.tensor(
                    [action if action >= 0 else int(greedy_actions[i]) for i, action in enumerate(targets)],
                    device=device,
                )
                teacher_logp = dist.log_prob(targets).cpu().numpy()
        greedy, student = greedy_actions, student.cpu().numpy()
        actions = []
        for i, seed in enumerate(order):
            label = labels[seed] if labels is not None else None
            executed, by_teacher = int(student[i]), False
            if label is not None and label["action"] >= 0 and beta > 0 and rng.random() < beta:
                executed, by_teacher = label["action"], True
            if on_decision is not None:
                on_decision(
                    seed,
                    steps[seed],
                    live[seed],
                    label=label,
                    student=int(student[i]),
                    greedy=int(greedy[i]),
                    executed=executed,
                    by_teacher=by_teacher,
                    teacher_logp=None if label is None or label["action"] < 0 else float(teacher_logp[i]),
                )
            actions.append((seed, executed))
            steps[seed] += 1
        results = envs.step(actions)
        for seed, _action in actions:
            obs, terminated, truncated, info, _seconds = results[seed]
            if terminated or truncated:
                del live[seed]
                cleared = [int(a) for a in info.get("cleared_antes", [])]
                outcomes[seed] = {
                    "seed": int(seed),
                    "won": bool(terminated and info.get("won")),
                    "final_ante": int(info["ante"]),
                    "max_ante": int(info["ante"]),
                    "cleared_antes": cleared,
                    "antes_cleared": len(cleared),
                    "steps": steps[seed],
                    "stalled": bool(truncated),
                }
                if on_decision is not None:
                    on_decision(seed, steps[seed], None, outcome=outcomes[seed])
                start()
            else:
                live[seed] = obs
    return [outcomes[seed] for seed in seeds]


def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


# ---------------------------------------------------------------------------
# Labelled shards


ROW_FIELDS = {
    "label": np.int64,
    "executed": np.int64,
    "student": np.int64,
    "greedy": np.int64,
    "by_teacher": bool,
    "seed": np.int64,
    "step": np.int32,
}


def compact_observation(obs):
    """Observation arrays as stored in shards; the 17k-entry action mask is bit-packed."""
    stored = {key: obs[key] for key in _OBSERVATION_SPEC if key != "action_mask"}
    stored["action_mask_packed"] = np.packbits(np.asarray(obs["action_mask"]).astype(bool))
    return stored


def write_shard(path, rows):
    """Write rows atomically as one compressed shard.

    Rows carry no outcome: shards are written while their games are still in
    progress. Critic targets are joined from the round's outcomes by seed.
    """
    arrays = {key: np.stack([row["obs"][key] for row in rows]) for key in rows[0]["obs"]}
    for key, dtype in ROW_FIELDS.items():
        arrays[key] = np.array([row[key] for row in rows], dtype)
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)
    return {"file": path.name, "rows": len(rows), "sha256": sha256(path)}


def read_shard(path):
    with np.load(path) as archive:
        return {key: archive[key] for key in archive.files}


def collate(arrays, device):
    """Model batch plus labels and critic targets from shard arrays."""
    batch = {}
    for key, (_shape, dtype) in _OBSERVATION_SPEC.items():
        if key == "action_mask":
            value = np.unpackbits(arrays["action_mask_packed"], axis=1, count=NUM_ACTIONS)
        else:
            value = arrays[key]
        batch[key] = torch.as_tensor(value).to(device=device, dtype=dtype)
    batch["labels"] = torch.as_tensor(arrays["label"], device=device, dtype=torch.long)
    batch["outcome_class"] = torch.as_tensor(arrays["outcome_class"], device=device, dtype=torch.long)
    batch["outcome_mask"] = torch.as_tensor(arrays["outcome_mask"], device=device, dtype=torch.float32)
    return batch


def outcome_target(outcome):
    """Categorical critic target and mask; stalled games carry no outcome label."""
    if outcome["stalled"]:
        return 0, 0.0
    return terminal_outcome_class(won=outcome["won"], final_ante=outcome["final_ante"]), 1.0


class ShardPool:
    """Sample rows from the aggregate while holding at most ``pool_shards`` shards in memory.

    Each refill draws shards without replacement, with probability
    proportional to their row counts, and the pool is consumed in a random
    order. Rows are therefore sampled uniformly from the aggregate in
    expectation, without loading every round at once.
    """

    def __init__(self, shards, outcomes, *, pool_shards, rng):
        self.paths = [Path(path) for path, _rows in shards]
        self.rows = np.array([rows for _path, rows in shards], dtype=np.float64)
        if not self.paths or self.rows.sum() <= 0:
            raise ValueError("The aggregate dataset is empty")
        self.targets = {int(seed): outcome_target(outcome) for seed, outcome in outcomes.items()}
        self.pool_shards = min(pool_shards, len(self.paths))
        self.rng = rng
        self.pool, self.order, self.cursor = None, None, 0

    def _refill(self):
        chosen = self.rng.choice(len(self.paths), size=self.pool_shards, replace=False, p=self.rows / self.rows.sum())
        loaded = [read_shard(self.paths[i]) for i in sorted(chosen)]
        self.pool = {key: np.concatenate([shard[key] for shard in loaded]) for key in loaded[0]}
        targets = [self.targets[int(seed)] for seed in self.pool["seed"]]
        self.pool["outcome_class"] = np.array([target for target, _mask in targets], np.int64)
        self.pool["outcome_mask"] = np.array([mask for _target, mask in targets], np.float32)
        self.order = self.rng.permutation(len(self.pool["label"]))
        self.cursor = 0

    def next_batch(self, size):
        if self.pool is None or self.cursor + size > len(self.order):
            self._refill()
        if len(self.order) < size:
            # A tiny aggregate cannot fill one batch without repetition.
            indices = self.rng.choice(self.order, size=size, replace=True)
        else:
            indices = self.order[self.cursor : self.cursor + size]
            self.cursor += size
        return {key: value[indices] for key, value in self.pool.items()}


# ---------------------------------------------------------------------------
# Collection


def collect_round(model, config, seeds, directory, *, round_index, device, data, vocab, log=print):
    """Play one round of student games and save every labelled decision."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    beta = config.beta(round_index)
    rng = np.random.default_rng([config.seed, round_index])
    seed_torch(config.seed * 1_000_003 + round_index)
    started = time.monotonic()
    buffer, shards, failures = [], [], []
    stats = Counter()
    families = {}
    teacher_seconds = 0.0
    nll = []

    def flush(force=False):
        nonlocal buffer
        while len(buffer) >= config.shard_rows or (force and buffer):
            rows, buffer = buffer[: config.shard_rows], buffer[config.shard_rows :]
            shards.append(write_shard(directory / f"shard_{len(shards):05d}.npz", rows))

    def on_decision(seed, step, obs, *, outcome=None, label=None, **decision):
        nonlocal teacher_seconds
        if outcome is not None:
            return
        stats["decisions"] += 1
        stats["executed_by_teacher"] += decision["by_teacher"]
        teacher_seconds += label["seconds"]
        if label["action"] < 0:
            stats["label_failures"] += 1
            if len(failures) < 20:
                failures.append({"seed": int(seed), "step": int(step), "error": label["error"]})
            return
        action = label["action"]
        if not obs["action_mask"][action]:
            # The worker labelled a different state than the one observed.
            raise RuntimeError(
                f"Teacher label {action} is illegal in the recorded observation (seed {seed}, step {step})"
            )
        family = decode_action(action).action_type.value
        counts = families.setdefault(family, Counter())
        counts["labels"] += 1
        counts["greedy_agreement"] += decision["greedy"] == action
        counts["student_agreement"] += decision["student"] == action
        stats["greedy_agreement"] += decision["greedy"] == action
        stats["student_agreement"] += decision["student"] == action
        nll.append(-decision["teacher_logp"])
        buffer.append(
            {
                "obs": compact_observation(obs),
                "label": action,
                "executed": decision["executed"],
                "student": decision["student"],
                "greedy": decision["greedy"],
                "by_teacher": decision["by_teacher"],
                "seed": int(seed),
                "step": int(step),
            }
        )
        flush()

    envs = make_envs(config, data, vocab)
    try:
        outcomes = play(
            model,
            envs,
            seeds,
            config,
            device=device,
            mode=config.student_action,
            beta=beta,
            rng=rng,
            teacher_view=config.teacher_view,
            on_decision=on_decision,
        )
    finally:
        envs.close()
    flush(force=True)
    if stats["label_failures"] > MAX_LABEL_FAILURE_FRACTION * max(stats["decisions"], 1):
        raise RuntimeError(f"{stats['label_failures']} of {stats['decisions']} teacher labels failed: {failures[:3]}")
    labelled = stats["decisions"] - stats["label_failures"]
    manifest = {
        "status": "complete",
        "round": round_index,
        "beta": beta,
        "student_action": config.student_action,
        "teacher_view": config.teacher_view,
        "teacher": TEACHER,
        "observation_mode": OBSERVATION_INTERPRETATION,
        "games": len(outcomes),
        "decisions": stats["decisions"],
        "labelled": labelled,
        "label_failures": stats["label_failures"],
        "label_failure_examples": failures,
        "executed_by_teacher": stats["executed_by_teacher"],
        # Agreement and NLL of the pre-update student on the states it visited.
        "greedy_agreement": stats["greedy_agreement"] / max(labelled, 1),
        "student_agreement": stats["student_agreement"] / max(labelled, 1),
        "teacher_nll": float(np.mean(nll)) if nll else None,
        "agreement_by_teacher_family": {
            family: {
                "labels": counts["labels"],
                "greedy_agreement": counts["greedy_agreement"] / counts["labels"],
                "student_agreement": counts["student_agreement"] / counts["labels"],
            }
            for family, counts in sorted(families.items())
        },
        # Outcomes of the executed mixture policy, not of the student alone.
        "behavior_wins": sum(o["won"] for o in outcomes),
        "behavior_mean_antes_cleared": float(np.mean([o["antes_cleared"] for o in outcomes])),
        "stalled_games": sum(o["stalled"] for o in outcomes),
        "teacher_seconds": teacher_seconds,
        "elapsed_seconds": time.monotonic() - started,
        "shards": shards,
    }
    with (directory / "outcomes.jsonl").open("w") as stream:
        for outcome in outcomes:
            stream.write(json.dumps(outcome) + "\n")
    # The manifest is written last: its presence marks a complete round.
    atomic_json(directory / "manifest.json", manifest)
    log(
        f"round {round_index}: {len(outcomes)} games, {labelled} labels, beta {beta:.3f}, "
        f"greedy agreement {manifest['greedy_agreement']:.3f}, teacher NLL {manifest['teacher_nll']}, "
        f"{manifest['elapsed_seconds']:.0f}s"
    )
    return manifest


# ---------------------------------------------------------------------------
# Fine-tuning


def dagger_loss(model, batch, outcome_loss_coeff):
    observations = {key: batch[key] for key in _OBSERVATION_SPEC}
    dist, values = _grammar_distribution(model, observations)
    logp = dist.log_prob(batch["labels"])
    if not torch.isfinite(logp).all() or (logp < -1e7).any():
        raise RuntimeError("A teacher label is unreachable under the student's action grammar")
    nll = -logp.mean()
    outcome = outcome_nll(values["outcome_probabilities"], batch["outcome_class"], batch["outcome_mask"])
    loss = nll + outcome_loss_coeff * outcome
    return loss, {"nll": float(nll.detach()), "outcome_nll": float(outcome.detach())}


def train_round(model, optimizer, pool, config, device, *, completed_updates, log_path, writer=None):
    """Run ``updates_per_round`` optimizer steps on the aggregate; return the new update count."""
    # Dropout off, gradients on. The backbone has no batch-dependent layers.
    model.eval()
    micro = config.micro_batch_size
    started = time.monotonic()
    # A retrained round restarts from the previous checkpoint, so its log restarts too.
    with Path(log_path).open("w", buffering=1) as stream:
        for update in range(config.updates_per_round):
            completed_updates += 1
            warmup = min(1.0, completed_updates / config.warmup_updates) if config.warmup_updates else 1.0
            lr = config.lr * warmup
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            arrays = pool.next_batch(config.batch_size)
            totals = Counter()
            for start in range(0, config.batch_size, micro):
                part = {key: value[start : start + micro] for key, value in arrays.items()}
                weight = len(part["label"]) / config.batch_size
                with torch.enable_grad():
                    loss, metrics = dagger_loss(model, collate(part, device), config.outcome_loss_coeff)
                    (loss * weight).backward()
                for key, value in metrics.items():
                    totals[key] += value * weight
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm, error_if_nonfinite=True)
            optimizer.step()
            if completed_updates % 10 == 0 or update + 1 == config.updates_per_round:
                row = {
                    "update": completed_updates,
                    "lr": lr,
                    "grad_norm": float(norm),
                    "elapsed_seconds": time.monotonic() - started,
                    **totals,
                }
                stream.write(json.dumps(row) + "\n")
                if writer is not None:
                    for key in ("nll", "outcome_nll", "grad_norm", "lr"):
                        writer.add_scalar(f"train/{key}", row[key], completed_updates)
    return completed_updates


# ---------------------------------------------------------------------------
# Evaluation panel


def evaluate_panel(model, config, seeds, *, mode, device, data, vocab):
    seed_torch(config.seed * 1_000_003 + 999_983)
    envs = make_envs(config, data, vocab)
    try:
        return play(model, envs, seeds, config, device=device, mode=mode)
    finally:
        envs.close()


def panel_summary(rows, baseline=None):
    summary = summarize(rows)
    summary["mean_antes_cleared"] = float(np.mean([row["antes_cleared"] for row in rows]))
    if baseline is not None:
        summary["vs_round_0"] = paired_statistics(rows, baseline)
        before = {row["seed"]: row["antes_cleared"] for row in baseline}
        differences = [row["antes_cleared"] - before[row["seed"]] for row in rows]
        summary["vs_round_0"]["antes_cleared_delta"] = float(np.mean(differences))
        summary["vs_round_0"]["antes_cleared_paired_ci95"] = _bootstrap(differences)
    return summary


# ---------------------------------------------------------------------------
# Run directory


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def read_json(path):
    return json.loads(Path(path).read_text())


def round_dir(out, index):
    return Path(out) / f"round_{index:03d}"


def _resolve_config(out, config):
    path = out / "config.json"
    if not path.exists():
        atomic_json(path, config.to_json())
        return config
    saved = read_json(path)
    requested = config.to_json()
    changed = sorted(key for key in requested if key not in RESUMABLE_FIELDS and saved.get(key) != requested[key])
    if changed:
        raise ValueError(f"Run directory {out} was created with different settings: {changed}")
    atomic_json(path, requested)
    return config


def _save_round_checkpoint(path, model, optimizer, reward, metadata):
    temporary = Path(path).with_name(Path(path).name + ".tmp")
    save_checkpoint(
        model,
        temporary,
        extra={
            "agent_config": asdict(model.config),
            **reward_checkpoint_metadata(reward),
            "observation_mode": OBSERVATION_INTERPRETATION,
            "optimizer_state_dict": optimizer.state_dict(),
            "dagger": metadata,
        },
    )
    os.replace(temporary, path)


def _stop_requested(out):
    return (Path(out) / "STOP").exists()


def _evaluate_round(out, index, checkpoint, config, panel, *, data, vocab, device, log):
    """Evaluate one round's checkpoint on every configured panel mode, skipping finished modes."""
    directory = round_dir(out, index)
    directory.mkdir(parents=True, exist_ok=True)
    model = None
    for mode in config.panel_modes:
        summary_path = directory / f"panel_{mode}_summary.json"
        if summary_path.exists():
            continue
        if model is None:
            model, _reward, _payload = load_policy(checkpoint, vocab, device)
        started = time.monotonic()
        rows = evaluate_panel(model, config, panel, mode=mode, device=device, data=data, vocab=vocab)
        with (directory / f"panel_{mode}.jsonl").open("w") as stream:
            for row in rows:
                stream.write(json.dumps(row) + "\n")
        baseline = None if index == 0 else read_jsonl(round_dir(out, 0) / f"panel_{mode}.jsonl")
        summary = panel_summary(rows, baseline)
        summary.update(round=index, mode=mode, checkpoint=str(checkpoint), elapsed_seconds=time.monotonic() - started)
        atomic_json(summary_path, summary)
        log(
            f"round {index} panel {mode}: {summary['wins']}/{summary['games']} wins "
            f"({summary['win_rate']:.3f}), mean antes cleared {summary['mean_antes_cleared']:.2f}"
        )


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def aggregate(out, through_round):
    """Every completed round's shards and the game outcomes that supply critic targets."""
    shards, outcomes = [], {}
    for index in range(1, through_round + 1):
        directory = round_dir(out, index) / "collect"
        for shard in read_json(directory / "manifest.json")["shards"]:
            shards.append((directory / shard["file"], shard["rows"]))
        for outcome in read_jsonl(directory / "outcomes.jsonl"):
            outcomes[outcome["seed"]] = outcome
    return shards, outcomes


def run(start_checkpoint, out, config, *, exclusions=(), log=print):
    """Run or resume a DAgger experiment in ``out``; return the learning curve."""
    from torch.utils.tensorboard import SummaryWriter

    from pylatro import load_game_data

    from .vocab import build_vocab

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    config = _resolve_config(out, config)
    start_checkpoint = Path(start_checkpoint).resolve()
    start_hash = sha256(start_checkpoint)
    identity_path = out / "start.json"
    if identity_path.exists():
        if read_json(identity_path)["sha256"] != start_hash:
            raise ValueError("The starting checkpoint differs from the one this run began with")
    else:
        atomic_json(identity_path, {"checkpoint": str(start_checkpoint), "sha256": start_hash})
    seeds = SeedPlan(out / "seeds.json", config, exclusions)
    data = load_game_data()
    vocab = build_vocab(data)
    device = torch.device(config.device)
    inference = torch.device(config.collection_device or config.device)
    checkpoints = {0: start_checkpoint}
    for index in range(1, config.rounds + 1):
        path = round_dir(out, index) / "checkpoint.pt"
        if not path.exists():
            break
        checkpoints[index] = path
    last = max(checkpoints)
    with SummaryWriter(str(out / "tensorboard")) as writer:
        evaluate = lambda index: _evaluate_round(  # noqa: E731
            out, index, checkpoints[index], config, seeds.panel, data=data, vocab=vocab, device=inference, log=log
        )
        for index in sorted(checkpoints):
            if index == 0 or index % config.eval_every == 0 or index == config.rounds:
                evaluate(index)
                _write_curve(out, config, writer)
            if _stop_requested(out):
                return _write_curve(out, config, writer)
        if last < config.rounds:
            model, reward, payload = load_policy(checkpoints[last], vocab, device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
            completed_updates = 0
            if last:
                optimizer.load_state_dict(payload["optimizer_state_dict"])
                completed_updates = payload["dagger"]["completed_updates"]
            for index in range(last + 1, config.rounds + 1):
                directory = round_dir(out, index)
                collect = directory / "collect"
                if not (collect / "manifest.json").exists():
                    if collect.exists():
                        _discard(collect)
                    directory.mkdir(parents=True, exist_ok=True)
                    model.to(inference)
                    manifest = collect_round(
                        model,
                        config,
                        seeds.round(index),
                        collect,
                        round_index=index,
                        device=inference,
                        data=data,
                        vocab=vocab,
                        log=log,
                    )
                    for key in ("greedy_agreement", "student_agreement", "behavior_wins"):
                        writer.add_scalar(f"collect/{key}", manifest[key], index)
                    if manifest["teacher_nll"] is not None:
                        writer.add_scalar("collect/teacher_nll", manifest["teacher_nll"], index)
                if _stop_requested(out):
                    return _write_curve(out, config, writer)
                model.to(device)
                optimizer_to(optimizer, device)
                shards, outcomes = aggregate(out, index)
                pool = ShardPool(
                    shards, outcomes, pool_shards=config.pool_shards, rng=np.random.default_rng([config.seed, index, 7])
                )
                seed_torch(config.seed * 1_000_003 + 500_000 + index)
                started = time.monotonic()
                completed_updates = train_round(
                    model,
                    optimizer,
                    pool,
                    config,
                    device,
                    completed_updates=completed_updates,
                    writer=writer,
                    log_path=directory / "train.jsonl",
                )
                checkpoints[index] = directory / "checkpoint.pt"
                _save_round_checkpoint(
                    checkpoints[index],
                    model,
                    optimizer,
                    reward,
                    {
                        "round": index,
                        "completed_updates": completed_updates,
                        "aggregate_rows": int(sum(rows for _path, rows in shards)),
                        "aggregate_shards": len(shards),
                        "start_checkpoint_sha256": start_hash,
                        "config": config.to_json(),
                        "training_seconds": time.monotonic() - started,
                    },
                )
                log(f"round {index}: trained to update {completed_updates} on {len(shards)} shards")
                if index % config.eval_every == 0 or index == config.rounds:
                    evaluate(index)
                _write_curve(out, config, writer)
                if _stop_requested(out):
                    break
        return _write_curve(out, config, writer)


def optimizer_to(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _discard(directory):
    import shutil

    # A collection without its manifest was interrupted; its seeds are replayed.
    shutil.rmtree(directory)


def _write_curve(out, config, writer=None):
    out = Path(out)
    curve = []
    for directory in sorted(out.glob("round_[0-9][0-9][0-9]")):
        index = int(directory.name.split("_")[1])
        row = {"round": index, "beta": config.beta(index) if index else None}
        manifest_path = directory / "collect" / "manifest.json"
        if manifest_path.exists():
            manifest = read_json(manifest_path)
            row["collection"] = {
                key: manifest[key]
                for key in (
                    "games",
                    "labelled",
                    "label_failures",
                    "executed_by_teacher",
                    "greedy_agreement",
                    "student_agreement",
                    "teacher_nll",
                    "behavior_wins",
                    "behavior_mean_antes_cleared",
                    "elapsed_seconds",
                )
            }
        for mode in config.panel_modes:
            summary_path = directory / f"panel_{mode}_summary.json"
            if summary_path.exists():
                summary = read_json(summary_path)
                entry = {
                    key: summary[key]
                    for key in ("games", "wins", "win_rate", "win_rate_wilson_ci95", "mean_antes_cleared", "stalls")
                }
                if "vs_round_0" in summary:
                    paired = summary["vs_round_0"]
                    entry["win_rate_delta"] = paired["sampled_or_greedy_delta"]
                    entry["win_rate_delta_ci95"] = paired["paired_ci95"]
                    entry["antes_cleared_delta"] = paired["antes_cleared_delta"]
                    entry["antes_cleared_delta_ci95"] = paired["antes_cleared_paired_ci95"]
                row[f"panel_{mode}"] = entry
                if writer is not None:
                    writer.add_scalar(f"panel_{mode}/win_rate", entry["win_rate"], index)
                    writer.add_scalar(f"panel_{mode}/mean_antes_cleared", entry["mean_antes_cleared"], index)
        curve.append(row)
    atomic_json(out / "learning_curve.json", curve)
    (out / "report.md").write_text(render_report(curve, config))
    return curve


def _percent(value):
    return "—" if value is None else f"{100 * value:.1f}%"


def _delta(value, interval, *, scale, unit):
    if value is None:
        return "—"
    low, high = interval
    return f"{scale * value:+.2f}{unit} [{scale * low:+.2f}, {scale * high:+.2f}]"


def render_report(curve, config):
    lines = [
        "# DAgger from V68",
        "",
        f"Panel: {config.panel_games} fixed seeds, {', '.join(config.panel_modes)} play. Round 0 is the starting "
        "checkpoint; deltas are paired against it (10,000-draw game-seed bootstrap).",
        "",
        "Stalled games hit the step cap; they count as non-wins here and should stay at zero. "
        "Collection columns describe the pre-update student on the states it visited that round "
        "(behavior wins include teacher-executed actions).",
        "",
    ]
    for mode in config.panel_modes:
        lines += [
            f"## Panel: {mode}",
            "",
            "| Round | Beta | Wins | Win rate | Δ vs round 0 | Mean antes cleared | Δ antes | Stalled |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in curve:
            entry = row.get(f"panel_{mode}")
            if entry is None:
                continue
            beta = "—" if row["beta"] is None else f"{row['beta']:.3f}"
            wins = _delta(entry.get("win_rate_delta"), entry.get("win_rate_delta_ci95"), scale=100, unit=" pts")
            antes = _delta(entry.get("antes_cleared_delta"), entry.get("antes_cleared_delta_ci95"), scale=1, unit="")
            lines.append(
                f"| {row['round']} | {beta} | {entry['wins']}/{entry['games']} | {_percent(entry['win_rate'])} "
                f"| {wins} | {entry['mean_antes_cleared']:.2f} | {antes} | {entry['stalls']} |"
            )
        lines.append("")
    lines += [
        "## Collection",
        "",
        "| Round | Games | Labels | Teacher-executed | Greedy agreement | Teacher NLL | Behavior wins | Minutes |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in curve:
        collection = row.get("collection")
        if collection is None:
            continue
        nll = "—" if collection["teacher_nll"] is None else f"{collection['teacher_nll']:.3f}"
        lines.append(
            f"| {row['round']} | {collection['games']} | {collection['labelled']} "
            f"| {collection['executed_by_teacher']} | {_percent(collection['greedy_agreement'])} "
            f"| {nll} | {collection['behavior_wins']}/{collection['games']} "
            f"| {collection['elapsed_seconds'] / 60:.1f} |"
        )
    return "\n".join(lines) + "\n"
