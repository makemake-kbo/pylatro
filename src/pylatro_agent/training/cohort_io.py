"""Portable complete trajectories and atomic checkpoint commits."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np

from ..constants import TOKENIZER_SEMANTICS, TOKENIZER_VERSION
from ..learning_bottleneck import atomic_json, read_json
from ..transfer import sha256
from .cohort import CohortEpisode, ObservationStore
from .cohort_config import COHORT_VERSION


def save_cohort(directory, episodes, buffer, config):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    records = []
    for episode in episodes:
        episode.observations.flush()
        obs = {
            key: np.concatenate([chunk[key] for chunk in episode.observations.chunks])
            for key in episode.observations.chunks[0]
        }
        name = f"game_{episode.seed}.npz"
        arrays = {
            key: np.asarray(getattr(episode, key))
            for key in ("actions", "rewards", "values", "log_probs", "slot_steps")
        }
        np.savez_compressed(
            directory / name,
            **{f"obs_{k}": v for k, v in obs.items()},
            **arrays,
            **{f"final_{k}": v for k, v in (episode.final_observation or {}).items()},
        )
        records.append(
            {
                "seed": episode.seed,
                "slot": episode.slot,
                "policy_version": episode.policy_version,
                "policy_sha256": episode.policy_sha256,
                "outcome": episode.outcome,
                "sampling_configuration": episode.sampling_configuration,
                "path": name,
                "sha256": sha256(directory / name),
            }
        )
    np.save(directory / "draw_indices.npy", buffer.draw_indices, allow_pickle=False)
    atomic_json(
        directory / "manifest.json",
        {
            "version": COHORT_VERSION,
            "tokenizer_version": TOKENIZER_VERSION,
            "tokenizer_semantics": TOKENIZER_SEMANTICS,
            "complete": True,
            "ppo_config": asdict(config),
            "episodes": records,
            "draw_sha256": sha256(directory / "draw_indices.npy"),
            "diagnostics": buffer.diagnostics(),
            "teacher_labels": False,
        },
    )


def load_cohort(directory):
    directory = Path(directory)
    manifest = read_json(directory / "manifest.json")
    if manifest.get("version") != COHORT_VERSION or manifest.get("complete") is not True:
        raise ValueError("Incomplete or incompatible saved cohort")
    if (
        manifest.get("tokenizer_version") != TOKENIZER_VERSION
        or manifest.get("tokenizer_semantics") != TOKENIZER_SEMANTICS
    ):
        raise ValueError("Saved cohort observation schema is incompatible")
    if sha256(directory / "draw_indices.npy") != manifest["draw_sha256"]:
        raise ValueError("Cohort draw checksum mismatch")
    episodes = []
    for row in manifest["episodes"]:
        path = directory / row["path"]
        if sha256(path) != row["sha256"]:
            raise ValueError("Cohort trajectory checksum mismatch")
        with np.load(path, allow_pickle=False) as archive:
            obs = {key.removeprefix("obs_"): archive[key] for key in archive.files if key.startswith("obs_")}
            store = ObservationStore()
            store.size = len(archive["actions"])
            store.chunks = [obs]
            store.ends = [store.size]
            episode = CohortEpisode(row["seed"], row["slot"], row["policy_version"], row["policy_sha256"], store)
            for key in ("actions", "rewards", "values", "log_probs", "slot_steps"):
                setattr(episode, key, archive[key].tolist())
            episode.final_observation = {
                key.removeprefix("final_"): archive[key] for key in archive.files if key.startswith("final_")
            }
            episode.outcome = row["outcome"]
            episode.sampling_configuration = row["sampling_configuration"]
            episodes.append(episode)
    return episodes, np.load(directory / "draw_indices.npy", allow_pickle=False), manifest
