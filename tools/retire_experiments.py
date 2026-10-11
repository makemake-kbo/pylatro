"""Inventory, preserve, verify, then explicitly unlink obsolete eval artifacts.

The archive contains evidence bytes, including unopened reserved outcomes. It
does not parse those outcomes. The original checkpoint directories are outside
the deletion scope. Run `prepare`, inspect inventory.jsonl, then run `apply`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "archives/experiments_20261002"
MODELS = {
    "ppo_replica0": (
        "eval/long_horizon_generalization_20260921_logical_kl/terminal_gae_seed0/checkpoints/ppo_update142.pt"
    ),
    "ppo_replica1": (
        "eval/long_horizon_generalization_20260921_logical_kl/terminal_gae_seed1/checkpoints/ppo_update142.pt"
    ),
    "supervised": "eval/pretraining_transfer_20260923/stage2/arms/unfrozen_encoder/checkpoint_4000.pt",
}
DATASETS = {
    "original_5000": ("cache/datasets/5k-balatro-games", 5000),
    "current_1000": ("eval/learning_bottleneck_20260913/data/training", 1000),
    "current_4000": ("eval/rule_aligned_scale_20260927/new_training_games_v1", 4000),
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def classify(path):
    rel = path.relative_to(ROOT).as_posix()
    for name, source in MODELS.items():
        if rel == source:
            return "keep", f"incumbents/{name}/checkpoint.pt"
    for name, (source, _) in DATASETS.items():
        if rel.startswith(source + "/"):
            return "keep", "datasets/" + name + rel[len(source) :]
    # Preserve all textual evidence and source, even if some redundant trace
    # files compress less well. This deliberately favors retention.
    if path.suffix in {
        ".py",
        ".md",
        ".json",
        ".jsonl",
        ".csv",
        ".toml",
        ".yaml",
        ".yml",
        ".txt",
        ".sh",
        ".patch",
        ".diff",
        ".log",
    }:
        return "archive", "archives/experiments_20261002/evidence.tar.gz::" + rel
    # Unopened teacher outcomes are retained as opaque bytes.
    if "pretraining_confirmation_20260923/stage3/teacher/" in rel:
        return "archive", "archives/experiments_20261002/evidence.tar.gz::" + rel
    # Unknown artifacts remain in place; only recognized expendable formats
    # and generated runtime files are eligible for deletion.
    if (
        path.suffix in {".pt", ".pth", ".npy", ".npz", ".parquet", ".so", ".pyc", ".c", ".o", ".a", ".lock"}
        or "events.out.tfevents" in path.name
    ):
        return "delete", None
    return "keep", rel


def prepare():
    if (ARCHIVE / "inventory.jsonl").exists():
        raise RuntimeError("Inventory already exists; refusing to overwrite preservation evidence")
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    # Save current edits before any implementation or retirement changes.
    (ARCHIVE / "working_tree.patch").write_bytes(subprocess.check_output(["git", "diff", "--binary"], cwd=ROOT))
    (ARCHIVE / "working_tree_status.txt").write_bytes(subprocess.check_output(["git", "status", "--short"], cwd=ROOT))
    with tarfile.open(ARCHIVE / "working_tree_source.tar.gz", "w:gz") as tar:
        for folder in ("src", "tests", "tools", "scripts", "docs"):
            for p in sorted((ROOT / folder).rglob("*")):
                if p.is_file() and p.suffix not in {".pyc", ".so", ".c"}:
                    tar.add(p, arcname=p.relative_to(ROOT), recursive=False)
        for p in ROOT.glob("*"):
            if p.is_file() and p.suffix in {".py", ".toml", ".md", ".lock"}:
                tar.add(p, arcname=p.name, recursive=False)
    rows = []
    for p in sorted((ROOT / "eval").rglob("*")):
        if not p.is_file() or p.is_symlink():
            continue
        action, destination = classify(p)
        stat = p.stat()
        rows.append(
            dict(
                path=p.relative_to(ROOT).as_posix(),
                bytes=stat.st_size,
                allocated_bytes=stat.st_blocks * 512,
                mtime_ns=stat.st_mtime_ns,
                action=action,
                destination=destination,
            )
        )
    with (ARCHIVE / "inventory.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    totals = Counter()
    for r in rows:
        totals[r["action"]] += r["bytes"]
    write_json(
        ARCHIVE / "before.json",
        dict(
            logical_bytes=sum(r["bytes"] for r in rows),
            allocated_bytes=sum(r["allocated_bytes"] for r in rows),
            files=len(rows),
            by_action=dict(totals),
        ),
    )
    print(json.dumps(dict(totals)), flush=True)


def preserve(rows):
    import torch

    from pylatro import load_game_data
    from pylatro_agent.agent import AgentConfig, BalatroAgent
    from pylatro_agent.checkpoint import load_checkpoint_payload
    from pylatro_agent.vocab import build_vocab

    torch.set_num_threads(1)
    registry = {}
    for name, source in MODELS.items():
        src, dst = ROOT / source, ROOT / "incumbents" / name / "checkpoint.pt"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        sha = digest(src)
        assert digest(dst) == sha
        payload = load_checkpoint_payload(dst, torch.device("cpu"))
        config = payload.get("agent_config")
        if config is None:
            raise RuntimeError(f"Missing model configuration: {source}")
        model = BalatroAgent(AgentConfig(**config), build_vocab(load_game_data()))
        model.load_state_dict(payload["state_dict"], strict=True)
        write_json(dst.parent / "agent_config.json", config)
        registry[name] = dict(original=source, path=str(dst.relative_to(ROOT)), sha256=sha, strict_load=True)
    write_json(ROOT / "incumbents/registry.json", registry)
    datasets = {}
    for name, (source, games) in DATASETS.items():
        src, dst = ROOT / source, ROOT / "datasets" / name
        # Original corpus already has a stable home; register it in place.
        if name == "original_5000":
            dst = src
        files = []
        for p in sorted(src.rglob("*")):
            if not p.is_file():
                continue
            target = dst / p.relative_to(src)
            if target != p:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, target)
            sha = digest(p)
            assert digest(target) == sha
            files.append(dict(original=str(p.relative_to(ROOT)), path=str(target.relative_to(ROOT)), sha256=sha))
        actual = (
            sum(1 for _ in (src / "metadata/games.jsonl").open())
            if name == "original_5000"
            else len(list(dst.glob("game_*.parquet")))
        )
        assert actual == games, (name, actual, games)
        datasets[name] = dict(path=str(dst.relative_to(ROOT)), games=actual, files=files)
    write_json(ROOT / "datasets/registry.json", datasets)
    print("Models strictly loaded; dataset counts and all copied hashes verified", flush=True)
    members = [r for r in rows if r["action"] == "archive"]
    with tarfile.open(ARCHIVE / "evidence.tar.gz", "w:gz", compresslevel=6) as tar:
        for row in members:
            p = ROOT / row["path"]
            row["sha256"] = digest(p)
            tar.add(p, arcname=row["path"], recursive=False)
    expected = {r["path"]: r["sha256"] for r in members}
    for name in ("evidence.tar.gz", "working_tree_source.tar.gz"):
        with tarfile.open(ARCHIVE / name, "r|gz") as tar:
            for member in tar:
                if member.isfile():
                    h = hashlib.sha256()
                    with tar.extractfile(member) as f:
                        for block in iter(lambda: f.read(8 << 20), b""):
                            h.update(block)
                    if name == "evidence.tar.gz":
                        assert h.hexdigest() == expected.pop(member.name)
    assert not expected
    write_json(
        ARCHIVE / "preservation_verified.json",
        dict(
            inventory_sha256=digest(ARCHIVE / "inventory.jsonl"),
            archives={name: digest(ARCHIVE / name) for name in ("evidence.tar.gz", "working_tree_source.tar.gz")},
            archived_files=len(members),
            reserved_outcomes="opaque bytes; not parsed or examined",
        ),
    )


def apply():
    rows = [json.loads(line) for line in (ARCHIVE / "inventory.jsonl").open()]
    if not (ARCHIVE / "preservation_verified.json").exists():
        preserve(rows)
    verified = json.loads((ARCHIVE / "preservation_verified.json").read_text())
    assert digest(ARCHIVE / "inventory.jsonl") == verified["inventory_sha256"]
    for name, sha in verified["archives"].items():
        assert digest(ARCHIVE / name) == sha
    with (ARCHIVE / "removed.jsonl").open("a") as receipt:
        for row in rows:
            if row["destination"] == row["path"]:
                continue
            p = ROOT / row["path"]
            assert p.resolve().is_relative_to((ROOT / "eval").resolve()) and not p.is_symlink()
            if not p.exists():
                continue
            st = p.stat()
            if (st.st_size, st.st_mtime_ns) != (row["bytes"], row["mtime_ns"]):
                raise RuntimeError(f"Artifact changed after inventory: {p}")
            p.unlink()
            receipt.write(json.dumps(row, sort_keys=True) + "\n")
            receipt.flush()
    remaining = [p for p in (ROOT / "eval").rglob("*") if p.is_file()]
    write_json(
        ARCHIVE / "receipt.json",
        dict(
            before=json.loads((ARCHIVE / "before.json").read_text()),
            eval_after_logical_bytes=sum(p.stat().st_size for p in remaining),
            eval_after_allocated_bytes=sum(p.stat().st_blocks * 512 for p in remaining),
            retained_bytes=sum(
                p.stat().st_size
                for folder in ("archives", "incumbents", "datasets")
                for p in (ROOT / folder).rglob("*")
                if p.is_file()
            ),
            replay_limits=(
                "Source/configuration/evidence retained; removed checkpoints and regenerated arrays "
                "prevent full replay of retired experiments."
            ),
            original_checkpoints_untouched=True,
        ),
    )
    print("Retirement complete; receipt.json records measured storage", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "apply"))
    args = parser.parse_args()
    prepare() if args.command == "prepare" else apply()
