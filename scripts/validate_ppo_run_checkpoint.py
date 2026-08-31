#!/usr/bin/env python3
"""Validate pinned v8 source provenance and strict v8 resume checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import uuid
from pathlib import Path

import torch

# Deliberately duplicated rather than imported: this validator runs before the
# training venv is exercised, so it must not depend on the agent package. Keep
# in lockstep with pylatro_agent.constants.TOKENIZER_VERSION.
TOKENIZER_VERSION = 10
TOKENIZER_SEMANTICS = "v8_conditional_survival_critic"
V8_PPO_CONFIG = {
    "mini_batch_size": 320,
    "micro_batch_size": 160,
    "lr": 3e-6,
    "clip_epsilon": 0.1,
}
MARKER_VERSION = 2


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_checkpoint(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path} is not a checkpoint mapping")
    return payload


def validate_source(path: Path, expected_sha256: str) -> None:
    actual_sha256 = file_sha256(path)
    if actual_sha256 != expected_sha256.lower():
        raise RuntimeError(
            f"source SHA256 mismatch: expected={expected_sha256.lower()}, actual={actual_sha256}"
        )
    payload = _load_checkpoint(path)
    if payload.get("tokenizer_version") != TOKENIZER_VERSION:
        raise RuntimeError(
            f"source tokenizer_version must be {TOKENIZER_VERSION}; fresh v8 supervised training is required"
        )
    if payload.get("tokenizer_semantics") != TOKENIZER_SEMANTICS:
        raise RuntimeError(
            f"source tokenizer_semantics must be {TOKENIZER_SEMANTICS!r}; "
            "fresh v8 supervised training is required"
        )


def create_marker(path: Path, *, run_name: str, source_sha256: str, recipe_id: str) -> str:
    run_uuid = str(uuid.uuid4())
    payload = {
        "version": MARKER_VERSION,
        "run_name": run_name,
        "run_uuid": run_uuid,
        "source_sha256": source_sha256.lower(),
        "recipe_id": recipe_id,
        "tokenizer_version": TOKENIZER_VERSION,
    }
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True)
        handle.write("\n")
    return run_uuid


def validate_marker(path: Path, *, run_name: str, source_sha256: str, recipe_id: str) -> str:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    expected = {
        "version": MARKER_VERSION,
        "run_name": run_name,
        "source_sha256": source_sha256.lower(),
        "recipe_id": recipe_id,
        "tokenizer_version": TOKENIZER_VERSION,
    }
    mismatches = [
        f"{key}: expected={value!r}, actual={payload.get(key)!r}"
        for key, value in expected.items()
        if payload.get(key) != value
    ]
    try:
        run_uuid = str(uuid.UUID(str(payload.get("run_uuid", ""))))
    except ValueError as exc:
        raise RuntimeError("ownership marker has an invalid run UUID") from exc
    if run_uuid != payload.get("run_uuid"):
        mismatches.append(f"run_uuid is not canonical: {payload.get('run_uuid')!r}")
    if mismatches:
        raise RuntimeError("ownership marker mismatch: " + "; ".join(mismatches))
    return run_uuid


def validate_resume(
    path: Path,
    *,
    run_uuid: str,
    source_sha256: str,
    recipe_id: str,
) -> None:
    payload = _load_checkpoint(path)
    if payload.get("checkpoint_format") != "ppo_full":
        raise RuntimeError("resume checkpoint is not a full PPO checkpoint")
    if payload.get("tokenizer_version") != TOKENIZER_VERSION:
        raise RuntimeError(
            f"resume tokenizer_version must be {TOKENIZER_VERSION}, got {payload.get('tokenizer_version')!r}"
        )
    if payload.get("tokenizer_semantics") != TOKENIZER_SEMANTICS:
        raise RuntimeError(
            f"resume tokenizer_semantics must be {TOKENIZER_SEMANTICS!r}, "
            f"got {payload.get('tokenizer_semantics')!r}"
        )
    saved_config = payload.get("ppo_config_fields") or {}
    mismatches = [
        f"{key}: expected={expected!r}, actual={saved_config.get(key)!r}"
        for key, expected in V8_PPO_CONFIG.items()
        if saved_config.get(key) != expected
    ]
    if mismatches:
        raise RuntimeError("resume recipe mismatch: " + "; ".join(mismatches))
    if payload.get("ppo_transition_state") is not None:
        raise RuntimeError("v8 resume checkpoint unexpectedly contains legacy transition state")
    expected_provenance = {
        "run_uuid": run_uuid,
        "source_sha256": source_sha256.lower(),
        "recipe_id": recipe_id,
    }
    if payload.get("ppo_run_provenance") != expected_provenance:
        raise RuntimeError(
            "resume run provenance mismatch: "
            f"expected={expected_provenance!r}, actual={payload.get('ppo_run_provenance')!r}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)
    source_parser = subparsers.add_parser("source")
    source_parser.add_argument("path", type=Path)
    source_parser.add_argument("--expected-sha256", required=True)
    resume_parser = subparsers.add_parser("resume")
    resume_parser.add_argument("path", type=Path)
    resume_parser.add_argument("--run-uuid", required=True)
    resume_parser.add_argument("--source-sha256", required=True)
    resume_parser.add_argument("--recipe-id", required=True)
    for mode in ("marker-create", "marker-validate"):
        marker = subparsers.add_parser(mode)
        marker.add_argument("path", type=Path)
        marker.add_argument("--run-name", required=True)
        marker.add_argument("--source-sha256", required=True)
        marker.add_argument("--recipe-id", required=True)
    args = parser.parse_args()
    try:
        if args.mode == "source":
            validate_source(args.path, args.expected_sha256)
        elif args.mode == "resume":
            validate_resume(
                args.path,
                run_uuid=args.run_uuid,
                source_sha256=args.source_sha256,
                recipe_id=args.recipe_id,
            )
        elif args.mode == "marker-create":
            print(
                create_marker(
                    args.path,
                    run_name=args.run_name,
                    source_sha256=args.source_sha256,
                    recipe_id=args.recipe_id,
                )
            )
        else:
            print(
                validate_marker(
                    args.path,
                    run_name=args.run_name,
                    source_sha256=args.source_sha256,
                    recipe_id=args.recipe_id,
                )
            )
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"checkpoint validation failed: {exc}\n")


if __name__ == "__main__":
    main()
