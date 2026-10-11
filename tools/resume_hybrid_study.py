"""Run the frozen study with JSON support for NumPy summary scalars.

This adapter changes report serialization only. The original driver still
verifies every frozen source/configuration hash and resumes complete games.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np


def json_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


def json_native(value):
    return json.loads(json.dumps(value, default=json_scalar))


def run(output):
    import hybrid_study as study

    original_atomic = study.atomic

    def atomic(path, value):
        original_atomic(path, json_native(value))

    study.atomic = atomic
    adapter_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    receipt = output / "reporting_adapter.json"
    if receipt.exists():
        if json.loads(receipt.read_text())["sha256"] != adapter_hash:
            raise RuntimeError("Reporting adapter changed after its first use")
    else:
        atomic(
            receipt,
            dict(
                path="tools/resume_hybrid_study.py",
                sha256=adapter_hash,
                started_unix=time.time(),
                scope="JSON scalar conversion only; frozen scientific source and gameplay unchanged",
            ),
        )
    try:
        study.run(output)
    except Exception:
        import traceback

        atomic(output / "resume_error.json", dict(traceback=traceback.format_exc(), time=time.time()))
        raise
    finally:
        study.report(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).resolve().parents[1] / "runs/public_hybrid_20261002"
    )
    run(parser.parse_args().output)
