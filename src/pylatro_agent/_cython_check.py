"""Warn at startup if cython-annotated modules are running in pure-python mode
or have a stale .so (source newer than compiled artifact)."""

from __future__ import annotations

import importlib
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Modules that carry @cython decorators and must be compiled for performance.
# Kept in sync with scripts/compile_cython.py AGENT_MODULES.
_EXPECTED_COMPILED = (
    "pylatro_agent.tokenizer",
    "pylatro_agent.heuristic",
    "pylatro_agent.hand_candidates",
    "pylatro_agent.training.fast_runner",
)


def check_cython_freshness(warn_only: bool = True) -> list[str]:
    """Return a list of human-readable issues; log warnings for each.

    Checks two things per module:
      1. Is the imported module a compiled .so (not a .py)?
      2. Is the source .py older than the compiled .so?

    When `warn_only` is True, never raises, just returns the list.
    """
    issues: list[str] = []
    for modname in _EXPECTED_COMPILED:
        try:
            mod = importlib.import_module(modname)
        except ImportError as exc:
            issues.append(f"{modname}: import failed, {exc}")
            continue

        mod_file = getattr(mod, "__file__", None)
        if mod_file is None:
            continue
        if not mod_file.endswith(".so"):
            issues.append(
                f"{modname} is running as pure Python ({mod_file}); "
                "run `uv run python scripts/compile_cython.py` for the full speedup."
            )
            continue

        so_path = Path(mod_file)
        py_path = so_path.with_suffix("").with_suffix(".py")
        if py_path.exists() and py_path.stat().st_mtime > so_path.stat().st_mtime:
            issues.append(
                f"{modname}: source {py_path.name} is newer than compiled .so; "
                "re-run `uv run python scripts/compile_cython.py` to refresh."
            )

    for issue in issues:
        logger.warning(issue)
    return issues


def _auto_check() -> None:
    if os.environ.get("PYLATRO_SKIP_CYTHON_CHECK"):
        return
    check_cython_freshness(warn_only=True)
