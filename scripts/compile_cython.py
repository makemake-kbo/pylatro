#!/usr/bin/env python3
"""Compile hot pylatro modules with cython

Usage:
    nix develop
    uv sync --group dev
    uv run python scripts/compile_cython.py
    uv run python scripts/compile_cython.py --clean
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "pylatro"
SRC_AGENT = ROOT / "src" / "pylatro_agent"

MODULES: list[str] = [
    "rng",
    "scoring",
    "flow",
    "instances",
    "blind",
    "consumables",
    "shop",
    "pool",
    "runtime",
    "_helpers",
]

AGENT_MODULES: list[str] = [
    "tokenizer",
    "heuristic",
    "hand_candidates",
    "training.fast_runner",
]

# -ffp-contract=off is REQUIRED for correctness, not speed. pylatro is a bit-exact
# port of Balatro's RNG (rng.pseudohash) and scoring. Once those hot loops carry C
# `double` locals (via @cython.locals), the compiler may fuse `a*b + c*d` into a
# single FMA instruction, skipping the intermediate rounding CPython's float
# arithmetic performs. pseudohash is a chaotic recurrence (it divides by its own
# running value each step), so one 1-ULP FMA difference snowballs into a totally
# different hash(i.e. a different card/shop/boss sequence for a given seed). With
# this repo's python build flags the fusion does happen and breaks
# tests/test_rng.py; `off` forces strict round-after-each-op IEEE that matches
# CPython.
EXTRA_COMPILE_ARGS = (
    "-ffp-contract=off",
    "-march=native",
    "-O3",
    # the commands below are commented out because they break
    # seed arithmetics, or very high score/naneinf assumptions
    # but are useful when you want max performance for
    # generating pretraining data and evaluating model play
    # "-ffast-math",
)

DIRECTIVES = {
    "language_level": "3",
    "boundscheck": False,
    "wraparound": False,
    "initializedcheck": False,
    "nonecheck": False,
    "overflowcheck": False,
    "cdivision": True,
    "always_allow_keywords": True,
    "legacy_implicit_noexcept": True,
}


def _build_distribution(ext_modules: list[object], include_agent: bool = False):
    from setuptools import Distribution

    packages = ["pylatro"]
    if include_agent:
        packages.append("pylatro_agent")

    return Distribution(
        {
            "name": "pylatro",
            "packages": packages,
            "package_dir": {"": "src"},
            "ext_modules": ext_modules,
        }
    )


def _legacy_src() -> Path:
    return SRC / "pylatro"


def compile_extensions() -> None:
    from Cython.Build import cythonize
    from setuptools import Extension

    extensions = [
        Extension(
            f"pylatro.{mod}",
            sources=[str(SRC / f"{mod}.py")],
            extra_compile_args=list(EXTRA_COMPILE_ARGS),
        )
        for mod in MODULES
    ]

    if AGENT_MODULES:
        import numpy as np

        extensions.extend(
            Extension(
                f"pylatro_agent.{mod}",
                sources=[str(SRC_AGENT / f"{mod.replace('.', '/')}.py")],
                include_dirs=[np.get_include()],
                extra_compile_args=list(EXTRA_COMPILE_ARGS),
            )
            for mod in AGENT_MODULES
        )

    ext_modules = cythonize(extensions, compiler_directives=DIRECTIVES, force=True, quiet=False)

    dist = _build_distribution(ext_modules, include_agent=bool(AGENT_MODULES))
    cmd = dist.get_command_obj("build_ext")
    cmd.ensure_finalized()
    cmd.inplace = True
    cmd.run()

    compiled = list(SRC.rglob("*.so")) + list(SRC_AGENT.rglob("*.so"))
    for so in compiled:
        print(f"  {so}")

    print(f"\nCompiled {len(compiled)} extensions. Run tests to verify.")


def clean_extensions() -> None:
    removed = 0
    for src_dir in (SRC, SRC_AGENT):
        for pattern in ("*.so", "*.c"):
            for p in src_dir.rglob(pattern):
                if p.name == "__init__.py":
                    continue
                p.unlink()
                removed += 1
                print(f"  rm {p}")
    legacy_src = _legacy_src()
    if legacy_src.exists():
        removed += sum(1 for _ in legacy_src.glob("*"))
        shutil.rmtree(legacy_src)
        print(f"  rm -r {legacy_src}")
    build_dir = ROOT / "build"
    if build_dir.exists():
        shutil.rmtree(build_dir)
        print(f"  rm -r {build_dir}")
    print(f"\nCleaned {removed} files.")


def main() -> None:
    if "--clean" in sys.argv:
        clean_extensions()
    else:
        compile_extensions()


if __name__ == "__main__":
    main()
