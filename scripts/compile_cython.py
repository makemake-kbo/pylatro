#!/usr/bin/env python3
"""Compile hot pylatro modules with Cython for ~5-20x speedup.

Usage:
    nix-shell -p gcc --run "python scripts/compile_cython.py"
    python scripts/compile_cython.py --clean

Compiled .so files sit alongside the .py sources.  Python prefers the
compiled extension on import, falling back to pure-Python when absent.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "pylatro"

MODULES = [
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


def _build_distribution(ext_modules: list[object]):
    from setuptools import Distribution

    return Distribution(
        {
            "name": "pylatro",
            "packages": ["pylatro"],
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
        )
        for mod in MODULES
    ]

    ext_modules = cythonize(extensions, compiler_directives=DIRECTIVES, force=True, quiet=False)

    dist = _build_distribution(ext_modules)
    cmd = dist.get_command_obj("build_ext")
    cmd.ensure_finalized()
    cmd.inplace = True
    cmd.run()

    compiled = list(SRC.glob("*.so"))
    for so in compiled:
        print(f"  {so}")

    print(f"\nCompiled {len(compiled)} extensions. Run tests to verify.")


def clean_extensions() -> None:
    removed = 0
    for pattern in ("*.so", "*.c"):
        for p in SRC.glob(pattern):
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
