"""`pylatro seed-search` subcommand: find seeds matching a JSON constraint spec."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

from pylatro.data import load_game_data
from pylatro.seedsearch import SpecError, check_seed, parse_spec, search_seeds


def _load_spec_file(path: Path) -> dict:
    """Load a spec file, tolerating full-line # / // comments."""
    try:
        text = path.read_text()
    except OSError as error:
        raise SystemExit(f"error: cannot read spec file: {error}") from error
    text = re.sub(r"^\s*(#|//).*$", "", text, flags=re.MULTILINE)
    try:
        return json.loads(text)
    except json.JSONDecodeError as error:
        raise SystemExit(f"error: spec is not valid JSON: {error}") from error


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pylatro seed-search",
        description="Search for run seeds whose vouchers, shops, blinds, and skips match a JSON spec. "
        "See docs/seed_search.md for the spec format.",
    )
    parser.add_argument("spec", type=Path, help="path to the JSON spec file")
    parser.add_argument("--check", metavar="SEED", help="verify a specific seed instead of searching")
    parser.add_argument(
        "--max-seeds", type=int, default=100_000, help="seeds to try before giving up (default: %(default)s)"
    )
    parser.add_argument(
        "--matches", type=int, default=1, help="stop after this many matching seeds (default: %(default)s)"
    )
    parser.add_argument("--rng-seed", type=int, help="seed for the seed generator itself, for reproducible searches")
    parser.add_argument("--json", action="store_true", help="print results as JSON")
    args = parser.parse_args(argv)

    raw = _load_spec_file(args.spec)
    data = load_game_data()
    try:
        spec = parse_spec(raw, data)
    except SpecError as error:
        print(f"spec error: {error}", file=sys.stderr)
        return 2

    if args.check:
        seed = args.check.upper()
        result = check_seed(spec, seed, data)
        if result is None:
            print(f"{seed}: does not match the spec")
            return 1
        _print_matches([result], args.json)
        return 0

    started = time.monotonic()

    def progress(checked: int, found: int) -> None:
        rate = checked / max(time.monotonic() - started, 1e-9)
        print(f"\rchecked {checked:,} seeds ({rate:,.0f}/s), {found} match(es)", end="", file=sys.stderr, flush=True)

    rng = random.Random(args.rng_seed)
    found = search_seeds(
        spec,
        max_seeds=args.max_seeds,
        matches=args.matches,
        rng=rng,
        data=data,
        progress=progress if sys.stderr.isatty() else None,
    )
    if sys.stderr.isatty():
        print(file=sys.stderr)
    if not found:
        print(f"no matching seed in {args.max_seeds:,} attempts", file=sys.stderr)
        return 1
    _print_matches(found, args.json)
    return 0


def _print_matches(matches: list, as_json: bool) -> None:
    if as_json:
        print(json.dumps([{"seed": m.seed, "notes": m.notes} for m in matches], indent=2))
        return
    for m in matches:
        print(m.seed)
        for note in m.notes:
            print(f"  {note}")
