"""`pylatro seed-search` subcommand: find seeds matching a JSON constraint spec."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

from pylatro.data import load_game_data
from pylatro.seedsearch import SpecError, check_seed, parse_spec, search_seeds


def _strip_jsonc(text: str) -> str:
    """Strip ``//`` and ``#`` comments (full-line or trailing) and trailing
    commas from jsonc text, returning strict JSON.

    String literals are scanned char-by-char so ``//``, ``#``, or ``,`` inside
    a quoted value are left untouched.
    """
    out: list[str] = []
    i, n = 0, len(text)
    in_string = escaped = False
    while i < n:
        ch = text[i]
        if in_string:
            out.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            i += 1
            continue
        if ch == '"':
            in_string = True
            out.append(ch)
            i += 1
            continue
        # Line comment: // or # to end of line.
        if ch == "#" or (ch == "/" and i + 1 < n and text[i + 1] == "/"):
            while i < n and text[i] != "\n":
                i += 1
            continue
        # Trailing comma: drop it if only whitespace separates it from } or ].
        if ch == ",":
            j = i + 1
            while j < n and text[j] in " \t\r\n":
                j += 1
            if j < n and text[j] in "}]":
                i += 1
                continue
        out.append(ch)
        i += 1
    return "".join(out)


def _load_spec_file(path: Path) -> dict:
    """Load a spec file, tolerating ``//`` / ``#`` comments and trailing commas."""
    try:
        text = path.read_text()
    except OSError as error:
        raise SystemExit(f"error: cannot read spec file: {error}") from error
    try:
        return json.loads(_strip_jsonc(text))
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
        "--max-seeds", type=int, default=10_000_000, help="seeds to try before giving up (default: %(default)s)"
    )
    parser.add_argument(
        "--matches", type=int, default=1, help="stop after this many matching seeds (default: %(default)s)"
    )
    parser.add_argument("--rng-seed", type=int, help="seed for the seed generator itself, for reproducible searches")
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 1,
        help="worker processes to check seeds in parallel (default: all CPUs, %(default)s)",
    )
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
        workers=max(args.workers, 1),
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
