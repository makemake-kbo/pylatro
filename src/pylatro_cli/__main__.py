"""Entry point for `python -m pylatro_cli`."""

import sys


def main() -> None:
    argv = sys.argv[1:]
    if argv and argv[0] in {"seed-search", "seedsearch"}:
        from .seedsearch import main as seedsearch_main

        raise SystemExit(seedsearch_main(argv[1:]))

    from .app import BalatroApp

    app = BalatroApp()
    app.run()


if __name__ == "__main__":
    main()
