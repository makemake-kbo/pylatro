"""Entry point for `python -m pylatro_cli`."""

import sys


def main() -> None:
    argv = sys.argv[1:]
    if argv and argv[0] in {"seed-search", "seedsearch"}:
        from .seedsearch import main as seedsearch_main

        raise SystemExit(seedsearch_main(argv[1:]))

    # `--model` (or the `watch` subcommand) hands off to the self-contained
    # model-viewer in pylatro_agent, which needs torch. Kept out of the
    # Textual CLI so the interactive UI has no ML dependencies.
    if argv and (argv[0] == "watch" or "--model" in argv or "-m" in argv):
        from pylatro_agent.watch import main as watch_main

        raise SystemExit(watch_main(argv[1:] if argv[0] == "watch" else argv))

    from .app import BalatroApp

    app = BalatroApp()
    app.run()


if __name__ == "__main__":
    main()
