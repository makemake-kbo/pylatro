"""Entry point for `python -m pylatro_cli`."""

from .app import BalatroApp


def main() -> None:
    app = BalatroApp()
    app.run()


if __name__ == "__main__":
    main()
