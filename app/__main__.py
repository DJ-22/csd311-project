"""``python -m app [--seed N]``: open the Dots-and-Boxes app."""

from __future__ import annotations

import argparse
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m app", description="Play Dots and Boxes against the solvers.")
    ap.add_argument("--seed", type=int, default=0, help="base seed for the bots (each new game adds 1)")
    args = ap.parse_args(argv)
    try:
        from app.ui import App
    except ModuleNotFoundError as exc:
        if exc.name == "pygame":
            raise SystemExit("The app needs pygame-ce: pip install pygame-ce") from None
        raise
    App(seed=args.seed).run()


if __name__ == "__main__":
    main()
