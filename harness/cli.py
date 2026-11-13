"""Command-line entry point.

Example::

    python -m harness.cli tournament --board 2x2 --games 100 --players random,greedy,exact
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence

from engine.board import parse_board
from harness.metrics import summarise
from harness.tournament import DEFAULT_TIME_LIMIT, round_robin, write_csv
from solvers.base import Player
from solvers.baselines import GreedyPlayer, RandomPlayer
from solvers.exact import ExactPlayer
from solvers.mcts import MCTSPlayer
from solvers.rules.rules_player import RulePlayer
from solvers.search import SearchPlayer
from solvers.tuned import tuned_player

PLAYERS: dict[str, Callable[[int], Player]] = {
    "random": lambda seed: RandomPlayer(seed),
    "greedy": lambda seed: GreedyPlayer(seed),
    "exact": lambda seed: ExactPlayer(seed),
    "search": lambda seed: SearchPlayer(seed=seed),
    "rules": lambda seed: RulePlayer(backend="python", seed=seed),
    "rules-prolog": lambda seed: RulePlayer(backend="prolog", seed=seed),
    "mcts": lambda seed: MCTSPlayer(seed=seed),
    "mcts-flat": lambda seed: MCTSPlayer(flat=True, seed=seed),
    "search-ga": lambda seed: tuned_player("search-ga", seed=seed),
    "search-hc": lambda seed: tuned_player("search-hc", seed=seed),
    "search-sa": lambda seed: tuned_player("search-sa", seed=seed),
    "search-tabu": lambda seed: tuned_player("search-tabu", seed=seed),
}


# Solver 7 players take a trained model: ``learned:<model_path>`` or ``search-learned:<model_path>``.
MODEL_PLAYERS: tuple[str, ...] = ("learned", "search-learned")


def make_player(name: str, seed: int) -> Player:
    """Build a registered player by name, or a solver 7 player from ``<kind>:<model_path>``."""
    kind, sep, path = name.partition(":")
    if sep and kind in MODEL_PLAYERS:
        # Imported here so that other players do not pay for importing torch / sklearn.
        from solvers.supervised.player import LearnedPlayer, search_learned

        return LearnedPlayer(path, seed=seed) if kind == "learned" else search_learned(path, seed=seed)
    try:
        return PLAYERS[name](seed)
    except KeyError:
        raise ValueError(
            f"unknown player {name!r}; choose from {sorted(PLAYERS)} or learned:<model_path>, "
            "search-learned:<model_path>"
        ) from None


def _tournament(args: argparse.Namespace) -> None:
    board = parse_board(args.board)
    names = [p.strip() for p in args.players.split(",") if p.strip()]
    labels: dict[str, str] = {}  # CLI name -> player name written to the CSV
    for name in names:
        try:
            labels[name] = make_player(name, 0).name
        except (ValueError, OSError) as exc:
            raise SystemExit(str(exc)) from None
    if len(names) < 2:
        raise SystemExit("need at least two players")
    rows = round_robin(names, make_player, board.m, board.n, args.games, args.seed, args.time_limit)
    path = write_csv(rows, args.name or f"tournament_{board.m}x{board.n}")

    print(
        f"board={board.m}x{board.n}  seed={args.seed}  games per pair={args.games}  "
        f"total games={len(rows)}  time limit={args.time_limit}s"
    )
    header = f"{'player':<10}{'games':>7}{'W':>6}{'D':>5}{'L':>6}  {'win rate [95% CI]':<24}{'margin':>8}{'ms/move':>10}{'nodes/move':>12}{'b*':>7}"
    print(header)
    print("-" * len(header))
    for name in names:
        s = summarise(rows, labels[name])
        ci = f"{s.win_rate:.3f} [{s.ci_low:.3f}, {s.ci_high:.3f}]"
        print(
            f"{s.player:<10}{s.games:>7}{s.wins:>6}{s.draws:>5}{s.losses:>6}  {ci:<24}"
            f"{s.mean_margin:>+8.2f}{1000 * s.mean_time_per_move:>10.3f}{s.mean_nodes_per_move:>12.1f}{s.mean_bstar:>7.2f}"
        )
    print(f"\nwrote {path}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m harness.cli", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("tournament", help="round-robin tournament; writes a CSV to experiments/results/")
    t.add_argument("--board", required=True, help="board size in boxes, e.g. 3x3")
    t.add_argument("--games", type=int, required=True, help="games per pair of players")
    t.add_argument(
        "--players",
        required=True,
        help=f"comma-separated, from {sorted(PLAYERS)}, or learned:<model_path> / search-learned:<model_path>",
    )
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--time-limit", type=float, default=DEFAULT_TIME_LIMIT, help="seconds per move")
    t.add_argument("--name", default=None, help="CSV file name prefix")
    t.set_defaults(func=_tournament)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
