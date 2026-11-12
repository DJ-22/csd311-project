"""E5 Deductive vs inductive: solver 4 (rules) vs solver 7 (learned), plus Greedy.

Round-robin with the shared harness between the learned one-ply player, the
``search-learned`` variant (optional), the rule-based player and Greedy. The
learned-vs-Greedy match is also the solver 7 acceptance check ("the best
LearnedPlayer beats Greedy on 3x3").

Output: the harness CSV ``results/e5_<board>_<timestamp>.csv`` (one row per
game, seeds included) and a printed summary with Wilson 95% CIs.

Example::

    python -m experiments.e5_deductive_vs_inductive --board 3x3 \
        --model data/models/e4_mlp-64x64_both_reg_3x3_s0.pkl --games 200 --seed 0
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from engine.board import parse_board
from harness.cli import make_player
from harness.metrics import summarise
from harness.tournament import round_robin, write_csv


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--board", required=True)
    ap.add_argument("--model", required=True, help="model .pkl for the learned player (same board)")
    ap.add_argument("--games", type=int, default=200, help="games per pair")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--opponents", default="rules,greedy")
    ap.add_argument("--search-learned", action="store_true", help="also enter search-learned")
    ap.add_argument("--time-limit", type=float, default=1.0)
    args = ap.parse_args(argv)

    board = parse_board(args.board)
    names = [f"learned:{args.model}"]
    if args.search_learned:
        names.append(f"search-learned:{args.model}")
    names += [s for s in args.opponents.split(",") if s]
    labels = {nm: make_player(nm, 0).name for nm in names}
    rows = round_robin(names, make_player, board.m, board.n, args.games, args.seed, args.time_limit)
    path = write_csv(rows, f"e5_{board.m}x{board.n}")

    print(f"board={board.m}x{board.n} seed={args.seed} games per pair={args.games} total={len(rows)} "
          f"time limit={args.time_limit}s model={args.model}")
    for nm in names:
        s = summarise(rows, labels[nm])
        print(f"{s.player:<16} {s.games:>5} games  W{s.wins:>4} D{s.draws:>3} L{s.losses:>4}  "
              f"win rate {s.win_rate:.3f} [{s.ci_low:.3f}, {s.ci_high:.3f}]  margin {s.mean_margin:+.2f}")
    print("\nhead-to-head (win rate of the first-named player, draw = 0.5):")
    for a in names:
        for b in names:
            if a >= b:
                continue
            la, lb = labels[a], labels[b]
            games = [r for r in rows if {r["p0"], r["p1"]} == {la, lb}]
            pts = sum(1.0 if (r["winner"] == 0 and r["p0"] == la) or (r["winner"] == 1 and r["p1"] == la)
                      else 0.5 if r["winner"] == -1 else 0.0 for r in games)
            print(f"  {la} vs {lb}: {pts / len(games):.3f} over {len(games)} games")
    print("wrote", path)


if __name__ == "__main__":
    main()
