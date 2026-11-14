"""E6 MCTS scaling: solver 5 win rate against a fixed opponent vs iteration budget.

For every variant in ``--variants`` and every budget in ``--iterations``, an
``MCTSPlayer`` with exactly that many iterations per move (UCT iterations, or
rollouts for flat MC) plays ``--games`` games against ``--opponent`` with the
shared harness, seats alternating. Every (variant, budget) uses the same game
seeds (common random numbers), so differences come from the player, not the draw.

Variants (the spec's three comparisons):

* ``uct``         UCT with Greedy rollouts (the tuned default, eps = 0.1)
* ``uct-random``  UCT with uniformly random rollouts
* ``flat``        flat Monte Carlo with Greedy rollouts

The budget is iterations per move, as the spec states for E6 ("win rate vs
iterations"); the mean seconds per move is recorded too, so budgets can also be
read as wall-clock cost.

Output: the harness CSV ``results/e6_<board>_<timestamp>.csv`` (one row per game;
the MCTS seat is named ``<variant>@<iterations>``), a printed table with Wilson
95% CIs, and ``results/e6_<board>_<timestamp>.png`` (win rate vs iterations).

Example::

    python -m experiments.e6_mcts_scaling --board 3x3 --games 100 --seed 0
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from engine.board import parse_board
from experiments._results import RESULTS_DIR, stamped_path
from harness.cli import make_player
from harness.metrics import PlayerSummary, summarise
from harness.tournament import match, write_csv
from solvers.base import Player
from solvers.mcts import MCTSPlayer

VARIANTS: dict[str, dict[str, Any]] = {
    "uct": {"rollout": "greedy", "flat": False},
    "uct-random": {"rollout": "random", "flat": False},
    "flat": {"rollout": "greedy", "flat": True},
}

# Validated categorical slots 1-3 (blue, orange, aqua) on a light surface, plus chart ink.
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a")
SERIES_MARKERS = ("o", "s", "^")
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"


def label(variant: str, iterations: int) -> str:
    """Seat name of an MCTS configuration in the CSV."""
    return f"{variant}@{iterations}"


def make_mcts(variant: str, iterations: int, seed: int) -> Player:
    """An ``MCTSPlayer`` for ``variant`` with a fixed iteration budget, named by :func:`label`."""
    p = MCTSPlayer(iterations=iterations, seed=seed, **VARIANTS[variant])
    p.name = label(variant, iterations)
    return p


def run(board: str, variants: Sequence[str], budgets: Sequence[int], opponent: str, games: int,
        seed: int, time_limit: float | None, progress: bool = False) -> list[dict[str, Any]]:
    """Harness rows for every (variant, budget) match against ``opponent``."""
    for v in variants:
        if v not in VARIANTS:
            raise ValueError(f"unknown variant {v!r}; choose from {sorted(VARIANTS)}")
    bd = parse_board(board)
    rows: list[dict[str, Any]] = []
    for v in variants:
        for it in budgets:
            me = label(v, it)

            def factory(name: str, s: int, v: str = v, it: int = it, me: str = me) -> Player:
                return make_mcts(v, it, s) if name == me else make_player(name, s)

            rows.extend(match(me, opponent, factory, bd.m, bd.n, games, seed, time_limit))
            if progress:
                s = summarise(rows, me)
                print(f"  {me:<18} win rate {s.win_rate:.3f} [{s.ci_low:.3f}, {s.ci_high:.3f}]  "
                      f"{1000 * s.mean_time_per_move:.0f} ms/move", flush=True)
    return rows


def plot(summaries: dict[str, list[tuple[int, PlayerSummary]]], opponent: str, board: str,
         games: int, seed: int, path: Path) -> None:
    """Win rate (95% CI) vs iterations per move, one line per variant, log x axis."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.axhline(0.5, color=MUTED, lw=1, ls="--", zorder=1)
    for i, (variant, pts) in enumerate(summaries.items()):
        xs = [it for it, _ in pts]
        ys = [s.win_rate for _, s in pts]
        # max(0, ...): the Wilson bounds can sit a rounding error inside the point at 0 or 1
        lo = [max(0.0, s.win_rate - s.ci_low) for _, s in pts]
        hi = [max(0.0, s.ci_high - s.win_rate) for _, s in pts]
        color = SERIES_COLORS[i % len(SERIES_COLORS)]
        ax.errorbar(xs, ys, yerr=[lo, hi], color=color, lw=2, marker=SERIES_MARKERS[i % 3], ms=6,
                    mec=SURFACE, mew=1.5, capsize=0, elinewidth=1, label=variant, zorder=3)
        ax.annotate(variant, (xs[-1], ys[-1]), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK)
    ax.set_xscale("log")
    budgets = sorted({it for pts in summaries.values() for it, _ in pts})
    ax.set_xticks(budgets, [f"{b:,}" for b in budgets])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.margins(x=0.12)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("iterations per move (log scale)", color=INK)
    ax.set_ylabel(f"win rate vs {opponent} (draw = 0.5)", color=INK)
    ax.set_title(f"E6 MCTS scaling - {board}, {games} games per point, seed {seed}, 95% CI",
                 fontsize=10, color=INK, loc="left", pad=28)
    ax.grid(True, which="major", color=GRID, lw=0.6)
    ax.tick_params(colors=MUTED)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=len(summaries), fontsize=9,
              handlelength=1.5, borderaxespad=0.2)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--board", default="3x3")
    ap.add_argument("--iterations", default="100,300,1000,3000,10000",
                    help="comma-separated budgets per move (the spec's range goes to 50000)")
    ap.add_argument("--variants", default=",".join(VARIANTS), help=f"from {sorted(VARIANTS)}")
    ap.add_argument("--opponent", default="greedy", help="any harness player name")
    ap.add_argument("--games", type=int, default=100, help="games per (variant, budget)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--time-limit", type=float, default=1.0, help="per-move limit for a time-bound opponent")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--results-dir", default=str(RESULTS_DIR))
    args = ap.parse_args(argv)

    bd = parse_board(args.board)
    board = f"{bd.m}x{bd.n}"
    budgets = sorted({int(x) for x in args.iterations.split(",") if x})
    variants = [v for v in args.variants.split(",") if v]
    print(f"board={board} seed={args.seed} games per point={args.games} opponent={args.opponent} "
          f"iterations={budgets} variants={variants}")
    rows = run(board, variants, budgets, args.opponent, args.games, args.seed, args.time_limit, progress=True)
    out_dir = Path(args.results_dir)
    path = write_csv(rows, f"e6_{board}", out_dir)

    summaries: dict[str, list[tuple[int, PlayerSummary]]] = {}
    print(f"\n{'variant':<12}{'iters':>7}{'games':>7}  {'win rate [95% CI]':<24}{'margin':>8}{'ms/move':>9}")
    for v in variants:
        for it in budgets:
            s = summarise(rows, label(v, it))
            summaries.setdefault(v, []).append((it, s))
            ci = f"{s.win_rate:.3f} [{s.ci_low:.3f}, {s.ci_high:.3f}]"
            print(f"{v:<12}{it:>7}{s.games:>7}  {ci:<24}{s.mean_margin:>+8.2f}{1000 * s.mean_time_per_move:>9.1f}")
    print("wrote", path)
    if not args.no_plot:
        png = stamped_path(f"e6_{board}", ".png", out_dir)
        plot(summaries, args.opponent, board, args.games, args.seed, png)
        print("wrote", png)


if __name__ == "__main__":
    main()
