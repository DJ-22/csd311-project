"""E3: tune solver-3 weights with four metaheuristics and re-evaluate them.

Runs each optimiser over ``--seeds`` seeds, writes a per-run trace CSV
(``experiments/results/E3_<opt>_trace_*.csv``) and the best weight vector
(``data/tuned/<opt>_best_w.json``), then re-evaluates the final best ``w`` of each
optimiser against the untuned ``SearchPlayer`` at equal depth on ``--games`` fresh
games and prints the win rate with its 95% CI.

Example::

    python -m experiments.e3_tune --board 3x3 --budget 600 --seeds 3 --games 400
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import numpy as np

from engine.board import parse_board
from harness.metrics import summarise
from harness.tournament import match, write_csv
from solvers.search import DEFAULT_WEIGHTS, SearchPlayer
from solvers.tuned import OPTIMISERS, PLAYER_NAMES, save_best_w, write_trace
from solvers.tuned.annealing import simulated_annealing
from solvers.tuned.fitness import FitnessEvaluator
from solvers.tuned.ga import genetic_algorithm
from solvers.tuned.hill_climb import hill_climb
from solvers.tuned.tabu import tabu_search

CORES = {
    "ga": genetic_algorithm,
    "hc": hill_climb,
    "sa": simulated_annealing,
    "tabu": tabu_search,
}
# Clear of every fitness game seed (fitness uses `seed * SEED_STRIDE + i`, stride 10_000).
REEVAL_SEED_OFFSET = 1_000_000


def _tune(args: argparse.Namespace) -> dict[str, np.ndarray]:
    best_per_opt: dict[str, np.ndarray] = {}
    for opt in OPTIMISERS:
        core = CORES[opt]
        candidates: list[tuple[float, int, np.ndarray]] = []
        for k in range(args.seeds):
            s = args.seed + k
            evaluator = FitnessEvaluator(args.board, args.budget_games, s)
            w, trace = core(evaluator, args.budget, s)
            final_best_f = trace[-1][2]
            write_trace(opt, trace, tag=f"seed{s}")
            candidates.append((final_best_f, s, w))
            print(f"{PLAYER_NAMES[opt]} seed={s}: best_f={final_best_f:.4f}  evals={evaluator.evals}")
        best_f, best_seed, best_w = max(candidates, key=lambda t: t[0])
        save_best_w(opt, best_w, board=args.board_spec, seed=best_seed, budget=args.budget, fitness=best_f)
        best_per_opt[opt] = best_w
        print(f"  -> {PLAYER_NAMES[opt]} best seed={best_seed} fitness={best_f:.4f}")
    return best_per_opt


def _reevaluate(args: argparse.Namespace, best_per_opt: dict[str, np.ndarray]) -> None:
    board = args.board
    rows = []
    for opt in OPTIMISERS:
        name = PLAYER_NAMES[opt]
        w = best_per_opt[opt]

        def factory(nm: str, pseed: int) -> SearchPlayer:
            tuned = nm == name
            p = SearchPlayer(weights=w if tuned else DEFAULT_WEIGHTS, max_depth=args.depth, seed=pseed)
            if tuned:
                p.name = name
            return p

        r = match(name, "search", factory, board.m, board.n, args.games, args.seed + REEVAL_SEED_OFFSET)
        rows.extend(r)
        s = summarise(r, name)
        print(
            f"{name:<10} vs search@{args.depth}: games={s.games}  "
            f"win_rate={s.win_rate:.3f} [{s.ci_low:.3f}, {s.ci_high:.3f}]  "
            f"margin={s.mean_margin:+.2f}"
        )
    write_csv(rows, "E3_reeval")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m experiments.e3_tune", description=__doc__)
    parser.add_argument("--board", default="3x3", help="board in boxes (default 3x3)")
    parser.add_argument("--budget", type=int, default=600, help="fitness evaluations per optimiser per seed")
    parser.add_argument("--budget-games", type=int, default=40, help="games per fitness evaluation")
    parser.add_argument("--seeds", type=int, default=3, help="optimiser runs per method")
    parser.add_argument("--seed", type=int, default=0, help="base seed")
    parser.add_argument("--games", type=int, default=400, help="re-evaluation games per matchup")
    parser.add_argument("--depth", type=int, default=2, help="search depth for the re-evaluation")
    args = parser.parse_args(argv)
    args.board = parse_board(args.board)
    args.board_spec = f"{args.board.m}x{args.board.n}"

    best_per_opt = _tune(args)
    print()
    _reevaluate(args, best_per_opt)


if __name__ == "__main__":
    main()
