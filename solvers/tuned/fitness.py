"""Solver 6: shared fitness for tuning solver-3's weight vector.

Plays ``budget_games`` games (half vs Greedy, half vs ``SearchPlayer`` at depth 2)
with alternating seat, and returns::

    mean(win=1, draw=0.5, loss=0)  +  0.01 * mean_box_margin / num_boxes

Common random numbers: game ``i`` uses seed ``seed * SEED_STRIDE + i``, so every
``w`` sees the same streams. Identical weights are cached.
"""

from __future__ import annotations

import csv
import random
from datetime import datetime
from pathlib import Path

import numpy as np

from engine.board import Board, get_board
from features.features import NUM_FEATURES
from harness.tournament import play_game
from solvers.baselines import GreedyPlayer
from solvers.search import DEFAULT_WEIGHTS, SearchPlayer

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "tuned"
RESULTS_DIR = REPO_ROOT / "experiments" / "results"

DEFAULT_BOARD = (3, 3)
DEFAULT_BUDGET_GAMES = 40
WEIGHT_BOUNDS = (-5.0, 5.0)
LO, HI = WEIGHT_BOUNDS
FITNESS_DEPTH = 2
CACHE_DECIMALS = 6
SEED_STRIDE = 10_000  # spaces run seeds so different seeds never share a game


def clip_weights(w: np.ndarray | list[float]) -> np.ndarray:
    """Project ``w`` onto the search box ``[-5, 5]^12`` as a float32 array."""
    return np.clip(np.asarray(w, dtype=np.float32), LO, HI)


def random_weights(rng: random.Random, n: int = NUM_FEATURES) -> np.ndarray:
    """Uniformly random weight vector in ``[-5, 5]^n`` from a seeded RNG."""
    return np.asarray([rng.uniform(LO, HI) for _ in range(n)], dtype=np.float32)


def game_seed(seed: int, i: int) -> int:
    """Seed for game ``i`` of a run with base ``seed`` (spaced so runs never overlap)."""
    return seed * SEED_STRIDE + i


class FitnessEvaluator:
    """Cached, common-random-number fitness over a fixed board.

    ``evals`` counts every :meth:`evaluate` call (cache hits included); ``misses``
    counts actual game simulations. Optimisers read ``evals`` for their budget.
    """

    def __init__(
        self,
        board: Board | tuple[int, int] = DEFAULT_BOARD,
        budget_games: int = DEFAULT_BUDGET_GAMES,
        seed: int = 0,
    ) -> None:
        self.board = board if isinstance(board, Board) else get_board(*board)
        self.budget_games = budget_games
        self.seed = seed
        self.evals = 0
        self.misses = 0
        self.hits = 0
        self._cache: dict[tuple[float, ...], float] = {}

    def game_seeds(self) -> list[int]:
        """Game seeds this evaluator will use, for cross-run disjointness checks."""
        return [game_seed(self.seed, i) for i in range(self.budget_games)]

    def _key(self, w: np.ndarray) -> tuple[float, ...]:
        return tuple(round(float(x), CACHE_DECIMALS) for x in w)

    def evaluate(self, w: np.ndarray) -> float:
        w = np.asarray(w, dtype=np.float32)
        if w.shape != (NUM_FEATURES,):
            raise ValueError(f"expected {NUM_FEATURES} weights, got shape {w.shape}")
        self.evals += 1
        key = self._key(w)
        if key in self._cache:
            self.hits += 1
            return self._cache[key]
        value = self._compute(w)
        self._cache[key] = value
        self.misses += 1
        return value

    __call__ = evaluate

    def _opponent(self, seed: int, greedy: bool) -> SearchPlayer | GreedyPlayer:
        if greedy:
            return GreedyPlayer(seed=seed)
        return SearchPlayer(weights=DEFAULT_WEIGHTS, max_depth=FITNESS_DEPTH, seed=seed)

    def _compute(self, w: np.ndarray) -> float:
        board = self.board
        m, n, num_boxes = board.m, board.n, board.num_boxes
        greedy_split = self.budget_games // 2
        win_total = 0.0
        margin_total = 0.0
        for i in range(self.budget_games):
            gs = game_seed(self.seed, i)
            seeder = random.Random(gs)
            s0 = seeder.getrandbits(32)
            s1 = seeder.getrandbits(32)
            greedy = i < greedy_split
            tuned_first = i % 2 == 0
            if tuned_first:
                p0 = SearchPlayer(weights=w, max_depth=FITNESS_DEPTH, seed=s0)
                p1 = self._opponent(s1, greedy)
                seat = 0
            else:
                p0 = self._opponent(s0, greedy)
                p1 = SearchPlayer(weights=w, max_depth=FITNESS_DEPTH, seed=s1)
                seat = 1
            row = play_game(p0, p1, m, n, gs, time_limit=None)
            winner = int(row["winner"])
            score = (int(row["score0"]), int(row["score1"]))
            margin = score[seat] - score[1 - seat]
            win = 1.0 if winner == seat else 0.5 if winner == -1 else 0.0
            win_total += win
            margin_total += margin
        win_mean = win_total / self.budget_games
        margin_mean = margin_total / self.budget_games
        return win_mean + 0.01 * margin_mean / num_boxes


def fitness(
    w: np.ndarray,
    budget_games: int = DEFAULT_BUDGET_GAMES,
    seed: int = 0,
    board: Board | tuple[int, int] = DEFAULT_BOARD,
) -> float:
    """One-off (uncached) fitness of ``w``; see :class:`FitnessEvaluator`."""
    return FitnessEvaluator(board, budget_games, seed).evaluate(w)


def write_trace(
    optimiser: str,
    trace: list[tuple[int, float, float]],
    tag: str | None = None,
    results_dir: Path | None = None,
) -> Path:
    """Write ``trace`` rows ``(eval_count, current_f, best_f)`` to a timestamped CSV."""
    results_dir = results_dir or RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = f"E3_{optimiser}_trace" + (f"_{tag}" if tag else "")
    path = results_dir / f"{base}_{stamp}.csv"
    suffix = 1
    while path.exists():
        path = results_dir / f"{base}_{stamp}_{suffix}.csv"
        suffix += 1
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["eval_count", "current_f", "best_f"])
        for eval_count, current_f, best_f in trace:
            writer.writerow([eval_count, current_f, best_f])
    return path
