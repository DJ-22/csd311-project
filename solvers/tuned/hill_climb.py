"""Solver 6: stochastic hill climbing on solver-3's weight vector.

Start from ``DEFAULT_WEIGHTS``; perturb one random weight by ``N(0, sigma^2)`` and
accept if better; after ``STUCK_LIMIT`` consecutive non-improving evaluations,
restart from a uniformly random point in ``[-5, 5]^12``.
"""

from __future__ import annotations

import random

import numpy as np

from engine.board import Board
from solvers.search import DEFAULT_WEIGHTS
from solvers.tuned.fitness import (
    DEFAULT_BOARD,
    DEFAULT_BUDGET_GAMES,
    FitnessEvaluator,
    clip_weights,
    random_weights,
)

SIGMA = 0.5
STUCK_LIMIT = 30

Trace = list[tuple[int, float, float]]


def hill_climb(evaluator: FitnessEvaluator, budget: int, seed: int) -> tuple[np.ndarray, Trace]:
    """Maximise fitness with stochastic hill climbing; return ``(best_w, trace)``."""
    if budget < 1:
        raise ValueError("budget must be >= 1")
    rng = random.Random(seed)
    n = len(DEFAULT_WEIGHTS)
    current = np.asarray(DEFAULT_WEIGHTS, dtype=np.float32).copy()
    current_f = evaluator.evaluate(current)
    best, best_f = current.copy(), current_f
    trace: Trace = [(evaluator.evals, current_f, best_f)]
    stuck = 0
    while evaluator.evals < budget:
        if stuck >= STUCK_LIMIT:
            current = random_weights(rng, n)
            current_f = evaluator.evaluate(current)
            stuck = 0
        else:
            candidate = current.copy()
            i = rng.randrange(n)
            candidate[i] += rng.gauss(0.0, SIGMA)
            candidate = clip_weights(candidate)
            cf = evaluator.evaluate(candidate)
            if cf > current_f:
                current, current_f = candidate, cf
            else:
                stuck += 1
        if current_f > best_f:
            best, best_f = current.copy(), current_f
        trace.append((evaluator.evals, current_f, best_f))
    return best, trace


def run_hill_climb(
    budget: int,
    seed: int,
    board: Board | tuple[int, int] = DEFAULT_BOARD,
    budget_games: int = DEFAULT_BUDGET_GAMES,
) -> np.ndarray:
    """Tune weights with hill climbing and return the best vector found."""
    w, _ = hill_climb(FitnessEvaluator(board, budget_games, seed), budget, seed)
    return w
