"""Solver 6: tabu search on solver-3's weight vector.

Neighbourhood of ``+/- delta`` on each of the 12 weights (up to 24 neighbours);
a random sample of 6 is evaluated each step. A move ``(index, sign)`` stays tabu
for 7 steps; a tabu move is allowed (aspiration) when it beats the best-so-far.
"""

from __future__ import annotations

import math
import random
from collections import deque

import numpy as np

from solvers.search import DEFAULT_WEIGHTS
from solvers.tuned.fitness import (
    DEFAULT_BOARD,
    DEFAULT_BUDGET_GAMES,
    FitnessEvaluator,
    clip_weights,
)

DELTA = 0.5
SAMPLE_SIZE = 6
TENURE = 7

Trace = list[tuple[int, float, float]]


def _neighbours(current: np.ndarray) -> list[tuple[np.ndarray, tuple[int, int]]]:
    out: list[tuple[np.ndarray, tuple[int, int]]] = []
    for i in range(len(current)):
        for sign in (+1, -1):
            cand = current.copy()
            cand[i] += sign * DELTA
            cand = clip_weights(cand)
            if not np.array_equal(cand, current):  # skip the no-op at a boundary
                out.append((cand, (i, sign)))
    return out


def tabu_search(evaluator: FitnessEvaluator, budget: int, seed: int) -> tuple[np.ndarray, Trace]:
    if budget < 1:
        raise ValueError("budget must be >= 1")
    rng = random.Random(seed)
    current = np.asarray(DEFAULT_WEIGHTS, dtype=np.float32).copy()
    current_f = evaluator.evaluate(current)
    best, best_f = current.copy(), current_f
    tabu: deque[tuple[int, int]] = deque(maxlen=TENURE)
    trace: Trace = [(evaluator.evals, current_f, best_f)]
    while evaluator.evals < budget:
        remaining = budget - evaluator.evals
        neighbours = _neighbours(current)
        k = min(SAMPLE_SIZE, remaining, len(neighbours))
        sample = rng.sample(neighbours, k)
        cands = []
        for w_cand, move in sample:
            f = evaluator.evaluate(w_cand)
            cands.append((w_cand, move, f))
        admissible = [c for c in cands if c[1] not in tabu or c[2] > best_f]
        if admissible:
            w_next, move, f_next = max(admissible, key=lambda c: c[2])
        else:  # every sampled move is tabu and none aspirates: take the least-bad
            w_next, move, f_next = max(cands, key=lambda c: c[2])
        current, current_f = w_next, f_next
        if current_f > best_f:
            best, best_f = current.copy(), current_f
        tabu.append(move)
        trace.append((evaluator.evals, current_f, best_f))
    return best, trace


def run_tabu(budget: int, seed: int, board=DEFAULT_BOARD, budget_games: int = DEFAULT_BUDGET_GAMES) -> np.ndarray:
    """Tune weights with tabu search and return the best vector found."""
    w, _ = tabu_search(FitnessEvaluator(board, budget_games, seed), budget, seed)
    return w
