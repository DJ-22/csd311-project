"""Solver 6: simulated annealing on solver-3's weight vector.

The initial temperature is the mean absolute fitness of 4 random solutions; one
random weight is perturbed by ``N(0, sigma^2)``; a worse move is accepted with
probability ``exp(delta / T)`` where ``delta < 0``; the temperature is cooled
geometrically (``T <- 0.95 T``) every 10 evaluations.
"""

from __future__ import annotations

import math
import random

import numpy as np

from solvers.search import DEFAULT_WEIGHTS
from solvers.tuned.fitness import (
    DEFAULT_BOARD,
    DEFAULT_BUDGET_GAMES,
    FitnessEvaluator,
    clip_weights,
    random_weights,
)

SIGMA = 0.5
COOLING = 0.95
COOL_INTERVAL = 10
T0_SAMPLES = 4
MIN_T = 1e-6

Trace = list[tuple[int, float, float]]


def simulated_annealing(evaluator: FitnessEvaluator, budget: int, seed: int) -> tuple[np.ndarray, Trace]:
    rng = random.Random(seed)
    n = len(DEFAULT_WEIGHTS)
    best: np.ndarray | None = None
    best_f = -math.inf
    trace: Trace = []

    # Initial temperature: mean |fitness| of 4 random solutions (counted in the budget).
    t_vals: list[float] = []
    for _ in range(T0_SAMPLES):
        if evaluator.evals >= budget:
            break
        w = random_weights(rng, n)
        f = evaluator.evaluate(w)
        t_vals.append(abs(f))
        if f > best_f:
            best_f, best = f, w.copy()
        trace.append((evaluator.evals, f, best_f))
    T = max(sum(t_vals) / len(t_vals) if t_vals else 1.0, MIN_T)

    current = np.asarray(DEFAULT_WEIGHTS, dtype=np.float32).copy()
    if evaluator.evals < budget:
        current_f = evaluator.evaluate(current)
        if current_f > best_f:
            best_f, best = current_f, current.copy()
        trace.append((evaluator.evals, current_f, best_f))
    else:
        current_f = best_f if best is not None else 0.0

    steps = 0
    while evaluator.evals < budget:
        candidate = current.copy()
        i = rng.randrange(n)
        candidate[i] += rng.gauss(0.0, SIGMA)
        candidate = clip_weights(candidate)
        cf = evaluator.evaluate(candidate)
        delta = cf - current_f
        if delta > 0 or rng.random() < math.exp(delta / T):
            current, current_f = candidate, cf
        if current_f > best_f:
            best_f, best = current_f, current.copy()
        steps += 1
        if steps % COOL_INTERVAL == 0:
            T = max(T * COOLING, MIN_T)
        trace.append((evaluator.evals, current_f, best_f))

    if best is None:
        best = np.asarray(DEFAULT_WEIGHTS, dtype=np.float32).copy()
    return best, trace


def run_annealing(budget: int, seed: int, board=DEFAULT_BOARD, budget_games: int = DEFAULT_BUDGET_GAMES) -> np.ndarray:
    """Tune weights with simulated annealing and return the best vector found."""
    w, _ = simulated_annealing(FitnessEvaluator(board, budget_games, seed), budget, seed)
    return w
