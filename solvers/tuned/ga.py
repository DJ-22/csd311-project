"""Solver 6: real-coded genetic algorithm on solver-3's weight vector.

Population of 20 uniform ``[-5, 5]^12`` individuals; tournament selection
(``q = 3``), BLX-0.5 crossover (rate 0.9), Gaussian mutation (sigma 0.3 per gene,
rate 1/12), elitism of 2, generational replacement.

The budget is a number of fitness evaluations: 20 for the initial population plus
18 per generation, so a budget of ``B`` runs ``(B - 20) // 18`` generations.
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

POP_SIZE = 20
ELITE = 2
TOURNAMENT_Q = 3
CROSSOVER_RATE = 0.9
BLX_ALPHA = 0.5
MUTATION_RATE = 1.0 / 12
MUTATION_SIGMA = 0.3

Trace = list[tuple[int, float, float]]


def _tournament(fits: list[float], rng: random.Random) -> int:
    idx = rng.sample(range(len(fits)), TOURNAMENT_Q)
    return max(idx, key=lambda i: fits[i])


def _blx_crossover(p1: np.ndarray, p2: np.ndarray, rng: random.Random) -> np.ndarray:
    child = np.empty_like(p1)
    for i in range(len(p1)):
        lo, hi = min(p1[i], p2[i]), max(p1[i], p2[i])
        d = hi - lo
        child[i] = rng.uniform(lo - BLX_ALPHA * d, hi + BLX_ALPHA * d)
    return child


def _mutate(child: np.ndarray, rng: random.Random) -> np.ndarray:
    for i in range(len(child)):
        if rng.random() < MUTATION_RATE:
            child[i] += rng.gauss(0.0, MUTATION_SIGMA)
    return child


def genetic_algorithm(evaluator: FitnessEvaluator, budget: int, seed: int) -> tuple[np.ndarray, Trace]:
    if budget < POP_SIZE:
        raise ValueError(f"GA needs budget >= pop size {POP_SIZE}, got {budget}")
    rng = random.Random(seed)
    n = len(DEFAULT_WEIGHTS)
    pop = [random_weights(rng, n) for _ in range(POP_SIZE)]
    fits: list[float] = []
    best: np.ndarray | None = None
    best_f = -math.inf
    trace: Trace = []
    for w in pop:
        f = evaluator.evaluate(w)
        fits.append(f)
        if f > best_f:
            best_f, best = f, w.copy()
        trace.append((evaluator.evals, f, best_f))
    offspring = POP_SIZE - ELITE
    while evaluator.evals + offspring <= budget:
        order = sorted(range(POP_SIZE), key=lambda i: fits[i], reverse=True)
        new_pop = [pop[i].copy() for i in order[:ELITE]]
        new_fits = [fits[i] for i in order[:ELITE]]
        for _ in range(offspring):
            p1 = pop[_tournament(fits, rng)]
            p2 = pop[_tournament(fits, rng)]
            child = _blx_crossover(p1, p2, rng) if rng.random() < CROSSOVER_RATE else p1.copy()
            child = clip_weights(_mutate(child, rng))
            f = evaluator.evaluate(child)
            new_pop.append(child)
            new_fits.append(f)
            if f > best_f:
                best_f, best = f, child.copy()
            trace.append((evaluator.evals, f, best_f))
        pop, fits = new_pop, new_fits
    assert best is not None
    return best, trace


def run_ga(budget: int, seed: int, board=DEFAULT_BOARD, budget_games: int = DEFAULT_BUDGET_GAMES) -> np.ndarray:
    """Tune weights with the GA and return the best vector found."""
    w, _ = genetic_algorithm(FitnessEvaluator(board, budget_games, seed), budget, seed)
    return w
