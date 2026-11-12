"""Solver 6: metaheuristic-tuned evaluation players.

Four optimisers tune solver-3's weight vector ``w`` (``h(s) = w . phi(s)``)
against the shared fitness function. Each produces a ``SearchPlayer`` with the
learned weights, registered as ``search-ga``, ``search-hc``, ``search-sa`` and
``search-tabu``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from features.features import NUM_FEATURES
from solvers.search import SearchPlayer
from solvers.tuned.annealing import run_annealing, simulated_annealing
from solvers.tuned.fitness import (
    DATA_DIR,
    DEFAULT_BOARD,
    DEFAULT_BUDGET_GAMES,
    WEIGHT_BOUNDS,
    FitnessEvaluator,
    fitness,
    write_trace,
)
from solvers.tuned.ga import genetic_algorithm, run_ga
from solvers.tuned.hill_climb import hill_climb, run_hill_climb
from solvers.tuned.tabu import run_tabu, tabu_search

__all__ = [
    "FitnessEvaluator",
    "fitness",
    "write_trace",
    "genetic_algorithm",
    "hill_climb",
    "simulated_annealing",
    "tabu_search",
    "run_ga",
    "run_hill_climb",
    "run_annealing",
    "run_tabu",
    "save_best_w",
    "load_best_w",
    "tuned_player",
    "OPTIMISERS",
    "PLAYER_NAMES",
    "DATA_DIR",
    "WEIGHT_BOUNDS",
    "DEFAULT_BOARD",
    "DEFAULT_BUDGET_GAMES",
]

OPTIMISERS: tuple[str, ...] = ("ga", "hc", "sa", "tabu")
PLAYER_NAMES: dict[str, str] = {
    "ga": "search-ga",
    "hc": "search-hc",
    "sa": "search-sa",
    "tabu": "search-tabu",
}


def save_best_w(
    optimiser: str,
    w: np.ndarray | list[float],
    data_dir: Path | str | None = None,
    **meta: object,
) -> Path:
    """Write ``w`` to ``<data_dir>/<optimiser>_best_w.json`` and return the path."""
    data_dir = Path(data_dir) if data_dir is not None else DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)
    payload = {"optimiser": optimiser, "weights": [float(x) for x in np.asarray(w, dtype=np.float32)], **meta}
    path = data_dir / f"{optimiser}_best_w.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def load_best_w(optimiser: str, data_dir: Path | str | None = None) -> np.ndarray:
    """Load the tuned weight vector for ``optimiser`` from disk."""
    data_dir = Path(data_dir) if data_dir is not None else DATA_DIR
    path = data_dir / f"{optimiser}_best_w.json"
    if not path.exists():
        raise FileNotFoundError(
            f"no tuned weights at {path}; run the E3 tuning first (python -m experiments.e3_tune)"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    w = np.asarray(payload["weights"], dtype=np.float32)
    if w.shape != (NUM_FEATURES,):
        raise ValueError(f"expected {NUM_FEATURES} weights in {path}, got shape {w.shape}")
    return w


def tuned_player(
    name: str,
    max_depth: int | None = None,
    seed: int = 0,
    data_dir: Path | str | None = None,
) -> SearchPlayer:
    """Build a ``SearchPlayer`` for ``name`` (``search-ga`` etc., or bare ``ga`` etc.)."""
    opt = name[len("search-"):] if name.startswith("search-") else name
    if opt not in OPTIMISERS:
        raise ValueError(f"unknown tuned player {name!r}; choose from {[PLAYER_NAMES[o] for o in OPTIMISERS]}")
    player = SearchPlayer(weights=load_best_w(opt, data_dir), max_depth=max_depth, seed=seed)
    player.name = PLAYER_NAMES[opt]
    return player
