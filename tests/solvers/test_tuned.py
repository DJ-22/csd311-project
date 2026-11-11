"""Solver 6 (metaheuristic-tuned players): fitness, optimisers, persistence.

The acceptance criteria from the spec:

* the fitness of ``DEFAULT_WEIGHTS`` is reproducible for a fixed seed;
* every optimiser's best-so-far trace is monotone non-decreasing and stays within
  the evaluation budget;
* tuned players load their weights and register in the CLI.
"""

from __future__ import annotations

import numpy as np
import pytest

from features.features import NUM_FEATURES
from solvers.search import DEFAULT_WEIGHTS, SearchPlayer
from solvers.tuned import OPTIMISERS, PLAYER_NAMES, load_best_w, save_best_w, tuned_player
from solvers.tuned.annealing import run_annealing, simulated_annealing
from solvers.tuned.fitness import WEIGHT_BOUNDS, FitnessEvaluator, fitness
from solvers.tuned.ga import genetic_algorithm, run_ga
from solvers.tuned.hill_climb import hill_climb, run_hill_climb
from solvers.tuned.tabu import run_tabu, tabu_search

CORES = {"ga": genetic_algorithm, "hc": hill_climb, "sa": simulated_annealing, "tabu": tabu_search}
RUNNERS = {"ga": run_ga, "hc": run_hill_climb, "sa": run_annealing, "tabu": run_tabu}


def test_fitness_reproducible_for_same_seed() -> None:
    w = np.asarray(DEFAULT_WEIGHTS, dtype=np.float32)
    a = fitness(w, budget_games=8, seed=7, board=(2, 2))
    b = fitness(w, budget_games=8, seed=7, board=(2, 2))
    assert a == pytest.approx(b, abs=1e-9)
    assert -0.011 <= a <= 1.011


def test_fitness_evaluator_caches_identical_weights() -> None:
    ev = FitnessEvaluator((2, 2), budget_games=8, seed=3)
    w = np.asarray(DEFAULT_WEIGHTS, dtype=np.float32)
    f1 = ev.evaluate(w)
    f2 = ev.evaluate(w.copy())
    assert f1 == f2
    assert ev.misses == 1
    assert ev.hits == 1
    assert ev.evals == 2


def test_fitness_evaluator_rejects_wrong_shape() -> None:
    ev = FitnessEvaluator((2, 2), budget_games=4, seed=0)
    with pytest.raises(ValueError):
        ev.evaluate(np.zeros(NUM_FEATURES - 1, dtype=np.float32))


@pytest.mark.parametrize("opt", OPTIMISERS)
def test_optimiser_returns_valid_weights_and_monotone_best(opt: str) -> None:
    budget = 40  # GA needs >= pop size (20)
    evaluator = FitnessEvaluator((2, 2), budget_games=4, seed=11)
    w, trace = CORES[opt](evaluator, budget, 11)
    assert w.shape == (NUM_FEATURES,)
    assert float(w.min()) >= WEIGHT_BOUNDS[0] - 1e-6
    assert float(w.max()) <= WEIGHT_BOUNDS[1] + 1e-6
    assert len(trace) >= 1
    best_f = -float("inf")
    prev_eval = 0
    for eval_count, current_f, bf in trace:
        assert eval_count > prev_eval
        prev_eval = eval_count
        assert bf >= best_f  # best-so-far is monotone non-decreasing
        best_f = bf
    assert evaluator.evals <= budget


@pytest.mark.parametrize("opt", OPTIMISERS)
def test_run_returns_valid_weights(opt: str) -> None:
    w = RUNNERS[opt](20, 5, board=(2, 2), budget_games=4)
    assert w.shape == (NUM_FEATURES,)
    assert float(w.min()) >= WEIGHT_BOUNDS[0] - 1e-6
    assert float(w.max()) <= WEIGHT_BOUNDS[1] + 1e-6


def test_save_load_and_tuned_player(tmp_path) -> None:
    w = [0.0] * NUM_FEATURES
    w[3] = 1.0
    path = save_best_w("hc", w, data_dir=tmp_path, board="3x3")
    assert path.exists()
    loaded = load_best_w("hc", data_dir=tmp_path)
    assert np.allclose(loaded, np.asarray(w, dtype=np.float32))
    player = tuned_player("search-hc", data_dir=tmp_path)
    assert isinstance(player, SearchPlayer)
    assert player.name == "search-hc"
    assert np.allclose(player.weights, w)
    assert tuned_player("hc", data_dir=tmp_path).name == "search-hc"


def test_tuned_player_missing_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        tuned_player("search-ga", data_dir=tmp_path)


def test_tuned_player_unknown_name() -> None:
    with pytest.raises(ValueError):
        tuned_player("search-bogus")


def test_tuned_players_registered_in_cli(monkeypatch) -> None:
    from harness import cli

    def fake_tuned(name: str, seed: int = 0):
        return SearchPlayer(seed=seed)

    monkeypatch.setattr(cli, "tuned_player", fake_tuned)
    for opt in OPTIMISERS:
        name = PLAYER_NAMES[opt]
        assert name in cli.PLAYERS
        player = cli.make_player(name, 3)
        assert isinstance(player, SearchPlayer)
