"""Solver 7: dataset split, labels, from-scratch learners, player and harness hook."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("torch")

from engine.board import get_board  # noqa: E402
from engine.game import apply_move, is_terminal, legal_moves, new_game  # noqa: E402
from engine.symmetry import canonical  # noqa: E402
from harness.cli import make_player  # noqa: E402
from solvers.exact import get_solver  # noqa: E402
from solvers.supervised.dataset import (  # noqa: E402
    TEST,
    TRAIN,
    VAL,
    build_dataset,
    input_size,
    label_positions,
    load_dataset,
    make_inputs,
    sample_positions,
)
from solvers.supervised.models import (  # noqa: E402
    MLP,
    LearnedModel,
    Model,
    Perceptron,
    SigmoidUnit,
    fit_model,
    load_model,
    one_ply_scores,
)
from solvers.supervised.player import LearnedPlayer, search_learned  # noqa: E402


@pytest.fixture(scope="module")
def data_2x2(tmp_path_factory: pytest.TempPathFactory) -> dict[str, np.ndarray]:
    path = build_dataset("2x2", "exact", 5000, seed=1, path=tmp_path_factory.mktemp("d") / "d.npz")
    return load_dataset(path)


@pytest.fixture(scope="module")
def model_path(data_2x2: dict[str, np.ndarray], tmp_path_factory: pytest.TempPathFactory) -> Path:
    lm = fit_model("mlp", data_2x2, "both", "reg", seed=0, hidden=(16,), epochs=30)
    return lm.save(tmp_path_factory.mktemp("m") / "mlp.pkl")


# dataset
def test_no_game_in_two_splits(data_2x2: dict[str, np.ndarray]) -> None:
    splits_per_game: dict[int, set[int]] = {}
    for g, s in zip(data_2x2["game"], data_2x2["split"]):
        splits_per_game.setdefault(int(g), set()).add(int(s))
    assert all(len(s) == 1 for s in splits_per_game.values())
    present = set(int(s) for s in data_2x2["split"])
    assert present == {TRAIN, VAL, TEST}


def test_positions_are_unique_up_to_symmetry(data_2x2: dict[str, np.ndarray]) -> None:
    keys = [
        (canonical(int(e), 2, 2), int(t) == int(f))
        for e, t, f in zip(data_2x2["edges"], data_2x2["to_move"], data_2x2["first_player"])
    ]
    assert len(keys) == len(set(keys))


def test_exact_labels_and_targets(data_2x2: dict[str, np.ndarray]) -> None:
    solver = get_solver(2, 2)
    board = get_board(2, 2)
    for i in range(0, len(data_2x2["edges"]), 7):
        e = int(data_2x2["edges"][i])
        assert data_2x2["V"][i] == solver.value(e)
        best = {m for m in range(board.E) if (int(data_2x2["best_mask"][i]) >> m) & 1}
        assert best == set(solver.best_moves(e))
    assert np.all(np.abs(data_2x2["y_reg"]) <= 1.0)
    assert np.array_equal(data_2x2["y_cls"], (data_2x2["V"] > 0).astype(np.int8))


def test_mc_labels_are_seeded_probabilities() -> None:
    board = get_board(2, 2)
    pos = sample_positions(board, 40, seed=3)
    a = label_positions(board, pos, "mc", seed=3, n_rollouts=20)
    b = label_positions(board, pos, "mc", seed=3, n_rollouts=20)
    assert np.array_equal(a["p"], b["p"])
    assert np.all((a["p"] >= 0) & (a["p"] <= 1))
    assert np.allclose(a["y_reg"], 2 * a["p"] - 1)


def test_input_kinds_have_documented_sizes(data_2x2: dict[str, np.ndarray]) -> None:
    board = get_board(2, 2)
    for kind in ("phi", "raw", "both"):
        X = make_inputs(data_2x2, kind)
        assert X.shape == (len(data_2x2["edges"]), input_size(board, kind))
    raw = make_inputs(data_2x2, "raw")
    assert set(np.unique(raw)) <= {0.0, 1.0}


# from-scratch learners
def _separable(seed: int, n: int = 200) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    y = (X[:, 0] + 2 * X[:, 1] > 0.1).astype(np.int8)
    return X, y


def test_perceptron_separates_linearly_separable_data() -> None:
    X, y = _separable(0)
    p = Perceptron("cls", seed=0, eta=0.1, epochs=200)
    p.fit(X, y, X, y)
    assert np.mean((p.score(X) > 0) != y) == 0.0


def test_perceptron_rejects_regression() -> None:
    with pytest.raises(ValueError):
        Perceptron("reg", seed=0)


@pytest.mark.parametrize("loss", ["mse", "xent"])
def test_sigmoid_unit_learns_separable_data(loss: str) -> None:
    X, y = _separable(1)
    s = SigmoidUnit("cls", seed=0, loss=loss, eta=1.0, epochs=300, l2=0.0)  # type: ignore[arg-type]
    s.fit(X, y, X, y)
    assert np.mean((s.score(X) > 0) != y) <= 0.03


def test_mlp_learns_xor_but_sigmoid_unit_cannot() -> None:
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]] * 25, dtype=np.float32)
    y = np.array([0, 1, 1, 0] * 25, dtype=np.int8)
    mlp = MLP("cls", seed=0, hidden=(8,), lr=0.05, l2=0.0, epochs=300, batch=16, patience=300)
    mlp.fit(X, y, X, y)
    assert np.mean((mlp.score(X) > 0) != y) == 0.0
    unit = SigmoidUnit("cls", seed=0, epochs=300, l2=0.0)
    unit.fit(X, y, X, y)
    assert np.mean((unit.score(X) > 0) != y) >= 0.25


# player
class _Const(Model):
    """Scores every position as +1 (good for the player to move)."""

    name, task, history = "const", "reg", {}

    def fit(self, X: np.ndarray, y: np.ndarray, Xv: np.ndarray, yv: np.ndarray) -> None: ...

    def score(self, X: np.ndarray) -> np.ndarray:
        return np.ones(len(X))


def test_one_ply_score_uses_extra_turn_rule() -> None:
    board = get_board(1, 2)
    lm = LearnedModel(_Const(), 1, 2, "raw", "reg", 0, np.zeros(board.E + 2), np.ones(board.E + 2))
    # Box 0 has three sides drawn (top, bottom, left): its right side completes it.
    top, bottom, left, right = board.BOX_EDGES[0]
    edges = (1 << top) | (1 << bottom) | (1 << left)
    scores = one_ply_scores(lm, edges, 0, 0, board)
    remaining_after = 1
    assert scores[right] == pytest.approx(1 + remaining_after)  # k + V̂(s'), same player continues
    other = next(e for e in scores if e != right)
    assert scores[other] == pytest.approx(-2)  # -V̂(s'), turn passes, 2 boxes left


def test_learned_player_plays_legal_games(model_path: Path) -> None:
    rng = random.Random(0)
    player = LearnedPlayer(model_path, seed=0)
    assert player.name == "learned-mlp"
    for _ in range(5):
        player.reset()
        state = new_game(2, 2)
        while not is_terminal(state):
            if state.to_move == 0:
                move = player.choose_move(state)
            else:
                move = rng.choice(legal_moves(state))
            assert move in legal_moves(state)
            state, _ = apply_move(state, move)


def test_learned_player_rejects_other_board(model_path: Path) -> None:
    with pytest.raises(ValueError):
        LearnedPlayer(model_path).choose_move(new_game(2, 3))


def test_model_round_trip(model_path: Path, data_2x2: dict[str, np.ndarray]) -> None:
    lm = load_model(model_path)
    X = make_inputs(data_2x2, "both")
    assert np.all(np.abs(lm.score_inputs(X)) <= 1.0)
    assert {"test_error", "test_pct_optimal"} <= set(lm.metrics)


def test_search_learned_and_harness_hook(model_path: Path) -> None:
    state = new_game(2, 2)
    p = search_learned(model_path, seed=0, max_depth=2)
    assert p.name == "search-learned"
    assert p.choose_move(state) in legal_moves(state)
    assert make_player(f"learned:{model_path}", 0).name == "learned-mlp"
    assert make_player(f"search-learned:{model_path}", 0).name == "search-learned"
