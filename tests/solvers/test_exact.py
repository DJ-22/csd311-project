"""Exact solver correctness.

Memo is checked against a brute-force (no memo) search. A no-memo search of the
empty 2x2 board alone visits ~e*12! = 1.3e9 nodes, so 2x2 is checked by:
  * true brute force on every 2x2 state with <= 7 undrawn edges, and
  * an independent bottom-up retrograde table (by popcount, no recursion) on all
    4096 states.
1x2 (E = 7) is brute-forced on every state.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from engine.board import Board, boxes_completed, get_board
from engine.game import GameState, apply_move, is_terminal, new_game
from engine.symmetry import apply_symmetry, symmetries
from solvers.baselines import GreedyPlayer
from solvers.exact import ExactPlayer, ExactSolver


def brute_force(edges: int, board: Board) -> int:
    """Plain negamax over the full game tree: no memo, no pruning."""
    if edges == board.full:
        return 0
    best = -(board.num_boxes + 1)
    for e in range(board.E):
        if (edges >> e) & 1:
            continue
        k = boxes_completed(edges, e, board)
        child = edges | (1 << e)
        val = k + brute_force(child, board) if k > 0 else -brute_force(child, board)
        best = max(best, val)
    return best


def retrograde(board: Board) -> list[int]:
    """Values of all 2^E states, filled from the full board down to the empty one."""
    E = board.E
    V = [0] * (1 << E)
    by_count: list[list[int]] = [[] for _ in range(E + 1)]
    for s in range(1 << E):
        by_count[s.bit_count()].append(s)
    for count in range(E - 1, -1, -1):
        for s in by_count[count]:
            best = -(board.num_boxes + 1)
            for e in range(E):
                if not (s >> e) & 1:
                    k = boxes_completed(s, e, board)
                    c = V[s | (1 << e)]
                    best = max(best, k + c if k > 0 else -c)
            V[s] = best
    return V


def test_1x1_values() -> None:
    solver = ExactSolver(1, 1)
    assert solver.value(0) == -1    # second player draws the 4th side
    assert solver.value(get_board(1, 1).full) == 0


@pytest.mark.parametrize("m, n", [(1, 1), (1, 2), (2, 2), (2, 3)])
@pytest.mark.parametrize("use_symmetry, alphabeta", [(False, False), (True, False), (True, True)])
def test_full_board_is_zero(m: int, n: int, use_symmetry: bool, alphabeta: bool) -> None:
    solver = ExactSolver(m, n, use_symmetry, alphabeta)
    assert solver.value(get_board(m, n).full) == 0


def test_memo_matches_brute_force_on_every_1x2_state() -> None:
    board = get_board(1, 2)
    solver = ExactSolver(1, 2, use_symmetry=False)
    for edges in range(1 << board.E):
        assert solver.value(edges) == brute_force(edges, board), edges


def test_memo_matches_brute_force_on_deep_2x2_states() -> None:
    board = get_board(2, 2)
    solver = ExactSolver(2, 2, use_symmetry=False)
    checked = 0
    for edges in range(1 << board.E):
        if board.E - edges.bit_count() <= 7:
            assert solver.value(edges) == brute_force(edges, board), edges
            checked += 1
    assert checked == sum(1 for e in range(1 << 12) if e.bit_count() >= 5)


@pytest.mark.parametrize("use_symmetry", [False, True])
def test_memo_matches_retrograde_on_every_2x2_state(use_symmetry: bool) -> None:
    board = get_board(2, 2)
    table = retrograde(board)
    solver = ExactSolver(2, 2, use_symmetry=use_symmetry)
    for edges in range(1 << board.E):
        assert solver.value(edges) == table[edges], edges


def test_symmetric_positions_have_equal_values() -> None:
    board = get_board(2, 2)
    solver = ExactSolver(2, 2, use_symmetry=False)  # so equality is not by construction
    rng = random.Random(2)
    perms = symmetries(2, 2)
    for _ in range(500):
        e = rng.getrandbits(board.E)
        v = solver.value(e)
        for p in perms:
            assert solver.value(apply_symmetry(e, p)) == v


@pytest.mark.parametrize("use_symmetry", [True, False])
def test_alphabeta_matches_plain_memo_on_every_2x2_state(use_symmetry: bool) -> None:
    board = get_board(2, 2)
    plain = ExactSolver(2, 2, use_symmetry=use_symmetry, alphabeta=False)
    ab = ExactSolver(2, 2, use_symmetry=use_symmetry, alphabeta=True)
    for edges in range(1 << board.E):
        assert ab.value(edges) == plain.value(edges), edges
    # Re-query in a different order so TT bound entries are exercised from warm state.
    for edges in reversed(range(1 << board.E)):
        assert ab.value(edges) == plain.value(edges), edges


def test_alphabeta_matches_plain_on_2x3_sample() -> None:
    board = get_board(2, 3)
    plain = ExactSolver(2, 3, use_symmetry=True)
    ab = ExactSolver(2, 3, use_symmetry=True, alphabeta=True)
    rng = random.Random(9)
    for _ in range(300):
        e = rng.getrandbits(board.E)
        assert ab.value(e) == plain.value(e)


def test_alphabeta_expands_fewer_nodes_on_2x3() -> None:
    plain = ExactSolver(2, 3, use_symmetry=True)
    ab = ExactSolver(2, 3, use_symmetry=True, alphabeta=True)
    assert plain.value(0) == ab.value(0)
    assert ab.nodes < plain.nodes


def test_best_moves_are_optimal() -> None:
    board = get_board(2, 2)
    solver = ExactSolver(2, 2)
    rng = random.Random(4)
    for _ in range(200):
        e = rng.getrandbits(board.E)
        if e == board.full:
            continue
        best = solver.best_moves(e)
        assert best
        v = solver.value(e)
        for mv in best:
            k = boxes_completed(e, mv, board)
            child = solver.value(e | (1 << mv))
            assert (k + child if k > 0 else -child) == v


def test_value_predicts_final_margin_under_optimal_play() -> None:
    """Two exact players: the final margin equals V(empty) for the first player."""
    for seed in range(5):
        s = new_game(2, 2)
        players = [ExactPlayer(seed), ExactPlayer(seed + 100)]
        while not is_terminal(s):
            s, _ = apply_move(s, players[s.to_move].choose_move(s))
        assert s.scores[0] - s.scores[1] == ExactSolver(2, 2).value(0)


def test_exact_never_loses_to_greedy_on_2x2() -> None:
    v0 = ExactSolver(2, 2).value(0)
    for seed in range(20):
        exact_first = seed % 2 == 0
        players = [ExactPlayer(seed), GreedyPlayer(seed)] if exact_first else [GreedyPlayer(seed), ExactPlayer(seed)]
        s = new_game(2, 2)
        while not is_terminal(s):
            s, _ = apply_move(s, players[s.to_move].choose_move(s))
        margin = s.scores[0] - s.scores[1]
        exact_margin = margin if exact_first else -margin
        assert exact_margin >= (v0 if exact_first else -v0)


def test_exact_player_reports_nodes() -> None:
    p = ExactPlayer(seed=0)
    s = GameState(get_board(2, 2))
    p.choose_move(s)
    assert p.nodes >= 1


def test_save_table_and_dataset(tmp_path) -> None:
    solver = ExactSolver(1, 2)
    path = solver.save_table(tmp_path / "exact_1x2.npy")
    table = np.load(path)
    assert table.shape == (1 << 7,) and table.dtype == np.int8
    assert table[0] == solver.value(0)
    ds = solver.export_dataset(tmp_path / "exact_1x2_dataset.npz", [(0, 0), (5, 1)])
    data = np.load(ds)
    assert list(data["edges"]) == [0, 5]
    assert list(data["first_player"]) == [0, 1]
    assert list(data["V"]) == [solver.value(0), solver.value(5)]
