"""Solver 3 (heuristic search) correctness and interface.

* At unlimited depth the root value must equal the exact solver's value for every
  flag combination: the key test for the extra-turn alpha-beta windows.
* At depths 1-4 alpha-beta must agree with an independent plain negamax (no
  pruning) that uses the same leaf: quiescence over captures, then the evaluator.
"""

from __future__ import annotations

import itertools
import random
import time
from collections.abc import Callable
from functools import lru_cache

import numpy as np
import pytest

from engine.board import Board, boxes_completed, get_board
from engine.game import GameState, apply_move, is_terminal, legal_moves, new_game
from features.features import NUM_FEATURES, phi_edges
from harness.cli import make_player
from solvers.exact import get_solver
from solvers.search import DEFAULT_WEIGHTS, SearchPlayer

FLAGS = list(itertools.product([True, False], repeat=3))  # use_tt, use_ordering, use_quiescence
FLAG_IDS = [f"tt{t:d}-ord{o:d}-q{q:d}" for t, o, q in FLAGS]


def random_position(board: Board, rng: random.Random, min_free: int, max_free: int) -> int:
    """Uniformly random edge set with between ``min_free`` and ``max_free`` undrawn edges."""
    free = rng.randint(min_free, max_free)
    return sum(1 << e for e in rng.sample(range(board.E), board.E - free))


def make_searcher(use_tt: bool, use_ordering: bool, use_quiescence: bool) -> SearchPlayer:
    return SearchPlayer(use_tt=use_tt, use_ordering=use_ordering, use_quiescence=use_quiescence)


# Exact agreement at unlimited depth


@pytest.mark.parametrize("m, n, max_free", [(2, 2, 12), (2, 3, 11)])
@pytest.mark.parametrize("use_tt, use_ordering, use_quiescence", FLAGS, ids=FLAG_IDS)
def test_full_depth_matches_exact_solver(
    m: int, n: int, max_free: int, use_tt: bool, use_ordering: bool, use_quiescence: bool
) -> None:
    board = get_board(m, n)
    solver = get_solver(m, n)
    rng = random.Random(1000 * m + n)
    # Without a TT, a full-depth search from a near-empty board is exponential;
    # those configurations get positions with fewer undrawn edges.
    hi = max_free if use_tt else min(max_free, 9)
    for _ in range(200):
        edges = random_position(board, rng, 1, hi)
        player = make_searcher(use_tt, use_ordering, use_quiescence)
        free = board.E - edges.bit_count()
        mif = rng.random() < 0.5
        move, val = player.search_root(edges, free, mif, board)
        assert val == solver.value(edges), f"edges={edges:#x}"
        assert not (edges >> move) & 1
        # The chosen move is optimal.
        assert move in solver.best_moves(edges)


def test_full_depth_from_empty_2x2() -> None:
    board = get_board(2, 2)
    _, val = SearchPlayer().search_root(0, board.E, True, board)
    assert val == get_solver(2, 2).value(0)


# Alpha-beta == plain negamax at limited depth


def reference_value(edges: int, depth: int, mif: bool, board: Board, quiescence: bool) -> float:
    """Plain depth-limited negamax: no pruning, no ordering. Memoised (values unchanged)."""
    return _reference(board, quiescence)(edges, depth, mif)


@lru_cache(maxsize=None)
def _reference(board: Board, quiescence: bool) -> Callable[[int, int, bool], float]:
    """One memoised reference search per (board, quiescence), shared across tests."""
    w = np.asarray(DEFAULT_WEIGHTS, dtype=np.float32)

    @lru_cache(maxsize=None)
    def leaf(e: int, mover_first: bool) -> float:
        """Quiescence by brute force: max over every capture move, recursively."""
        if e == board.full:
            return 0.0
        if quiescence:
            caps = [(x, k) for x in range(board.E) if not (e >> x) & 1 and (k := boxes_completed(e, x, board))]
            if caps:
                return max(k + leaf(e | (1 << x), mover_first) for x, k in caps)
        return float(w @ phi_edges(e, board, mover_first))

    @lru_cache(maxsize=None)
    def negamax(e: int, d: int, mover_first: bool) -> float:
        if e == board.full:
            return 0.0
        if d == 0:
            return leaf(e, mover_first)
        best = -float("inf")
        for x in range(board.E):
            if (e >> x) & 1:
                continue
            k = boxes_completed(e, x, board)
            child = e | (1 << x)
            if k:
                val = k + negamax(child, d - 1, mover_first)
            else:
                val = -negamax(child, d - 1, not mover_first)
            best = max(best, val)
        return best

    return negamax


@pytest.mark.parametrize("use_tt, use_ordering, use_quiescence", FLAGS, ids=FLAG_IDS)
def test_alphabeta_matches_plain_negamax_3x3(use_tt: bool, use_ordering: bool, use_quiescence: bool) -> None:
    board = get_board(3, 3)
    rng = random.Random(33)
    positions = [(random_position(board, rng, 6, 14), rng.random() < 0.5) for _ in range(100)]
    for edges, mif in positions:
        player = make_searcher(use_tt, use_ordering, use_quiescence)
        for depth in range(1, 5):
            _, val = player.search_root(edges, depth, mif, board)
            ref = reference_value(edges, depth, mif, board, use_quiescence)
            assert val == pytest.approx(ref, abs=1e-5), f"edges={edges:#x} depth={depth}"


# Iterative deepening, time limit, per-move counters


def test_time_limit_respected_on_empty_3x3() -> None:
    player = SearchPlayer(seed=1)
    state = new_game(3, 3)
    for _ in range(20):
        t0 = time.perf_counter()
        move = player.choose_move(state, time_limit=0.5)
        elapsed = time.perf_counter() - t0
        assert elapsed <= 0.55, f"took {elapsed:.3f}s"
        assert move in legal_moves(state)
        assert player.depth_reached >= 1


def test_counters_positive_and_reset_every_move() -> None:
    player = SearchPlayer(max_depth=3, seed=2)
    rng = random.Random(2)
    state = new_game(3, 3)
    while not is_terminal(state):
        if state.to_move == 0:
            move = player.choose_move(state, time_limit=None)
            assert player.nodes > 0
            assert 1 <= player.depth_reached <= 3
        else:
            move = rng.choice(legal_moves(state))
        state, _ = apply_move(state, move)
    # Counters describe the latest move only: a fixed-depth search from a fresh
    # TT costs the same number of nodes every time.
    fresh = new_game(3, 3)
    counts = []
    for _ in range(3):
        player.reset()
        player.choose_move(fresh, time_limit=None)
        counts.append((player.nodes, player.depth_reached))
    assert counts[0] == counts[1] == counts[2]
    assert counts[0][1] == 3


def test_max_depth_caps_iterative_deepening() -> None:
    player = SearchPlayer(max_depth=2)
    player.choose_move(new_game(3, 3), time_limit=5.0)
    assert player.depth_reached == 2


def test_full_depth_play_is_optimal_on_2x2() -> None:
    """Unlimited depth with no time limit searches to the end: every move is optimal."""
    solver = get_solver(2, 2)
    rng = random.Random(4)
    board = get_board(2, 2)
    for _ in range(30):
        edges = random_position(board, rng, 1, 10)
        state = GameState(board, edges, (0, 0), rng.randrange(2))
        player = SearchPlayer(max_depth=board.E)
        assert player.choose_move(state, time_limit=None) in solver.best_moves(edges)


# Evaluator


def test_default_evaluator_is_weights_dot_phi() -> None:
    board = get_board(3, 3)
    rng = random.Random(5)
    player = SearchPlayer()
    w = np.asarray(DEFAULT_WEIGHTS, dtype=np.float32)
    assert len(DEFAULT_WEIGHTS) == NUM_FEATURES
    for _ in range(50):
        edges = rng.getrandbits(board.E)
        for mif in (True, False):
            assert player.evaluator(edges, board, mif) == pytest.approx(float(w @ phi_edges(edges, board, mif)))


def test_custom_weights_are_used() -> None:
    board = get_board(3, 3)
    weights = [0.0] * NUM_FEATURES
    weights[11] = 1.0  # fraction of edges drawn
    player = SearchPlayer(weights=weights)
    assert player.evaluator(0b111, board, True) == pytest.approx(3 / board.E)
    with pytest.raises(ValueError):
        SearchPlayer(weights=[1.0] * (NUM_FEATURES - 1))
    with pytest.raises(ValueError):
        SearchPlayer(weights=weights, evaluator=lambda e, b, f: 0.0)


def test_pluggable_evaluator_gets_mover_from_turn() -> None:
    """``mover_is_first`` flips exactly when the turn passes (k == 0)."""
    board = get_board(3, 3)
    rng = random.Random(6)
    for _ in range(30):
        root = random_position(board, rng, 5, 15)
        root_mif = rng.random() < 0.5
        calls: dict[int, bool] = {}

        def evaluator(edges: int, b: Board, mover_is_first: bool) -> float:
            assert b is board
            calls[edges] = mover_is_first
            return 0.0

        player = SearchPlayer(evaluator=evaluator, use_quiescence=False, use_tt=False)
        player.search_root(root, 1, root_mif, board)
        for e in range(board.E):
            if (root >> e) & 1:
                continue
            child = root | (1 << e)
            if child == board.full:
                continue
            expected = root_mif if boxes_completed(root, e, board) else not root_mif
            assert calls[child] == expected


def test_search_player_registered_in_cli() -> None:
    player = make_player("search", 3)
    assert isinstance(player, SearchPlayer)
    assert player.name == "search"
