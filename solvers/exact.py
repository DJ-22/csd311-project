"""Solver 2: exact solver for small boards.

``V(edges)`` is the net number of boxes the player to move can still gain,
excluding boxes already scored. For a move completing ``k`` boxes into ``child``::

    value = k + V(child)    if k > 0   (same player moves again)
    value = -V(child)   if k == 0  (turn passes)

``V`` depends only on ``edges``, so it is memoised on the bitmask, or on its
canonical form when ``use_symmetry`` is set. With ``alphabeta`` the search uses
negamax alpha-beta with extra-turn windows and a transposition table whose entries
carry an EXACT / LOWER / UPPER bound flag.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable

import numpy as np

from engine.board import Board, get_board
from engine.game import GameState
from engine.symmetry import canonical
from solvers.base import Player

EXACT, LOWER, UPPER = 0, 1, 2
TABLE_SENTINEL = 127


class ExactSolver:
    """Game-theoretic values of ``m x n`` positions."""

    def __init__(self, m: int, n: int, use_symmetry: bool = True, alphabeta: bool = False) -> None:
        self.m, self.n = m, n
        self.board: Board = get_board(m, n)
        self.use_symmetry = use_symmetry
        self.alphabeta = alphabeta
        self.nodes = 0  # cumulative; callers take differences
        self._tt: dict[int, int] = {}
        self._ab_tt: dict[int, tuple[int, int]] = {}
        self._inf = self.board.num_boxes + 1
        # Per-edge (boxes touched, their masks) for the hot loop.
        self._edge_masks: tuple[tuple[int, ...], ...] = tuple(
            tuple(self.board.BOX_MASK[b] for b in bs) for bs in self.board.EDGE_BOXES
        )

    # helpers
    def _key(self, edges: int) -> int:
        return canonical(edges, self.m, self.n) if self.use_symmetry else edges

    def _completed(self, edges: int, e: int) -> int:
        k = 0
        for mk in self._edge_masks[e]:
            if (edges & mk).bit_count() == 3:
                k += 1
        return k

    # plain memo
    def _memo(self, edges: int) -> int:
        self.nodes += 1
        full = self.board.full
        if edges == full:
            return 0
        key = self._key(edges)
        hit = self._tt.get(key)
        if hit is not None:
            return hit
        best = -self._inf
        free = full & ~edges
        while free:
            bit = free & -free
            free ^= bit
            e = bit.bit_length() - 1
            k = self._completed(edges, e)
            child = edges | bit
            val = k + self._memo(child) if k > 0 else -self._memo(child)
            if val > best:
                best = val
        self._tt[key] = best
        return best

    # alpha-beta
    def _ab(self, edges: int, alpha: int, beta: int) -> int:
        """Fail-soft negamax alpha-beta. Exact whenever ``alpha < V < beta``."""
        self.nodes += 1
        full = self.board.full
        if edges == full:
            return 0
        key = self._key(edges)
        entry = self._ab_tt.get(key)
        if entry is not None:
            v, flag = entry
            if flag == EXACT or (flag == LOWER and v >= beta) or (flag == UPPER and v <= alpha):
                return v
        alpha0 = alpha
        best = -self._inf
        free = full & ~edges
        while free:
            bit = free & -free
            free ^= bit
            e = bit.bit_length() - 1
            k = self._completed(edges, e)
            child = edges | bit
            if k > 0:
                val = k + self._ab(child, alpha - k, beta - k)
            else:
                val = -self._ab(child, -beta, -alpha)
            best = max(best, val)
            alpha = max(alpha, val)
            if alpha >= beta:
                break
        if best <= alpha0:
            flag = UPPER
        elif best >= beta:
            flag = LOWER
        else:
            flag = EXACT
        self._ab_tt[key] = (best, flag)
        return best

    # public API
    def value(self, edges: int) -> int:
        """Net boxes the player to move can still gain from ``edges`` with perfect play."""
        if self.alphabeta:
            return self._ab(edges, -self._inf, self._inf)
        return self._memo(edges)

    def move_values(self, edges: int) -> dict[int, int]:
        """Value of every legal move at ``edges`` (``k + V(child)`` or ``-V(child)``)."""
        out: dict[int, int] = {}
        for e in range(self.board.E):
            if (edges >> e) & 1:
                continue
            k = self._completed(edges, e)
            child = edges | (1 << e)
            out[e] = k + self.value(child) if k > 0 else -self.value(child)
        return out

    def best_moves(self, edges: int) -> list[int]:
        """All optimal moves at ``edges`` (ascending edge order)."""
        vals = self.move_values(edges)
        if not vals:
            return []
        best = max(vals.values())
        return [e for e, v in vals.items() if v == best]

    def value_table(self) -> np.ndarray:
        """``int8`` array of ``V`` for all ``2^E`` bitmasks (small boards only)."""
        E = self.board.E
        if E > 20:
            raise ValueError(f"value_table is for small boards; E={E} is too large")
        return np.array([self.value(e) for e in range(1 << E)], dtype=np.int8)

    def save_table(self, path: str | Path) -> Path:
        """Write :meth:`value_table` to ``path`` (``data/exact_<m>x<n>.npy``)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, self.value_table())
        return path

    def export_dataset(self, path: str | Path, samples: Iterable[tuple[int, int]]) -> Path:
        """Write ``(edges, first_player, V)`` arrays for ``(edges, first_player)`` samples.

        Output: ``data/exact_<m>x<n>_dataset.npz`` with keys ``edges``,
        ``first_player`` and ``V``.
        """
        rows = list(samples)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            edges=np.array([e for e, _ in rows], dtype=np.int64),
            first_player=np.array([p for _, p in rows], dtype=np.int8),
            V=np.array([self.value(e) for e, _ in rows], dtype=np.int8),
        )
        return path


_SOLVERS: dict[tuple[int, int, bool, bool], ExactSolver] = {}


def get_solver(m: int, n: int, use_symmetry: bool = True, alphabeta: bool = False) -> ExactSolver:
    """Shared solver per configuration. Values are position-only, so sharing is safe."""
    key = (m, n, use_symmetry, alphabeta)
    if key not in _SOLVERS:
        _SOLVERS[key] = ExactSolver(m, n, use_symmetry, alphabeta)
    return _SOLVERS[key]


class ExactPlayer(Player):
    """Perfect player for small boards; random tie-break among optimal moves."""

    name = "exact"

    def __init__(self, seed: int = 0, use_symmetry: bool = True, alphabeta: bool = False) -> None:
        self.rng = random.Random(seed)
        self.use_symmetry = use_symmetry
        self.alphabeta = alphabeta
        self.nodes = 0

    def choose_move(self, state: GameState, time_limit: float | None = None) -> int:
        b = state.board
        solver = get_solver(b.m, b.n, self.use_symmetry, self.alphabeta)
        start = solver.nodes
        moves = solver.best_moves(state.edges)
        self.nodes = solver.nodes - start
        if not moves:
            raise ValueError("no legal moves: board is full")
        return self.rng.choice(moves)
