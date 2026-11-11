"""Solver 3: heuristic search player.

Depth-limited negamax alpha-beta on the raw edge bitmask with the extra-turn rule
(CLAUDE.md §5). For a move completing ``k`` boxes into ``child``::

    value = k + V(child)    if k > 0   (same player moves again, no sign flip)
    value = -V(child)       if k == 0  (turn passes, sign flips)

and the child window is ``(alpha - k, beta - k)`` or ``(-beta, -alpha)`` to match.

At the depth cutoff the position is scored by a pluggable evaluator
``evaluator(edges, board, mover_is_first) -> float`` that estimates the same
quantity (net boxes still to be gained by the player to move). The default is
``DEFAULT_WEIGHTS · phi_edges(edges, board, mover_is_first)``. ``mover_is_first``
is tracked per node: it flips exactly when the turn passes.

Components (each can be switched off for the E2 ablation): transposition table
with EXACT / LOWER / UPPER bound flags, move ordering, quiescence over captures.
``choose_move`` runs iterative deepening under the time limit.
"""

from __future__ import annotations

import math
import random
import time
from collections.abc import Callable, Sequence

import numpy as np

from engine.board import Board
from engine.game import GameState
from features.features import NUM_FEATURES, phi_edges
from harness.metrics import effective_branching_factor
from solvers.base import Player
from solvers.baselines import capture_run

Evaluator = Callable[[int, Board, bool], float]

# One weight per feature of CLAUDE.md §7, in the same order.
DEFAULT_WEIGHTS: tuple[float, ...] = (
    0.0,  # 0 boxes with 0 sides: neutral, says nothing about who profits
    0.0,  # 1 boxes with 1 side: neutral
    0.0,  # 2 boxes with 2 sides: neutral, chain structure is covered by 5-9
    1.0,  # 3 boxes with 3 sides: the mover captures each one next
    0.0,  # 4 safe moves: only their parity matters, which a linear term cannot express
    0.0,  # 5 long chains: their count only matters through the parity term (10)
    0.0,  # 6 boxes in long chains: who gets them is decided by parity (10)
    0.0,  # 7 short chains: handed out one by one, roughly a wash
    0.0,  # 8 loops: cost the controller 4 boxes each, roughly a wash at this level
    0.0,  # 9 boxes in loops: same as 8
    1.5,  # 10 long-chain parity: the player it favours tends to take control of the long chains
    0.0,  # 11 fraction of edges drawn: game phase, no value on its own
)

DEFAULT_TIME_LIMIT = 1.0
TIME_SAFETY = 0.95  # fraction of the time limit the search may use
TT_MAX_ENTRIES = 2_000_000  # the TT is cleared when it reaches this size (4x4)
EXACT, LOWER, UPPER = 0, 1, 2  # TT bound flags


class _Timeout(Exception):
    """Raised inside the search when the deadline passes."""


class SearchPlayer(Player):
    """Iterative-deepening negamax alpha-beta with a heuristic evaluation at the cutoff."""

    name = "search"

    def __init__(
        self,
        weights: Sequence[float] | None = None,
        evaluator: Evaluator | None = None,
        max_depth: int | None = None,
        use_tt: bool = True,
        use_ordering: bool = True,
        use_quiescence: bool = True,
        seed: int = 0,
    ) -> None:
        if weights is not None and evaluator is not None:
            raise ValueError("pass either weights or evaluator, not both")
        w = np.asarray(DEFAULT_WEIGHTS if weights is None else weights, dtype=np.float32)
        if w.shape != (NUM_FEATURES,):
            raise ValueError(f"expected {NUM_FEATURES} weights, got shape {w.shape}")
        if max_depth is not None and max_depth < 1:
            raise ValueError("max_depth must be >= 1")
        self.weights = w
        self.evaluator: Evaluator = evaluator if evaluator is not None else self._linear_eval
        self.max_depth = max_depth
        self.use_tt = use_tt
        self.use_ordering = use_ordering
        self.use_quiescence = use_quiescence
        self.rng = random.Random(seed)
        self.nodes = 0  # nodes expanded for the current move (read by the harness)
        self.depth_reached = 0  # last completed iteration for the current move (read by the harness)
        self._board: Board | None = None
        self._edge_masks: tuple[tuple[int, ...], ...] = ()
        # (edges << 1 | mover_is_first) -> (depth, value, flag, best_move)
        self._tt: dict[int, tuple[int, float, int, int]] = {}
        self._deadline: float | None = None

    # evaluation
    def _linear_eval(self, edges: int, board: Board, mover_is_first: bool) -> float:
        """Default evaluator: ``weights · phi_edges(edges, board, mover_is_first)``."""
        return float(self.weights @ phi_edges(edges, board, mover_is_first))

    # search
    def _set_board(self, board: Board) -> None:
        if board is not self._board:
            self._board = board
            self._edge_masks = tuple(tuple(board.BOX_MASK[b] for b in bs) for bs in board.EDGE_BOXES)
            self._tt.clear()  # keys are only meaningful for one board geometry

    def _moves(self, edges: int, first: int = -1, rng: random.Random | None = None) -> list[tuple[int, int]]:
        """``(edge, boxes_completed)`` for every undrawn edge.

        With ``use_ordering``: captures (2-box captures first), then safe moves (no
        box becomes 3-sided), then sacrifices, with ``first`` (the TT or
        previous-iteration best move) moved to the front. ``rng`` shuffles within
        each class (root only, for variety between games). Without
        ``use_ordering``: ascending edge order.
        """
        free = self._board.full & ~edges  # type: ignore[union-attr]
        em = self._edge_masks
        if not self.use_ordering:
            plain: list[tuple[int, int]] = []
            while free:
                bit = free & -free
                free ^= bit
                e = bit.bit_length() - 1
                k = 0
                for mk in em[e]:
                    if (edges & mk).bit_count() == 3:
                        k += 1
                plain.append((e, k))
            return plain
        doubles: list[tuple[int, int]] = []
        singles: list[tuple[int, int]] = []
        safe: list[tuple[int, int]] = []
        sacrifices: list[tuple[int, int]] = []
        while free:
            bit = free & -free
            free ^= bit
            e = bit.bit_length() - 1
            k = 0
            unsafe = False
            for mk in em[e]:
                c = (edges & mk).bit_count()
                if c == 3:
                    k += 1
                elif c == 2:
                    unsafe = True
            if k == 2:
                doubles.append((e, 2))
            elif k:
                singles.append((e, 1))
            elif unsafe:
                sacrifices.append((e, 0))
            else:
                safe.append((e, 0))
        if rng is not None:
            for group in (doubles, singles, safe, sacrifices):
                rng.shuffle(group)
        out = doubles + singles + safe + sacrifices
        if first >= 0:
            for i, (e, _) in enumerate(out):
                if e == first:
                    if i:
                        out.insert(0, out.pop(i))
                    break
        return out

    def _leaf(self, edges: int, mif: bool) -> float:
        """Value at the depth cutoff: quiescence over captures, then the evaluator.

        Quiescence searches capture moves only and never declines a box (no
        double-dealing; solver 4 handles that). Captures commute: drawing an edge
        never makes a capturable box uncapturable, so every capture order ends in
        the same position with the same number of boxes taken. The capture search
        therefore collapses to one line (``capture_run``); one node is counted per
        box captured.
        """
        board = self._board
        if self.use_quiescence:
            gained, final = capture_run(edges, board)  # type: ignore[arg-type]
            if gained:
                self.nodes += gained
                if final == board.full:  # type: ignore[union-attr]
                    return float(gained)
                return gained + self.evaluator(final, board, mif)  # type: ignore[arg-type]
        return self.evaluator(edges, board, mif)  # type: ignore[arg-type]

    def _search(self, edges: int, depth: int, alpha: float, beta: float, mif: bool) -> float:
        """Fail-soft negamax alpha-beta value of ``edges`` for the player to move.

        ``mif``: the player to move is the game's first player. Exact whenever
        ``alpha < value < beta``; otherwise a bound on the correct side.
        """
        self.nodes += 1
        if self._deadline is not None and not self.nodes & 127 and time.perf_counter() > self._deadline:
            raise _Timeout
        board = self._board
        if edges == board.full:  # type: ignore[union-attr]
            return 0.0
        free = board.E - edges.bit_count()  # type: ignore[union-attr]
        if depth > free:  # the subtree is searched to the end: the value is exact
            depth = free
        key = (edges << 1) | mif
        tt_move = -1
        if self.use_tt:
            entry = self._tt.get(key)
            if entry is not None:
                d, v, flag, mv = entry
                if d >= depth and (
                    flag == EXACT or (flag == LOWER and v >= beta) or (flag == UPPER and v <= alpha)
                ):
                    return v
                tt_move = mv
        if depth == 0:
            return self._leaf(edges, mif)
        alpha0 = alpha
        best, best_move = -math.inf, -1
        for e, k in self._moves(edges, tt_move):
            child = edges | (1 << e)
            if k:  # same player continues: shift the window, no negation
                val = k + self._search(child, depth - 1, alpha - k, beta - k, mif)
            else:  # turn passes: negate and swap the window
                val = -self._search(child, depth - 1, -beta, -alpha, not mif)
            if val > best:
                best, best_move = val, e
            if val > alpha:
                alpha = val
            if alpha >= beta:
                break
        if self.use_tt:
            flag = UPPER if best <= alpha0 else LOWER if best >= beta else EXACT
            if len(self._tt) >= TT_MAX_ENTRIES:
                self._tt.clear()
            self._tt[key] = (depth, best, flag, best_move)
        return best

    def search_root(
        self, edges: int, depth: int, mover_is_first: bool, board: Board, first_move: int = -1
    ) -> tuple[int, float]:
        """Search ``edges`` to ``depth`` with a full window.

        Returns ``(best_move, value)`` for the player to move; ``mover_is_first``
        says whether that player moved first in the game. ``first_move`` (e.g. the
        previous iteration's best move) is searched first when ordering is on.
        The TT persists between calls until :meth:`reset`.
        """
        if edges == board.full:
            raise ValueError("no legal moves: board is full")
        self._set_board(board)
        depth = max(1, min(depth, board.E - edges.bit_count()))
        alpha, beta = -math.inf, math.inf
        best, best_move = -math.inf, -1
        if self.use_ordering:
            moves = self._moves(edges, first_move, self.rng)
        else:
            moves = self._moves(edges)
        for e, k in moves:
            child = edges | (1 << e)
            if k:
                val = k + self._search(child, depth - 1, alpha - k, beta - k, mover_is_first)
            else:
                val = -self._search(child, depth - 1, -beta, -alpha, not mover_is_first)
            if val > best:
                best, best_move = val, e
            if val > alpha:
                alpha = val
        return best_move, best

    # Player API
    def choose_move(self, state: GameState, time_limit: float | None = None) -> int:
        """Iterative deepening ``D = 1, 2, ...``; returns the best move of the last completed depth.

        Stops when the next iteration is predicted (last iteration's time x its
        ``b*``) to overrun the time limit, and aborts an iteration that does. Depth 1
        always completes, so ``depth_reached >= 1``. With neither ``time_limit``
        nor ``max_depth`` the default 1 s limit applies.
        """
        self.nodes = 0
        self.depth_reached = 0
        board = state.board
        edges = state.edges
        if edges == board.full:
            raise ValueError("no legal moves: board is full")
        mif = state.to_move == 0  # the harness always starts games with seat 0
        free = board.E - edges.bit_count()
        max_depth = free if self.max_depth is None else min(self.max_depth, free)
        if time_limit is None and self.max_depth is None:
            time_limit = DEFAULT_TIME_LIMIT
        start = time.perf_counter()
        deadline = None if time_limit is None else start + TIME_SAFETY * time_limit
        best_move = -1
        for depth in range(1, max_depth + 1):
            n0, t0 = self.nodes, time.perf_counter()
            self._deadline = deadline if depth > 1 else None
            try:
                best_move, _ = self.search_root(edges, depth, mif, board, best_move)
            except _Timeout:
                break
            finally:
                self._deadline = None
            self.depth_reached = depth
            if deadline is not None and depth < max_depth:
                now = time.perf_counter()
                bstar = effective_branching_factor(self.nodes - n0, depth)
                if now + (now - t0) * bstar > deadline:
                    break
        return best_move

    def reset(self) -> None:
        """Clear the TT and the per-move counters."""
        self._tt.clear()
        self.nodes = 0
        self.depth_reached = 0
