"""Solver 1: Random and Greedy baselines."""

from __future__ import annotations

import random

from engine.board import Board
from engine.game import GameState, legal_moves
from solvers.base import Player


def capture_run(edges: int, board: Board) -> tuple[int, int]:
    """Repeatedly capture any capturable box. Returns ``(boxes_captured, new_edges)``.

    The set of boxes captured is independent of capture order, because drawing an
    edge never makes a capturable box uncapturable.
    """
    mask = board.BOX_MASK
    edge_boxes = board.EDGE_BOXES
    captured = 0
    stack = [b for b in range(board.num_boxes) if (edges & mask[b]).bit_count() == 3]
    while stack:
        b = stack.pop()
        missing = mask[b] & ~edges
        if missing == 0 or missing & (missing - 1):
            continue    # already taken, or no longer exactly one side missing
        edges |= missing
        e = missing.bit_length() - 1
        for x in edge_boxes[e]:
            drawn = (edges & mask[x]).bit_count()
            if drawn == 4:
                captured += 1
            elif drawn == 3:
                stack.append(x)
    return captured, edges


def greedy_move(edges: int, rng: random.Random, board: Board) -> int:
    """Greedy policy on the raw bitmask (also the MCTS rollout policy).

    1. Capture: a move completing the most boxes (2 before 1), random tie-break.
    2. Safe move: a random move that creates no 3-sided box.
    3. Least-damage sacrifice: the move after which the opponent's capture run is
       shortest, random tie-break.
    """
    mask = board.BOX_MASK
    edge_boxes = board.EDGE_BOXES
    sides = [(edges & mk).bit_count() for mk in mask]

    best_k = 0
    captures: list[int] = []
    safe: list[int] = []
    free: list[int] = []
    for e in range(board.E):
        if (edges >> e) & 1:
            continue
        free.append(e)
        k = 0
        unsafe = False
        for b in edge_boxes[e]:
            s = sides[b]
            if s == 3:
                k += 1
            elif s == 2:
                unsafe = True
        if k > 0:
            if k > best_k:
                best_k, captures = k, [e]
            elif k == best_k:
                captures.append(e)
        elif not unsafe:
            safe.append(e)

    if captures:
        return rng.choice(captures)
    if safe:
        return rng.choice(safe)
    if not free:
        raise ValueError("no legal moves: board is full")

    best_cost = board.num_boxes + 1
    best: list[int] = []
    for e in free:
        cost, _ = capture_run(edges | (1 << e), board)
        if cost < best_cost:
            best_cost, best = cost, [e]
        elif cost == best_cost:
            best.append(e)
    return rng.choice(best)


class RandomPlayer(Player):
    """Uniformly random legal move from a seeded RNG."""

    name = "random"

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)
        self.nodes = 0

    def choose_move(self, state: GameState, time_limit: float | None = None) -> int:
        return self.rng.choice(legal_moves(state))


class GreedyPlayer(Player):
    """Captures everything, plays safe when it can, otherwise sacrifices least."""

    name = "greedy"

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)
        self.nodes = 0

    def choose_move(self, state: GameState, time_limit: float | None = None) -> int:
        return greedy_move(state.edges, self.rng, state.board)
