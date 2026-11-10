"""Shared feature vector ``phi``. Append new features at the end only; never reorder."""

from __future__ import annotations

import numpy as np

from engine.board import Board
from engine.chains import LONG_CHAIN, find_components
from engine.game import GameState

FEATURE_NAMES: tuple[str, ...] = (
    "boxes_0_sides",
    "boxes_1_side",
    "boxes_2_sides",
    "boxes_3_sides",
    "safe_moves",
    "long_chains",
    "long_chain_boxes",
    "short_chains",
    "loops",
    "loop_boxes",
    "parity_favours_mover",
    "fraction_edges_drawn",
)
NUM_FEATURES = len(FEATURE_NAMES)


def phi_edges(edges: int, board: Board, mover_is_first: bool) -> np.ndarray:
    """``phi`` on the raw bitmask. ``mover_is_first``: the player to move moved first in the game."""
    mask = board.BOX_MASK
    sides = [(edges & mk).bit_count() for mk in mask]
    counts = [0, 0, 0, 0, 0]
    for s in sides:
        counts[s] += 1

    # Safe move: an undrawn edge that turns no box 3-sided (no adjacent 2-sided box).
    safe = 0
    for e in range(board.E):
        if not (edges >> e) & 1 and all(sides[b] != 2 for b in board.EDGE_BOXES[e]):
            safe += 1

    long_chains = long_boxes = short_chains = loops = loop_boxes = 0
    for c in find_components(edges, board):
        if c.kind == "loop":
            loops += 1
            loop_boxes += c.length
        elif c.length >= LONG_CHAIN:
            long_chains += 1
            long_boxes += c.length
        else:
            short_chains += 1

    # Long chain rule [outside] (Berlekamp): first player wants dots + long chains even.
    even = (board.dots + long_chains) % 2 == 0
    parity = 1.0 if even == mover_is_first else -1.0

    return np.array(
        [
            counts[0],
            counts[1],
            counts[2],
            counts[3],
            safe,
            long_chains,
            long_boxes,
            short_chains,
            loops,
            loop_boxes,
            parity,
            edges.bit_count() / board.E,
        ],
        dtype=np.float32,
    )


def phi(state: GameState, first_player: int) -> np.ndarray:
    """Feature vector of ``state`` from the perspective of the player to move.

    ``first_player`` is the player (0 or 1) who made the first move of the game.
    """
    return phi_edges(state.edges, state.board, state.to_move == first_player)
