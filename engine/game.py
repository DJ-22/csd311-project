"""Immutable game state and rules, including the extra-turn rule."""

from __future__ import annotations

from dataclasses import dataclass

from engine.board import Board, boxes_completed, get_board


@dataclass(frozen=True, slots=True)
class GameState:
    """Immutable game state.

    ``board`` is carried so that the rule functions know the board geometry;
    the playing state itself is ``edges``, ``scores`` and ``to_move``.
    """

    board: Board
    edges: int = 0
    scores: tuple[int, int] = (0, 0)
    to_move: int = 0


def new_game(m: int, n: int, to_move: int = 0) -> GameState:
    """Empty ``m x n`` board with ``to_move`` to play first."""
    return GameState(board=get_board(m, n), edges=0, scores=(0, 0), to_move=to_move)


def legal_moves(state: GameState) -> list[int]:
    """Indices of all undrawn edges, in ascending order."""
    edges = state.edges
    return [e for e in range(state.board.E) if not (edges >> e) & 1]


def apply_move(state: GameState, edge: int) -> tuple[GameState, int]:
    """Draw ``edge`` and return ``(new_state, boxes_completed)``.

    If the move completes at least one box the same player moves again,
    otherwise the turn passes. ``state`` is never mutated.
    """
    board = state.board
    if not 0 <= edge < board.E:
        raise ValueError(f"edge {edge} out of range [0, {board.E})")
    if (state.edges >> edge) & 1:
        raise ValueError(f"edge {edge} is already drawn")
    k = boxes_completed(state.edges, edge, board)
    new_edges = state.edges | (1 << edge)
    if k > 0:
        s0, s1 = state.scores
        scores = (s0 + k, s1) if state.to_move == 0 else (s0, s1 + k)
        return GameState(board, new_edges, scores, state.to_move), k
    return GameState(board, new_edges, state.scores, 1 - state.to_move), 0


def is_terminal(state: GameState) -> bool:
    """True when every edge has been drawn."""
    return state.edges == state.board.full
