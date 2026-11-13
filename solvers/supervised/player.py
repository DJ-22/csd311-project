"""Solver 7 player: a learned evaluator used with one-ply lookahead, or inside solver 3.

One-ply lookahead with the extra-turn rule::

    score(a) = k + V̂(s')   if move a completes k > 0 boxes (same player moves again)
    score(a) = -V̂(s')      otherwise (turn passes)

``V̂`` is :meth:`LearnedModel.value`: the model's score in ``[-1, 1]`` times the
boxes remaining, i.e. an estimate of net boxes for the player to move (CLAUDE.md §5).
Ties are broken with the seeded ``rng``.

``lookahead > 1`` (and :func:`search_learned`) hand ``V̂`` to solver 3's
``SearchPlayer`` as its ``evaluator``: variant ``search-learned``.
"""

from __future__ import annotations

import random
from pathlib import Path

from engine.game import GameState
from solvers.base import Player
from solvers.search import SearchPlayer
from solvers.supervised.models import LearnedModel, load_model, one_ply_scores

_CACHE: dict[str, LearnedModel] = {}


def _load(model_path: str | Path) -> LearnedModel:
    key = str(Path(model_path).resolve())
    if key not in _CACHE:
        _CACHE[key] = load_model(model_path)
    return _CACHE[key]


class LearnedPlayer(Player):
    """Plays the argmax of the learned one-ply score."""

    def __init__(self, model_path: str | Path, lookahead: int = 1, seed: int = 0) -> None:
        if lookahead < 1:
            raise ValueError("lookahead must be >= 1")
        self.model = _load(model_path)
        self.lookahead = lookahead
        self.rng = random.Random(seed)
        self.name = f"learned-{self.model.label}"
        self._search = (
            SearchPlayer(evaluator=self.model.value, max_depth=lookahead, seed=seed) if lookahead > 1 else None
        )
        self.nodes = 0
        self.depth_reached = 0

    def choose_move(self, state: GameState, time_limit: float | None = None) -> int:
        board = state.board
        if (board.m, board.n) != (self.model.m, self.model.n):
            raise ValueError(
                f"model was trained on {self.model.m}x{self.model.n}, game is {board.m}x{board.n}"
            )
        if self._search is not None:
            move = self._search.choose_move(state, time_limit)
            self.nodes, self.depth_reached = self._search.nodes, self._search.depth_reached
            return move
        # The harness always starts games with seat 0, so seat 0 is the first player.
        scores = one_ply_scores(self.model, state.edges, state.to_move, 0, board)
        if not scores:
            raise ValueError("no legal moves: board is full")
        best = max(scores.values())
        moves = [e for e, s in scores.items() if s >= best - 1e-9]
        self.nodes = len(scores)
        return self.rng.choice(moves)

    def reset(self) -> None:
        self.nodes = 0
        self.depth_reached = 0
        if self._search is not None:
            self._search.reset()


def search_learned(model_path: str | Path, seed: int = 0, **search_kwargs: object) -> SearchPlayer:
    """Solver 3 with the learned ``V̂`` as its evaluator (``search-learned``)."""
    player = SearchPlayer(evaluator=_load(model_path).value, seed=seed, **search_kwargs)  # type: ignore[arg-type]
    player.name = "search-learned"
    return player
