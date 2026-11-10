"""Player API shared by every solver."""

from __future__ import annotations

from abc import ABC, abstractmethod

from engine.game import GameState


class Player(ABC):
    """A Dots-and-Boxes player.

    Searchers additionally expose an integer ``nodes`` attribute, reset at the
    start of every ``choose_move`` call, which the harness records.
    """

    name: str

    @abstractmethod
    def choose_move(self, state: GameState, time_limit: float | None = None) -> int:
        """Return a legal edge index for ``state`` (never mutate ``state``)."""

    def reset(self) -> None:
        """Clear per-game caches. Called before each game."""
